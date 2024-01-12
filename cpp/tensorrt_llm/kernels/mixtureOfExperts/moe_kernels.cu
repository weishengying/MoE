/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeSoftmax(const T* input, const bool* finished, T* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = T(val);
    }
}

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(const T* inputs_after_softmax, const bool* finished, T* output,
    int* indices, int* source_rows, const int num_experts, const int k, const int start_expert, const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = T(-1.f); // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
  
  WARPS_PER_CTA 其实就是每个 thread block 中的线程束个数，在该代码中，等同于 WARPS_PER_TB
  "CAT" 通常指的是 Cooperative Thread Arrays （合作线程数组），即一个 thread block

*/

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const T* input, const bool* finished, T* output, const int num_rows, int* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT; // 每行需要 THREADS_PER_ROW 个线程来处理
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG; // 每个线程需要执行 LDG_PER_THREAD 次 load 操作

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT; // 每个 warp 需要处理的元素个数
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW; // 每个 warp 需要处理的行数
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP; // 一个thread block 能够 处理的行数

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP; // block_dim (WARP_SIZE, WARPS_PER_TB) == (32, 4), 因此 threadIdx.y 表示第几个 warps

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW; 
    const int thread_row = warp_base_row + thread_row_in_warp; // 当前线程需要处理的对应行

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW; // thread.x 表示 thread block 中的 warp 中的线程，这样计算出来，不同 warp 中的 thread 需要计算对应行中的对应列
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    cutlass::Array<T, VPT> row_chunk_input; // 每个线程需要处理 VPT 个元素
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_input);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) // 将全局内存中的 input 数据，加载到 row_chunk_input 来
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    using ComputeType = float;
    using Converter = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
    Converter compute_type_converter;
    cutlass::Array<ComputeType, VPT> row_chunk = compute_type_converter(row_chunk_input); // 把加载进来的输入数据，转换为 float

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    ComputeType thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float total_scale = 0.0;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = T(max_val);
            total_scale += max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = ComputeType(-10000.f);
            }
        }
    }
    // 这里增加除和归一化
    if (thread_group_idx == 0) {
        float inv_total_scale = 1.0f / total_scale;
#pragma unroll
        for (int k_idx = 0; k_idx < k; k_idx++) {
            const int idx = k * thread_row + k_idx;
            const float format_scale = float(output[idx]) * inv_total_scale;
            output[idx] = T(format_scale);
        }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    /*
    读懂这里的常量需要理解算法的设计思路：
    1. 算法处理的输入shape 为 (num_rows, num_experts)
    2. 以线程束 warp 为单位，一个 warp 处理几行，具体多少行需要计算
    3. 向量化
    Here, for every row, there are EXPERTS elements, and ELTS_PER_LDG of elements make up a vector, so the total vecs number is  EXPERTS / ELTS_PER_LDG
    The orginal algo design is "all threads in a warp responsible for a row", but if we found the EXPERTS of a row is not enough, a warp will responsible more rows 
    and a thread in a warp will handle only one vec
    */
    // LDG typically refers to "load global memory".
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T); // 一次内存加载操作可以加载的元素个数，如果类型是 half 或者 bfloat16, 则每次可以加载 8 个元素
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, ""); // WARP_SIZE = 32
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE)); // 每一行有 EXPERTS 个元素，EXPERTS/WARP_SIZE，可以得到，平均每个线程需要处理的元素个数，再除以ELTS_PER_LDG，就是每个线程需要处理的向量个数
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG; //This gives the total number of elements (or vectors, depending on how you interpret it) processed by a single thread.
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT; // 每一行需要 THREADS_PER_ROW 个 thread 来处理
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW; // 一个线程束 wrap 可以处理 ROWS_PER_WARP 行
};
} // namespace detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(const T* input, const bool* finished, T* output, int* indices, int* source_row,
    const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16; // 每次从全局内存中加载 16 个字节

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
    using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP; //一行需要 ROWS_PER_WARP 个 warp 来处理，所以一共需要 num_warps 个 wrap
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB; //一个 thread block 中最多有 WARPS_PER_TB 个 warp， 所以需要 num_blocks 个 thread block

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB); //一个 thread block 中有 WARPS_PER_TB 个 warp
    topkGatingSoftmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert);
}
/*
input 是 gating 之后的输出， shape 为 (num_rows, num_experts), 该 kernel 实现的逻辑等同于下面 python 代码：
```python
expert_scales = F.softmax(input_dict["gating_output"], dim=-1）
expert_scales, experts_for_row = torch.topk(expert_scales, k, dim=-1)
expert_scales /= expert_scales.sum(dim=-1, keepdim=True)
```
*/
// finished 一般传入空指针，表示所有的行都完成，需要被处理
template <typename T> // T 表示 input 的数据类型
void topkGatingSoftmaxKernelLauncher(const T* input, const bool* finished, T* output, T* softmax_temp_output,
    int* indices, int* source_row, const int num_rows, const int num_experts, const int k, const int start_expert,
    const int end_expert, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4; //每个thread block 有 4 个 warps

    switch (num_experts)
    {
    case 1:
    {
        topkGatingSoftmaxLauncherHelper<T, 1, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 2:
    {
        topkGatingSoftmaxLauncherHelper<T, 2, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 4:
    {
        topkGatingSoftmaxLauncherHelper<T, 4, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 8:
    {
        topkGatingSoftmaxLauncherHelper<T, 8, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 16:
    {
        topkGatingSoftmaxLauncherHelper<T, 16, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 32:
    {
        topkGatingSoftmaxLauncherHelper<T, 32, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 64:
    {
        topkGatingSoftmaxLauncherHelper<T, 64, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 128:
    {
        topkGatingSoftmaxLauncherHelper<T, 128, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    case 256:
    {
        topkGatingSoftmaxLauncherHelper<T, 256, WARPS_PER_TB>(
            input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
        break;
    }
    default:
    {
        static constexpr int TPB = 256;
        TLLM_CHECK(softmax_temp_output != nullptr);
        moeSoftmax<T, TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
        moeTopK<T, TPB><<<num_rows, TPB, 0, stream>>>(
            softmax_temp_output, finished, output, indices, source_row, num_experts, k, start_expert, end_expert);
    }
    }
}

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0)
    , num_bits_(sizeof(int) * 8)
{
}

CubKeyValueSorter::CubKeyValueSorter(const int num_experts)
    : num_experts_(num_experts)
    , num_bits_((int) log2(num_experts) + 1)
{
}

void CubKeyValueSorter::updateNumExperts(const int num_experts)
{
    num_experts_ = num_experts;
    num_bits_ = (int) log2(num_experts) + 1;
}

size_t CubKeyValueSorter::getWorkspaceSize(const size_t num_key_value_pairs, const int num_experts)
{
    size_t num_bits = (int) log2(num_experts) + 1;
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        NULL, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);
    return required_storage;
}

void CubKeyValueSorter::run(void* workspace, const size_t workspace_size, const int* keys_in, int* keys_out,
    const int* values_in, int* values_out, const size_t num_key_value_pairs, cudaStream_t stream)
{
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size = workspace_size;

    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
        "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
__device__ inline int findTotalEltsLeqTarget(const int* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Sets up the gemm assuming the inputs, experts and outputs are stored in row major order.
// Assumes we want to perform output = matmul(inputs, experts) + bias
//
// "total_rows_before_expert" contains the index one past the last occurrence of the corresponding expert.
// e.g. Index 0 is the start offset of expert 1, the final entry is the total number of active rows
__global__ void computeTotalRowsBeforeExpertKernel(const int* sorted_experts, const int sorted_experts_len,
    const int64_t num_experts, int64_t* total_rows_before_expert)
{
    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = findTotalEltsLeqTarget(sorted_experts, sorted_experts_len, expert);
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(const T* unpermuted_input, T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    const int num_rows, const int64_t* num_dest_rows, const int cols)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    const int expanded_dest_row = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
    {
        // Duplicate and permute rows
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
        {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

template <typename T>
void expandInputRowsKernelLauncher(const T* unpermuted_input, T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    const int num_rows, const int64_t* num_valid_tokens_ptr, const int cols, const int k, cudaStream_t stream)
{
    const int blocks = num_rows * k;
    const int threads = std::min(cols, 1024);
    auto func = (num_valid_tokens_ptr != nullptr) ? expandInputRowsKernel<T, true> : expandInputRowsKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output, expanded_dest_row_to_expanded_source_row,
        expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, cols);
}

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, int RESIDUAL_NUM, bool HAS_BIAS, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(const T* expanded_permuted_rows, T* reduced_unpermuted_output, const T* skip_1,
    const T* skip_2, const T* bias, const T* scales, const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row, const int cols, const int k, const int64_t* num_valid_ptr)
{
    const int original_row = blockIdx.x;
    const int num_rows = gridDim.x;
    const auto offset = original_row * cols;
    T* reduced_row_ptr = reduced_unpermuted_output + offset;
    const T* skip_1_row_ptr{};
    const T* skip_2_row_ptr{};

    if (RESIDUAL_NUM >= 1)
    {
        skip_1_row_ptr = skip_1 + offset;
    }

    if (RESIDUAL_NUM == 2)
    {
        skip_2_row_ptr = skip_2 + offset;
    }
    const int64_t num_valid = *num_valid_ptr;
    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
    {
        T thread_output{0.f};
        T row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            const int expanded_original_row = original_row + k_idx * num_rows;
            const int expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            const int64_t k_offset = original_row * k + k_idx;
            const T row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? T(1.f) : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row sum has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            const T* expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];

            const T* bias_ptr = bias + expert_idx * cols;
            const T bias_value = HAS_BIAS ? bias_ptr[tid] : T(0.f);

            thread_output = thread_output + row_scale * (expanded_permuted_rows_row_ptr[tid] + bias_value);
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || thread_output))
        {
            assert(row_rescale != T(0.f));
            thread_output = thread_output / row_rescale;
        }

        if (RESIDUAL_NUM == 1)
        {
            thread_output = thread_output + skip_1_row_ptr[tid];
        }
        else if (RESIDUAL_NUM == 2)
        {
            thread_output = thread_output + skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
        reduced_row_ptr[tid] = thread_output;
    }
}

template <typename T, int RESIDUAL_NUM>
void finalizeMoeRoutingKernelLauncherSelectBias(const T* expanded_permuted_rows, T* reduced_unpermuted_output,
    const T* skip_1, const T* skip_2, const T* bias, const T* scales,
    const int* expanded_source_row_to_expanded_dest_row, const int* expert_for_source_row, const int num_rows,
    const int cols, const int k, const int64_t* num_valid_ptr, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    const int blocks = num_rows;
    const int threads = std::min(cols, 1024);

    // Only add bias on rank 0 for tensor parallelism
    const bool is_rank_0 = parallelism_config.tp_rank == 0;
    const bool has_bias = bias != nullptr && is_rank_0;

    const bool check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3][2]
        = {{
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::NO_SCALE, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::NO_SCALE, false>},
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::DEFAULT, false>},
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::RENORM_SCALE, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::RENORM_SCALE, false>},
           },
            {
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::NO_SCALE, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::NO_SCALE, true>},
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::DEFAULT, true>},
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::RENORM_SCALE, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::RENORM_SCALE, true>},
            }};
    auto* const func = func_map[check_finished][int(renorm_scales)][has_bias];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, skip_1, skip_2, bias,
        scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

template <typename T>
void finalizeMoeRoutingKernelLauncher(const T* expanded_permuted_rows, T* reduced_unpermuted_output, const T* skip_1,
    const T* skip_2, const T* bias, const T* scales, const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row, const int num_rows, const int cols, const int k, const int64_t* num_valid_ptr,
    MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    // If we are not rank 0 we should not add any residuals because the allreduce would sum multiple copies
    const bool is_rank_0 = parallelism_config.tp_rank == 0;
    if (skip_1 == nullptr || !is_rank_0)
    {
        assert(skip_2 == nullptr);
        finalizeMoeRoutingKernelLauncherSelectBias<T, 0>(expanded_permuted_rows, reduced_unpermuted_output, skip_1,
            skip_2, bias, scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, cols, k,
            num_valid_ptr, parallelism_config, normalization_mode, stream);
    }
    else if (skip_2 == nullptr)
    {
        finalizeMoeRoutingKernelLauncherSelectBias<T, 1>(expanded_permuted_rows, reduced_unpermuted_output, skip_1,
            skip_2, bias, scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, cols, k,
            num_valid_ptr, parallelism_config, normalization_mode, stream);
    }
    else
    {
        finalizeMoeRoutingKernelLauncherSelectBias<T, 2>(expanded_permuted_rows, reduced_unpermuted_output, skip_1,
            skip_2, bias, scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, cols, k,
            num_valid_ptr, parallelism_config, normalization_mode, stream);
    }
}

// ============================== Gated Activation =================================

template <class T, class ActFn>
__global__ void doGatedActivationKernel(
    T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, size_t inter_size)
{
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    ActFn fn{};
    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;
    for (int i = tid; i < inter_size; i += blockDim.x)
    {
        T fc1_value = gemm_result[i];
        // BF16 isn't supported, use FP32 for activation function
        float gate_value = gemm_result[i + inter_size];
        T gate_act = fn(gate_value);
        output[i] = fc1_value * gate_act;
    }
}

template <class T>
void doGatedActivation(T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, int inter_size,
    int num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    const int blocks = num_tokens;
    const int threads = std::min(inter_size, 1024);

    // TODO Instead of T use a vectored type if performance would benefit
    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>
        : &doGatedActivationKernel<T, cutlass::epilogue::thread::GELU<float>>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

template <typename T, typename WeightType, typename Enable>
size_t CutlassMoeFCRunner<T, WeightType, Enable>::getWorkspaceSize(const int num_rows, const int hidden_size,
    const int inter_size, const int num_experts, const int k, ActivationType activation_type,
    MOEParallelismConfig parallelism_config) const
{
    const int ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of tp size");
    const int buf_size = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int glu_inter_size
        = isGatedActivation(activation_type) ? pad_to_multiple_of_16(k * num_rows * inter_size * 2) : 0;
    const int interbuf_size = pad_to_multiple_of_16(k * num_rows * inter_size) + glu_inter_size;
    const int padded_experts = pad_to_multiple_of_16(num_experts / ep_size);
    const int num_moe_inputs = pad_to_multiple_of_16(k * num_rows);
    int num_softmax_outs = 0;

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        num_softmax_outs = pad_to_multiple_of_16(num_rows * num_experts);
    }

    // softmax output, permuted_rows and permuted_experts have moved to outside of moe kernel, allocate them
    // in Encoder or Decoder before invoking FfnLayer forward.
    size_t total_ws_bytes = 3 * num_moe_inputs * sizeof(int); // source_rows_, permuted_rows_, permuted_experts_
    total_ws_bytes += buf_size * sizeof(T);                   // permuted_data
    total_ws_bytes += padded_experts * sizeof(int64_t);       // Hold total_rows_before_expert_
    total_ws_bytes += num_softmax_outs * sizeof(T);
    const int bytes_for_fc1_result = interbuf_size * sizeof(T);
    const int sorter_ws_size_bytes = pad_to_multiple_of_16(CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts));

    int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
    if (sorter_ws_size_bytes > bytes_for_fc1_result)
    {
        int remaining_bytes = pad_to_multiple_of_16(sorter_ws_size_bytes - bytes_for_fc1_result);
        bytes_for_intermediate_and_sorting += remaining_bytes;
    }

    total_ws_bytes += bytes_for_intermediate_and_sorting; // intermediate (fc1) output + cub sorting workspace
    return total_ws_bytes;
}

template <typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::configureWsPtrs(char* ws_ptr, const int num_rows, const int hidden_size,
    const int inter_size, const int num_experts, const int num_experts_per_node, const int k,
    ActivationType activation_type)
{
    const int buf_size = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int glu_inter_size
        = isGatedActivation(activation_type) ? pad_to_multiple_of_16(k * num_rows * inter_size * 2) : 0;
    const int interbuf_size = pad_to_multiple_of_16(k * num_rows * inter_size) + glu_inter_size;
    const int sorter_ws_size_bytes = pad_to_multiple_of_16(CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts));
    const size_t interbuf_sorter_size_bytes = std::max(interbuf_size * sizeof(T), (size_t) sorter_ws_size_bytes);
    const int padded_experts = pad_to_multiple_of_16(num_experts_per_node);
    const int num_moe_inputs = pad_to_multiple_of_16(k * num_rows);

    /*
    [source_rows][permuted_rows][permuted_experts][permuted_data][total_rows_before_expert][glu_inter_result][fc1_result]
    among them, the length of each field is:
    source_rows = pad_to_multiple_of_16(k * num_rows)
    permuted_rows = pad_to_multiple_of_16(k * num_rows)
    permuted_expert = pad_to_multiple_of_16(k * num_rows)
    permuted_data = pad_to_multiple_of_16(k * num_rows * hidden_size);
    total_rows_before_expert = pad_to_multiple_of_16(num_experts_per_node);
    glu_inter_result = pad_to_multiple_of_16(k * num_rows * inter_size * 2)
    */
    source_rows_ = (int*) ws_ptr;
    permuted_rows_ = source_rows_ + num_moe_inputs;
    permuted_experts_ = permuted_rows_ + num_moe_inputs;
    permuted_data_ = (T*) (permuted_experts_ + num_moe_inputs);

    total_rows_before_expert_ = (int64_t*) (permuted_data_ + buf_size);

    // These pointers are aliased. Since the sort ws can be overwritten after it is finished
    glu_inter_result_ = (T*) (total_rows_before_expert_ + padded_experts);
    fc1_result_ = glu_inter_result_ + glu_inter_size;
    sorter_ws_ = (char*) glu_inter_result_;

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        softmax_out_ = (T*) (sorter_ws_ + interbuf_sorter_size_bytes);
    }
    else
    {
        softmax_out_ = nullptr;
    }
}

template <typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::runMoe(const void* input_activations_void,
    const void* gating_output_void, const void* fc1_expert_weights_void, const void* fc1_scales_void,
    const void* fc1_expert_biases_void, ActivationType fc1_activation_type, const void* fc2_expert_weights_void,
    const void* fc2_scales_void, const void* fc2_expert_biases_void, const int num_rows, const int hidden_size,
    const int inter_size, const int num_experts, const int k, char* workspace_ptr, void* final_output_void,
    void* fc2_result_void, const bool* finished, const int active_rows, void* expert_scales_void,
    int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    static constexpr bool scales_required
        = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;

    auto* input_activations = static_cast<const T*>(input_activations_void);
    auto* gating_output = static_cast<const T*>(gating_output_void);
    auto* fc1_expert_weights = static_cast<const WeightType*>(fc1_expert_weights_void);
    auto* fc1_scales = static_cast<const T*>(fc1_scales_void);
    auto* fc1_expert_biases = static_cast<const T*>(fc1_expert_biases_void);
    auto* fc2_expert_weights = static_cast<const WeightType*>(fc2_expert_weights_void);
    auto* fc2_scales = static_cast<const T*>(fc2_scales_void);
    auto* fc2_expert_biases = static_cast<const T*>(fc2_expert_biases_void);
    auto* final_output = static_cast<T*>(final_output_void);
    auto* fc2_result = static_cast<T*>(fc2_result_void);
    auto* expert_scales = static_cast<T*>(expert_scales_void);

    TLLM_CHECK(input_activations);
    TLLM_CHECK(gating_output);
    TLLM_CHECK(fc1_expert_weights);
    TLLM_CHECK(fc2_expert_weights);
    TLLM_CHECK(workspace_ptr);
    TLLM_CHECK(fc2_result);
    TLLM_CHECK(expert_scales);
    TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
    TLLM_CHECK(expert_for_source_row);
    TLLM_CHECK(num_experts % parallelism_config.ep_size == 0);

    if (scales_required)
    {
        TLLM_CHECK_WITH_INFO(fc1_scales != nullptr, "Scales expected but scale for first matmul is a null pointer");
        TLLM_CHECK_WITH_INFO(fc2_scales != nullptr, "Scales expected but scale for second matmul is a null pointer");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(fc1_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received scale for FC1");
        TLLM_CHECK_WITH_INFO(fc2_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received scale for FC2");
    }

    const int num_experts_per_node = num_experts / parallelism_config.ep_size;
    const int start_expert = num_experts_per_node * parallelism_config.ep_rank;
    const int end_expert = start_expert + num_experts_per_node;

    configureWsPtrs(
        workspace_ptr, num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, fc1_activation_type);
    topkGatingSoftmaxKernelLauncher<T>(gating_output, finished, expert_scales, softmax_out_, expert_for_source_row,
        source_rows_, num_rows, num_experts, k, start_expert, end_expert, stream);

    sync_check_cuda_error();
    // printToScreen(expert_scales, num_rows * k);

    sorter_.updateNumExperts(num_experts);
    const int sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(k * num_rows, num_experts));
    sorter_.run((void*) sorter_ws_, sorter_ws_size_bytes, expert_for_source_row, permuted_experts_, source_rows_,
        permuted_rows_, k * num_rows, stream);

    // sync_check_cuda_error();

    // Upper bound on number of expanded rows
    const int expanded_active_expert_rows = k * active_rows;
    computeTotalRowsBeforeExpert(
        permuted_experts_, expanded_active_expert_rows, num_experts_per_node, total_rows_before_expert_, stream);

    // sync_check_cuda_error();

    const bool needs_num_valid = finished || parallelism_config.ep_size > 1;
    const int64_t* num_valid_tokens_ptr
        = needs_num_valid ? total_rows_before_expert_ + num_experts_per_node - 1 : nullptr;
    expandInputRowsKernelLauncher(input_activations, permuted_data_, permuted_rows_,
        expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, hidden_size, k, stream);

    // sync_check_cuda_error();

    if (!isGatedActivation(fc1_activation_type))
    {
        moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_scales, nullptr, fc1_result_,
            total_rows_before_expert_, expanded_active_expert_rows, inter_size, hidden_size, num_experts_per_node,
            fc1_activation_type, stream);
    }
    else
    {
        const size_t fc1_out_size = inter_size * 2;
        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_scales, nullptr,
            glu_inter_result_, total_rows_before_expert_, expanded_active_expert_rows, fc1_out_size, hidden_size,
            num_experts_per_node, ActivationType::Identity, stream);

        // sync_check_cuda_error();

        // printf("glu_inter_result_:\n");
        // printToScreen<T>(glu_inter_result_, k*num_rows*fc1_out_size);

        doGatedActivation<T>(
            fc1_result_, glu_inter_result_, num_valid_tokens_ptr, inter_size, num_rows*k, fc1_activation_type, stream);
        
        // printf("fc1_result_:\n");
        // printToScreen<T>(fc1_result_, k*num_rows*inter_size);
    }
    // sync_check_cuda_error();


    moe_gemm_runner_.moeGemm(fc1_result_, fc2_expert_weights, fc2_scales, fc2_result, total_rows_before_expert_,
        expanded_active_expert_rows, hidden_size, inter_size, num_experts_per_node, stream);

    // sync_check_cuda_error();

    // printf("fc2_result:\n");
    //     printToScreen<T>(fc2_result, k*num_rows*hidden_size);

    finalizeMoeRoutingKernelLauncher<T>(fc2_result, final_output,
        // TODO pass 'skip' connections (residuals)
        nullptr, nullptr, nullptr, expert_scales, expanded_source_row_to_expanded_dest_row,
        expert_for_source_row, num_rows, hidden_size, k, num_valid_tokens_ptr, parallelism_config, normalization_mode,
        stream);

    // sync_check_cuda_error();
}

template <class T, class WeightType, class Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::computeTotalRowsBeforeExpert(const int* sorted_indices,
    const int total_indices, const int num_experts, int64_t* total_rows_before_expert, cudaStream_t stream)
{
    const int threads = std::min(1024, num_experts);
    const int blocks = (num_experts + threads - 1) / threads;

    computeTotalRowsBeforeExpertKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

// ==================== Helper for getting load balanced routing for profiling ==================================

template <class T>
__global__ void initRoutingKernelDiagonal(void* data_void, int num_experts, int num_tokens, int k, int stride)
{
    assert(k == 1 || (stride % num_experts) != 0);
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens)
    {
        return;
    }
    T* data = (T*) data_void + token * num_experts;
    int start = token % num_experts;
    for (int i = 0; i < k; i++)
    {
        data[start] = T{1.f};
        start += stride;
        if (start >= num_experts) // Wrap
            start -= num_experts;
    }
}

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream)
{
    size_t item_size = sizeof(float);
    auto* func = &initRoutingKernelDiagonal<float>;
    if (type == nvinfer1::DataType::kHALF)
    {
        func = &initRoutingKernelDiagonal<half>;
        item_size = sizeof(half);
    }
#ifdef ENABLE_BF16
    else if (type == nvinfer1::DataType::kBF16)
    {
        func = &initRoutingKernelDiagonal<__nv_bfloat16>;
        item_size = sizeof(__nv_bfloat16);
    }
#endif

    check_cuda_error(cudaMemsetAsync(data_void, 0x0, num_experts * num_tokens * item_size, stream));

    int stride = tensorrt_llm::common::ceilDiv(num_experts, k);

    int blockDim = 256;
    int gridDim = tensorrt_llm::common::ceilDiv(num_tokens, blockDim);
    func<<<gridDim, blockDim, 0, stream>>>(data_void, num_experts, num_tokens, k, stride);

    // sync_check_cuda_error();
}

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;

} // namespace tensorrt_llm::kernels
