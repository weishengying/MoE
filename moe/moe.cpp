#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

#include "th_utils.h"

using torch::Tensor;

// template<typename T>
// inline T* get_ptr(torch::Tensor& t)
// {
//     return reinterpret_cast<T*>(t.data_ptr());
// }

tensorrt_llm::ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return tensorrt_llm::ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return tensorrt_llm::ActivationType::Geglu;
    }
    else {
        std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
    }
    return tensorrt_llm::ActivationType::InvalidType;
}

template<typename T, typename WeightType>
Tensor run_moe_fc_helper(Tensor                            input_activations, //(num_tokens, hidden_size)
                         Tensor                            gating_output, //(num_tokens, num_experts)
                         Tensor                            fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                         Tensor                            fc1_scales, //(num_experts, inter_size) 量化scale
                         Tensor                            fc1_expert_biases, //(num_experts, inter_size)
                         tensorrt_llm::ActivationType fc1_activation_type,
                         Tensor                            fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                         Tensor                            fc2_scales, //(num_experts, hidden_size) 量化scale
                         Tensor                            fc2_expert_biases, //(num_experts, hidden_size)
                         Tensor                            skip_layer, //(num_rows, hidden_size)
                         Tensor                            finished, //(num_rows)
                         const int                         active_rows,
                         const int                         k)
{

    const int num_rows    = input_activations.size(0); //(num_tokens, hidden_size)
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1); //(num_experts, inter_size, hidden_size)
    const int num_experts = gating_output.size(-1); //(num_tokens, num_experts)
    auto      stream      = at::cuda::getCurrentCUDAStream().stream();

    T* input_act_ptr     = get_ptr<T>(input_activations);
    T* gating_output_ptr = get_ptr<T>(gating_output);

    WeightType*           fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);
    static constexpr bool is_fp16_or_fp32 =
        std::is_same<WeightType, float>::value || std::is_same<WeightType, half>::value;
#ifdef ENABLE_BF16
    static constexpr bool ignore_scales = is_fp16_or_fp32 || std::is_same<WeightType, __nv_bfloat16>::value;
#else
    static constexpr bool ignore_scales = is_fp16_or_fp32;
#endif

    T* fc1_scales_ptr        = ignore_scales ? nullptr : get_ptr<T>(fc1_scales);
    T* fc1_expert_biases_ptr = get_ptr<T>(fc1_expert_biases);

    WeightType* fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
    T*          fc2_scales_ptr         = ignore_scales ? nullptr : get_ptr<T>(fc2_scales);
    T*          fc2_expert_biases_ptr  = get_ptr<T>(fc2_expert_biases);

    T*    skip_layer_ptr = get_ptr<T>(skip_layer);
    bool* finished_ptr   = get_ptr<bool>(finished);

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig::TensorParallelism(1, 0);
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType> moe_runner;
    long int bytes        = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, moe_parallel_config);

    auto workspace_tensor = torch::empty({bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    char* workspace_ptr   = get_ptr<char>(workspace_tensor);

    const at::ScalarType _st = input_activations.scalar_type();
    auto                 fc2_output =
        torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* fc2_output_ptr = get_ptr<T>(fc2_output);

    auto expert_scales     = torch::empty({num_rows, k}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T*   expert_scales_ptr = get_ptr<T>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

    auto output_tensor =
        torch::empty({num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* output_tensor_ptr = get_ptr<T>(output_tensor);

    moe_runner.runMoe(input_act_ptr,
                        gating_output_ptr,
                        fc1_expert_weights_ptr,
                        fc1_scales_ptr,
                        fc1_expert_biases_ptr,
                        fc1_activation_type,
                        fc2_expert_weights_ptr,
                        fc2_scales_ptr,
                        fc2_expert_biases_ptr,
                        num_rows,
                        hidden_size,
                        inter_size,
                        num_experts,
                        k,
                        workspace_ptr,
                        output_tensor_ptr,
                        fc2_output_ptr,
                        finished_ptr,
                        active_rows,
                        expert_scales_ptr,
                        expanded_source_row_to_expanded_dest_row_ptr,
                        expert_for_source_row_ptr,
                        moe_parallel_config,
                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE,
                        stream);

    return output_tensor;
}


Tensor run_moe_fc(Tensor      input_activations, //(num_tokens, hidden_size)
                  Tensor      gating_output, //(num_tokens, num_experts)
                  Tensor      fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                  Tensor      fc1_scales, //(num_experts, inter_size) 量化scale
                  Tensor      fc1_expert_biases, //(num_experts, inter_size)
                  std::string fc1_activation_type_str,
                  Tensor      fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                  Tensor      fc2_scales, //(num_experts, hidden_size) 量化scale
                  Tensor      fc2_expert_biases, //(num_experts, hidden_size)
                  Tensor      skip_layer, //(num_rows, hidden_size)
                  Tensor      finished, //(num_rows)
                  int64_t     active_rows,
                  int64_t     k)
{

    const at::ScalarType _st = input_activations.scalar_type();

    const int num_rows    = input_activations.size(0);
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1);
    const int num_experts = gating_output.size(-1);

    // We signal int4 by having the last weight dim be half the size of the scales. This is because int4 elements are
    // packed into a single byte.
    torch::ScalarType quant_type = fc2_expert_weights.scalar_type();
    TORCH_CHECK(fc2_expert_weights.scalar_type() == fc1_expert_weights.scalar_type(),
                "FC1 and FC2 must be quantized to the same type");
    if (fc1_scales.dim() > 0 && fc1_expert_weights.size(-1) == fc1_scales.size(-1) / 2) {
        TORCH_CHECK(fc2_expert_weights.size(-1) == fc2_scales.size(-1) / 2, "FC1 and FC2 must be both be int4.");
        quant_type = at::ScalarType::QUInt4x2;
    }

    CHECK_INPUT(input_activations, _st);
    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");

    CHECK_INPUT(gating_output, _st);
    TORCH_CHECK(gating_output.dim() == 2, "Invalid rank for gating output");
    TORCH_CHECK(gating_output.size(0) == num_rows, "gating output and activations must have same number of rows");

    CHECK_TH_CUDA(fc1_expert_weights);
    CHECK_CONTIGUOUS(fc1_expert_weights);
    TORCH_CHECK(fc1_expert_weights.dim() == 3, "Invalid rank for fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(1) == hidden_size,
                "Activation last dim must equal size of dim 1 for fc1 weight");

    const int fc1_num_cols =
        quant_type == at::ScalarType::QUInt4x2 ? 2 * fc1_expert_weights.size(-1) : fc1_expert_weights.size(-1);
    if (_st != torch::kFloat32 && _st != torch::kFloat16) {
        CHECK_INPUT(fc1_scales, _st);
        TORCH_CHECK(fc1_scales.dim() == 2, "Invalid rank for fc1 scales");
        TORCH_CHECK(fc1_scales.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 scales");
        TORCH_CHECK(fc1_scales.size(-1) == fc1_num_cols, "Mismatch between fc1 weights and scale shapes");
        TORCH_CHECK(fc1_scales.size(-1) == fc1_expert_biases.size(-1), "Mismatch between fc1 scale and bias shapes");
    }

    CHECK_INPUT(fc1_expert_biases, _st);
    TORCH_CHECK(fc1_expert_biases.dim() == 2, "Invalid rank for fc1 biases");
    TORCH_CHECK(fc1_expert_biases.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc1 biases");

    CHECK_TH_CUDA(fc2_expert_weights);
    CHECK_CONTIGUOUS(fc2_expert_weights);
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(1) == fc1_num_cols, "fc1 weight last dim must equal dim 1 of fc2 weights");

    if (_st != torch::kFloat32 && _st != torch::kFloat16) {
        CHECK_INPUT(fc2_scales, _st);
        TORCH_CHECK(fc2_scales.dim() == 2, "Invalid rank for fc2 scales");
        TORCH_CHECK(fc2_scales.size(0) == gating_output.size(-1),
                    "Experts mismatch between gate outputs and fc2 scales");
        const int fc2_num_cols =
            quant_type == at::ScalarType::QUInt4x2 ? 2 * fc2_expert_weights.size(-1) : fc2_expert_weights.size(-1);
        TORCH_CHECK(fc2_scales.size(-1) == fc2_num_cols, "Mismatch between fc2 weights and scale shapes");
        TORCH_CHECK(fc2_scales.size(-1) == fc2_expert_biases.size(-1), "Mismatch between fc2 scale and bias shapes");
    }

    CHECK_INPUT(fc2_expert_biases, _st);
    TORCH_CHECK(fc2_expert_biases.dim() == 2, "Invalid rank for fc2 biases");
    TORCH_CHECK(fc2_expert_biases.size(0) == num_experts, "Experts mismatch between gate outputs and fc2 biases");

    CHECK_INPUT(skip_layer, _st);
    TORCH_CHECK(skip_layer.sizes() == input_activations.sizes(), "Invalid rank for skip connection");

    CHECK_INPUT(finished, torch::kBool);
    TORCH_CHECK(finished.dim() == 1, "Invalid rank for finished tensor");
    TORCH_CHECK(finished.size(0) == input_activations.size(0),
                "Finished and activations must have same number of rows");

    Tensor output_tensor;

    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::InvalidType;
    if (fc1_activation_type_str == "identity") {
        fc1_activation_type = tensorrt_llm::ActivationType::Identity;
    }
    else {
        fc1_activation_type = getActivationType(fc1_activation_type_str);
    }

    switch (_st) {
        case at::ScalarType::Float: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<float, float>(input_activations,
                                                                gating_output,
                                                                fc1_expert_weights,
                                                                fc1_scales,
                                                                fc1_expert_biases,
                                                                fc1_activation_type,
                                                                fc2_expert_weights,
                                                                fc2_scales,
                                                                fc2_expert_biases,
                                                                skip_layer,
                                                                finished,
                                                                active_rows,
                                                                k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case at::ScalarType::Half: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<half, half>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_scales,
                                                              fc1_expert_biases,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc2_scales,
                                                              fc2_expert_biases,
                                                              skip_layer,
                                                              finished,
                                                              active_rows,
                                                              k);
            }
            else if (quant_type == torch::kInt8) {
                output_tensor = run_moe_fc_helper<half, uint8_t>(input_activations,
                                                                 gating_output,
                                                                 fc1_expert_weights,
                                                                 fc1_scales,
                                                                 fc1_expert_biases,
                                                                 fc1_activation_type,
                                                                 fc2_expert_weights,
                                                                 fc2_scales,
                                                                 fc2_expert_biases,
                                                                 skip_layer,
                                                                 finished,
                                                                 active_rows,
                                                                 k);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = run_moe_fc_helper<half, cutlass::uint4b_t>(input_activations,
                                                                           gating_output,
                                                                           fc1_expert_weights,
                                                                           fc1_scales,
                                                                           fc1_expert_biases,
                                                                           fc1_activation_type,
                                                                           fc2_expert_weights,
                                                                           fc2_scales,
                                                                           fc2_expert_biases,
                                                                           skip_layer,
                                                                           finished,
                                                                           active_rows,
                                                                           k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_scales,
                                                                                fc1_expert_biases,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                fc2_scales,
                                                                                fc2_expert_biases,
                                                                                skip_layer,
                                                                                finished,
                                                                                active_rows,
                                                                                k);
            }
            else if (quant_type == torch::kInt8) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                          gating_output,
                                                                          fc1_expert_weights,
                                                                          fc1_scales,
                                                                          fc1_expert_biases,
                                                                          fc1_activation_type,
                                                                          fc2_expert_weights,
                                                                          fc2_scales,
                                                                          fc2_expert_biases,
                                                                          skip_layer,
                                                                          finished,
                                                                          active_rows,
                                                                          k);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                                                    gating_output,
                                                                                    fc1_expert_weights,
                                                                                    fc1_scales,
                                                                                    fc1_expert_biases,
                                                                                    fc1_activation_type,
                                                                                    fc2_expert_weights,
                                                                                    fc2_scales,
                                                                                    fc2_expert_biases,
                                                                                    skip_layer,
                                                                                    finished,
                                                                                    active_rows,
                                                                                    k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_moe_fc", &run_moe_fc, "moe.");
}
