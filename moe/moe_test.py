# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import numpy as np
import unittest

def random_cuda_tensor(shape, dtype, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

def basic_moe_fc(activations, expert_for_row, weights, scales, biases):
  if weights.dtype == torch.int8:
      weights = torch.multiply(weights, scales.unsqueeze(1))
      weights = weights.to(activations.dtype)
  elif weights.dtype != torch.bfloat16 and weights.dtype != torch.float16 and weights.dtype != torch.float32:
      raise ValueError("Invalid data type for weights")
      

  res = torch.zeros(size=[activations.shape[0], weights.shape[-1]], dtype=activations.dtype, device='cuda')
  for row in range(activations.shape[0]):
      row_expert = expert_for_row[row]
      torch.matmul(activations[row], weights[row_expert], out=res[row : row + 1, :])
      res[row] += biases[row_expert]

  return res

def apply_act(inp, act_str):
  if act_str == "identity":
    return inp
  elif act_str == "silu":
    return torch.nn.SiLU()(inp)
  elif act_str == "relu":
    return torch.nn.ReLU()(inp)
  elif act_str == "gelu":
    return torch.nn.GELU(approximate="tanh")(inp)
  else:
    assert False, "Unsupported activation"


class TestMoe(unittest.TestCase):

  def setUp(self) -> None:
    import moe_ops
    self.run_moe_fc = moe_ops.run_moe_fc
    torch.manual_seed(734876213)
  
  def generate_inputs(self, num_rows, active_rows, hidden_size, num_experts, dtype, quant_type):
    inputs = dict()

    inputs["input_activations"] = random_cuda_tensor([num_rows, hidden_size], dtype, mean=0, std=0.002)
    inputs["gating_output"] = random_cuda_tensor([num_rows, num_experts], dtype)

    inputs["skip_layer"] = random_cuda_tensor([num_rows, hidden_size], dtype)

    num_finished_sentences = num_rows - active_rows
    finished_sentences = torch.randint(0, num_rows, [num_finished_sentences], device="cuda")
    inputs["finished"] = torch.zeros([num_rows], dtype=torch.bool, device="cuda")
    inputs["finished"][finished_sentences] = True

    return inputs
  
  def generate_weights(self, hidden_size, inter_size, num_experts, dtype, quant_type):
    weights = dict()
    quantize = quant_type == torch.int8 or quant_type == torch.quint4x2

    weights["fc1_expert_weights_for_ref"] = random_cuda_tensor([num_experts, hidden_size, inter_size], dtype, mean=0, std=0.002)
    weights["fc1_expert_weights_for_ft"] = weights["fc1_expert_weights_for_ref"]
    weights["fc1_scales"] = torch.ones(size=[num_experts, inter_size], dtype=dtype, device="cuda")
    weights["fc1_expert_biases"] = random_cuda_tensor([num_experts, inter_size], dtype, mean=0, std=0.002)

    weights["fc2_expert_weights_for_ref"] = random_cuda_tensor([num_experts, inter_size, hidden_size], dtype, mean=0, std=0.002)
    weights["fc2_expert_weights_for_ft"] = weights["fc2_expert_weights_for_ref"]
    weights["fc2_scales"] = torch.ones(size=[num_experts, hidden_size], dtype=dtype, device="cuda")
    weights["fc2_expert_biases"] = random_cuda_tensor([num_experts, hidden_size], dtype, mean=0, std=0.002)

    if quantize:
        ref_torch_weights_fc1, act_torch_weights_fc1, torch_weight_scales_fc1 = self.symmetric_quantizer(weights["fc1_expert_weights_for_ft"].cpu(), quant_type)
        ref_torch_weights_fc2, act_torch_weights_fc2, torch_weight_scales_fc2 = self.symmetric_quantizer(weights["fc2_expert_weights_for_ft"].cpu(), quant_type)

        if quant_type == torch.quint4x2:
          ref_torch_weights_fc1 = self.unpack_packed_int4s(ref_torch_weights_fc1)
          ref_torch_weights_fc2 = self.unpack_packed_int4s(ref_torch_weights_fc2)


        weights["fc1_expert_weights_for_ref"] = ref_torch_weights_fc1.to("cuda")
        weights["fc1_expert_weights_for_ft"] = act_torch_weights_fc1.to("cuda")
        weights["fc1_scales"] = torch_weight_scales_fc1.to("cuda")

        weights["fc2_expert_weights_for_ref"] = ref_torch_weights_fc2.to("cuda")
        weights["fc2_expert_weights_for_ft"] = act_torch_weights_fc2.to("cuda")
        weights["fc2_scales"] = torch_weight_scales_fc2.to("cuda")
  
    return weights
  
  def run_ft_moe(self, input_dict, active_rows, k, activation_str):
    moe_output = self.run_moe_fc(input_dict["input_activations"], input_dict["gating_output"], \
                    input_dict["fc1_expert_weights_for_ft"], input_dict["fc1_scales"], input_dict["fc1_expert_biases"], \
                    activation_str, \
                    input_dict["fc2_expert_weights_for_ft"], input_dict["fc2_scales"], input_dict["fc2_expert_biases"], \
                    input_dict["skip_layer"], input_dict["finished"], active_rows, k)
    return moe_output
  
  def run_ref_moe(self, input_dict, k, activation_str):
    gates = F.softmax(input_dict["gating_output"].to(torch.float32), dim=-1).to(input_dict["gating_output"].dtype)
    expert_scales, experts_for_row = torch.topk(gates, k, dim=-1)

    output = torch.zeros_like(input_dict["input_activations"])
    output += input_dict["skip_layer"]

    for k_idx in range(k):
      current_expert_scales = expert_scales[:, k_idx].unsqueeze(1)
      current_experts_for_row = experts_for_row[:, k_idx]

      moe_fc_1_result = basic_moe_fc(input_dict["input_activations"], current_experts_for_row, 
                                     input_dict["fc1_expert_weights_for_ref"], input_dict["fc1_scales"], input_dict["fc1_expert_biases"])
      moe_fc_1_result = apply_act(moe_fc_1_result, activation_str)

      moe_fc_2_result = basic_moe_fc(moe_fc_1_result, current_experts_for_row, 
                                     input_dict["fc2_expert_weights_for_ref"], input_dict["fc2_scales"], input_dict["fc2_expert_biases"])
      
      output = output + current_expert_scales * moe_fc_2_result
    
    return output

  def moe_test_helper(self, dtype, quant_type, rtol, atol, activation_str="gelu", experts_list=[32], hidden_sizes=[1024], inter_sizes=[4096]):
    torch.cuda.empty_cache() # Empty the cache here so a bad ordering does not cause OOM.
    rows = [1]
    ks = [1]

    for hidden_size in hidden_sizes:
      for inter_size in inter_sizes:
        for experts in experts_list:
          weights = self.generate_weights(hidden_size, inter_size, experts, dtype, quant_type)
          for row in rows:
            for active_rows in [row]:
              for k in ks:
                if k > experts:
                  continue
                input_dict = self.generate_inputs(row, active_rows, hidden_size, experts, dtype, quant_type)
                input_dict.update(weights)            
                rows_to_check = torch.logical_not(input_dict["finished"])

                # Only take unfinished rows. We can write anything to the output of rows that already complete.
                act_output = self.run_ft_moe(input_dict, row, k, activation_str)[rows_to_check]
                print(f"act_output: {act_output}")
                ref_output = self.run_ref_moe(input_dict, k, activation_str)[rows_to_check]
                print(f"ref_output: {ref_output}")

                # msg = "Moe Failed on rows={}, active_rows={}, experts={}, k={}, hidden_size={}, inter_size={}" \
                #         .format(row, active_rows, experts, k, hidden_size, inter_size)
                # torch.testing.assert_close(act_output, ref_output, rtol=rtol, atol=atol, msg=msg, check_dtype=False)
  
  def test_moe_fp32_relu(self):
    self.moe_test_helper(torch.float32, torch.float32, rtol=1e-3, atol=1e-6, \
                         activation_str="relu", \
                         experts_list=[32], hidden_sizes=[16], \
                         inter_sizes=[32])

  # def test_moe_fp16_gelu(self):
  #   self.moe_test_helper(torch.float16, torch.float16, rtol=1e-3, atol=0.005, \
  #                        activation_str="gelu", \
  #                        experts_list=[128, 30, 7, 5, 3], hidden_sizes=[2048, 1024], \
  #                        inter_sizes=[4096])

#   def test_moe_bf16_gelu(self):
#     self.moe_test_helper(torch.bfloat16, torch.bfloat16, rtol=1e-2, atol=0.005, \
#                          activation_str="gelu", \
#                          experts_list=[64, 32], hidden_sizes=[1024], \
#                          inter_sizes=[4096])

if __name__ == '__main__':
    unittest.main()