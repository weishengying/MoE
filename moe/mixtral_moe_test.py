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

def basic_moe_fc(activations, expert_for_row, weights):
  res = torch.zeros(size=[activations.shape[0], weights.shape[-1]], dtype=activations.dtype, device='cuda')
  for row in range(activations.shape[0]):
      row_expert = expert_for_row[row]
      torch.matmul(activations[row], weights[row_expert], out=res[row : row + 1, :])
  return res

def apply_act(inp, act_str): # inp (num_rows, inter_size)
  assert(act_str == "Swiglu")
  gated_size = int(inp.size(-1) / 2)
  return torch.nn.SiLU()(inp[:, gated_size : ]) * inp[:,  : gated_size] #这里请注意顺序，这样做的目的是为了适配 dogatedkernel, 当然也可以只接修改 dogatedkernel

class TestMoe(unittest.TestCase):

  def setUp(self) -> None:
    import moe_ops
    self.run_moe_fc = moe_ops.run_moe_fc
    torch.manual_seed(734876213)
  
  def generate_inputs(self, num_rows, active_rows, hidden_size, num_experts, dtype, quant_type):
    inputs = dict()
    inputs["input_activations"] = random_cuda_tensor([num_rows, hidden_size], dtype, mean=0, std=0.02)
    inputs["gating_output"] = random_cuda_tensor([num_rows, num_experts], dtype)

    return inputs
  
  def generate_weights(self, hidden_size, inter_size, num_experts, dtype, quant_type):
    weights = dict()
    weights["fc1_expert_weights_for_ref"] = random_cuda_tensor([num_experts, hidden_size, inter_size*2], dtype, mean=0, std=0.02)
    weights["fc1_expert_weights_for_ft"] = weights["fc1_expert_weights_for_ref"]

    weights["fc2_expert_weights_for_ref"] = random_cuda_tensor([num_experts, inter_size, hidden_size], dtype, mean=0, std=0.02)
    weights["fc2_expert_weights_for_ft"] = weights["fc2_expert_weights_for_ref"]

    return weights
  
  def run_ft_moe(self, input_dict, active_rows, k, activation_str):
    moe_output = self.run_moe_fc(input_dict["input_activations"], input_dict["gating_output"], \
                    input_dict["fc1_expert_weights_for_ft"], \
                    activation_str, \
                    input_dict["fc2_expert_weights_for_ft"], \
                    active_rows, k)
    return moe_output
  
  def run_ref_moe(self, input_dict, k, activation_str):
    gates = F.softmax(input_dict["gating_output"].to(torch.float32), dim=-1).to(input_dict["gating_output"].dtype)
    expert_scales, experts_for_row = torch.topk(gates, k, dim=-1)

    output = torch.zeros_like(input_dict["input_activations"])

    for k_idx in range(k):
      current_expert_scales = expert_scales[:, k_idx].unsqueeze(1)
      current_experts_for_row = experts_for_row[:, k_idx]

      moe_fc_1_result = basic_moe_fc(input_dict["input_activations"], current_experts_for_row, 
                                     input_dict["fc1_expert_weights_for_ref"])

      # print(f"moe_fc_1_result: {moe_fc_1_result}")
      moe_fc_1_result = apply_act(moe_fc_1_result, activation_str)
      # print(f"moe_fc_1_result: {moe_fc_1_result}")

      moe_fc_2_result = basic_moe_fc(moe_fc_1_result, current_experts_for_row, 
                                     input_dict["fc2_expert_weights_for_ref"])
      
      # print(f"moe_fc_2_result: {moe_fc_2_result}")
      output = output + current_expert_scales * moe_fc_2_result
    return output

  def moe_test_helper(self, dtype, quant_type, rtol, atol, activation_str="gelu", experts_list=[32], hidden_sizes=[1024], inter_sizes=[4096]):
    torch.cuda.empty_cache() # Empty the cache here so a bad ordering does not cause OOM.
    rows = [4096]
    ks = [2]

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

                act_output = self.run_ft_moe(input_dict, row, k, activation_str)
                ref_output = self.run_ref_moe(input_dict, k, activation_str)

                msg = "Moe Failed on rows={}, active_rows={}, experts={}, k={}, hidden_size={}, inter_size={}" \
                        .format(row, active_rows, experts, k, hidden_size, inter_size)
                print(f"act_output: {act_output}")
                print(f"ref_output: {ref_output}")
                torch.testing.assert_close(act_output, ref_output, rtol=rtol, atol=atol, msg=msg, check_dtype=False)
  
  def test_moe_fp32_relu(self):
    self.moe_test_helper(torch.float32, torch.float32, rtol=1e-3, atol=1e-5, \
                         activation_str="Swiglu", \
                         experts_list=[8], hidden_sizes=[4096], \
                         inter_sizes=[14336])

  def test_moe_fp16_gelu(self):
    self.moe_test_helper(torch.float16, torch.float16, rtol=1e-3, atol=1e-3, \
                         activation_str="Swiglu", \
                         experts_list=[8], hidden_sizes=[4096], \
                         inter_sizes=[14336])

  def test_moe_bf16_gelu(self):
    self.moe_test_helper(torch.bfloat16, torch.bfloat16, rtol=1e-3, atol=0.05, \
                         activation_str="Swiglu", \
                         experts_list=[8], hidden_sizes=[4096], \
                         inter_sizes=[14336])

if __name__ == '__main__':
    unittest.main()