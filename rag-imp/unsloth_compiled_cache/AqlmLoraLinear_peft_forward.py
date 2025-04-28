"""
2025.3.17
2025.3.19
4.51.3
0.15.2
__UNSLOTH_VERSIONING__
"""

torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from peft.tuners.lora.aqlm import (torch)


torch_addmm = torch.addmm
torch_add   = torch.add
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x) @ lora_A.weight.t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
        output,
        bias,
        alpha = scaling,
    )
    return output
pass

def unsloth_forward(self, x: torch.Tensor):
    # note: logic differs from default Linear because merging is not supported
    result = self.base_layer(x)

    if self.disable_adapters:
        return result

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

        output = lora_B(lora_A(dropout(x)))
        if requires_conversion:
            output = output.to(expected_dtype)
        output = output * scaling
        result += output
    return result
