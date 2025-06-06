from fastai.layers import NormType, SelfAttention # Specific imports
# spectral_norm and weight_norm will be used from torch.nn.utils
# from fastai.torch_core import * # Avoid wildcard, specific imports if needed elsewhere
import torch.nn as nn
from torch.nn.utils import spectral_norm as torch_spectral_norm, weight_norm as torch_weight_norm
from typing import Optional, Callable # For type hints

# The code below is meant to be merged into fastaiv1 ideally (now fastai v2)


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Optional[Callable] = None, # Default init is now PyTorch's Kaiming He for Conv layers
    self_attention: bool = False,
    extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0

    # Determine if batchnorm is used
    is_bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not is_bn # Set bias to not is_bn if not explicitly specified

    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding)

    # Apply initialization if provided (otherwise PyTorch default is used)
    if init is not None:
        init(conv.weight)
        if hasattr(conv, 'bias') and conv.bias is not None:
            nn.init.zeros_(conv.bias) # Example: zero init bias, or pass specific bias init

    # Apply normalization to the conv layer itself
    if norm_type == NormType.Weight:
        conv = torch_weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = torch_spectral_norm(conv)

    layers = [conv]

    if use_activ:
        if leaky is not None:
            layers.append(nn.LeakyReLU(negative_slope=leaky, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

    if is_bn: # Use the determined 'is_bn'
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))

    if self_attention:
        layers.append(SelfAttention(nf))

    return nn.Sequential(*layers)
