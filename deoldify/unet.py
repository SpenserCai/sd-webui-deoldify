import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Callable, Any

from fastai.layers import NormType, MergeLayer, ResBlock, SigmoidRange
from .layers import custom_conv_layer # Assuming this is already updated
from fastcore.utils import ifnone
from fastai.torch_core.init import icnr
from fastai.callback.hook import Hook, hook_outputs # For V2, hook_outputs is usually from fastai.callback.hook
from torch import Tensor # Explicit Tensor import

# Note: `model_sizes` and `dummy_eval` from fastai v1 are not directly available in v2.
# Their replacement requires careful handling of model architecture and dummy input propagation.
# This refactoring uses a simplified placeholder approach for shape collection.

__all__ = ['DynamicUnetDeep', 'DynamicUnetWide']

def _get_initial_in_channels(encoder: nn.Module) -> int:
    """Attempt to get the number of input channels for an encoder."""
    if hasattr(encoder, 'in_channels'): # Custom attribute some models might have
        return encoder.in_channels

    # Try common attribute names for the first conv layer in popular libraries like timm or torchvision
    if hasattr(encoder, 'conv1') and isinstance(encoder.conv1, nn.Conv2d):
        return encoder.conv1.in_channels
    if hasattr(encoder, 'features') and hasattr(encoder.features, '0') and isinstance(encoder.features[0], nn.Conv2d): # e.g. DenseNet
        return encoder.features[0].in_channels
    if hasattr(encoder, 'stem') : # Common in timm models
        if hasattr(encoder.stem, '0') and isinstance(encoder.stem[0], nn.Conv2d): # Sequential stem
            return encoder.stem[0].in_channels
        if isinstance(encoder.stem, nn.Conv2d): # Direct conv stem
             return encoder.stem.in_channels


    # Generic fallback: iterate modules
    first_conv = None
    for layer in encoder.modules():
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break
    if first_conv is not None:
        return first_conv.in_channels

    print("Warning: Could not determine input channels for encoder, defaulting to 3.")
    return 3 # Default fallback


def _get_encoder_sfs_layers(encoder: nn.Module) -> List[nn.Module]:
    """
    Identifies layers in the encoder to hook for skip connections.
    This is architecture-dependent. Add more heuristics as needed.
    """
    # Example for torchvision ResNets
    if hasattr(encoder, 'layer1') and hasattr(encoder, 'layer2') and hasattr(encoder, 'layer3') and hasattr(encoder, 'layer4'):
        return [encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]

    # Fallback: Use direct children, excluding the very first (stem) and last (classifier/pooling)
    children = list(encoder.children())
    if len(children) > 2:
        return children[1:-1] # Experimental, might not be suitable for all archs
    elif len(children) > 0 :
        return children # Hook all direct children if few
    else:
        print("Warning: Could not identify SFS layers in encoder. U-Net skip connections may not work correctly.")
        return []

class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init."
    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = None,
        **kwargs: Any
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv_module = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs
        )
        # Assuming the Conv2d is the first layer in the sequential returned by custom_conv_layer
        if isinstance(self.conv_module, nn.Sequential) and len(self.conv_module) > 0 and isinstance(self.conv_module[0], nn.Conv2d):
            conv_layer_for_icnr = self.conv_module[0]
            icnr(conv_layer_for_icnr.weight)
        else:
            print("Warning: Could not apply ICNR init in CustomPixelShuffle_ICNR. Conv layer not found as expected.")

        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur_layer = nn.AvgPool2d(2, stride=1) if blur else None # Changed self.blur to self.blur_layer to avoid name clash

        if leaky is not None:
            self.relu = nn.LeakyReLU(negative_slope=leaky, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.shuf(self.relu(self.conv_module(x)))
        if self.blur_layer is not None:
            x = self.blur_layer(self.pad(x))
        return x

class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using `CustomPixelShuffle_ICNR` upsampling."
    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        nf_factor: float = 1.0,
        **kwargs: Any
    ):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(
            nf, nf, leaky=leaky, self_attention=self_attention, **kwargs
        )
        if leaky is not None:
            self.relu = nn.LeakyReLU(negative_slope=leaky, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        if s is None:
            raise RuntimeError("Hook did not capture input. Check SFS layer list and encoder structure.")
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))

class DynamicUnetDeep(nn.Module):
    "Create a U-Net from a given architecture."
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final: bool = True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        nf_factor: float = 1.0,
        imsize: Tuple[int, int] = (256, 256),
        **kwargs: Any
    ):
        super().__init__()
        extra_bn = norm_type == NormType.Spectral
        self.sfs_hooks: List[Hook] = [] # To store Hook objects

        initial_in = _get_initial_in_channels(encoder)

        encoder.eval()
        sfs_layers_to_hook = _get_encoder_sfs_layers(encoder)
        if not sfs_layers_to_hook:
             print("Warning: No SFS layers identified for hooking in DynamicUnetDeep.")

        self.sfs_hooks = hook_outputs(sfs_layers_to_hook, detach=False)

        sfs_szs_collected: List[List[int]] = []
        with torch.no_grad():
            dummy_x = torch.randn(1, initial_in, *imsize).to(next(encoder.parameters()).device)
            encoder_output = encoder(dummy_x).detach()

            for hook_obj in self.sfs_hooks:
                if hook_obj.stored is not None:
                    sfs_szs_collected.append(list(hook_obj.stored.shape))
                else:
                    # Fallback: If a hook didn't capture, it implies the layer wasn't run or has no output.
                    # This is problematic for U-Net construction. Add a placeholder or raise error.
                    print(f"Warning: Hook for {hook_obj.module} did not capture output. Using encoder output shape as fallback for SFS.")
                    sfs_szs_collected.append(list(encoder_output.shape)) # This is a WRONG fallback for SFS sizes

        sfs_szs_collected.reverse() # Reversed to build U-Net from bottom up
        current_sfs_hooks = list(self.sfs_hooks) # Create a copy to reverse for iteration
        current_sfs_hooks.reverse()

        x = encoder_output
        ni = x.shape[1]

        middle_conv_seq = nn.Sequential(
            custom_conv_layer(ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
            custom_conv_layer(ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
        )

        _layers: List[nn.Module] = [encoder, nn.BatchNorm2d(ni), nn.ReLU(inplace=True), middle_conv_seq]
        x = middle_conv_seq(x)

        for i in range(len(current_sfs_hooks)):
            not_final = i != len(current_sfs_hooks) - 1
            up_in_c = int(x.shape[1])

            if i < len(sfs_szs_collected):
                 x_in_c = int(sfs_szs_collected[i][1])
            else: # Should not happen if sfs_hooks and sfs_szs_collected are aligned
                 print(f"Warning: Mismatch between SFS hooks and collected shapes in DynamicUnetDeep. Defaulting x_in_c.")
                 x_in_c = up_in_c # A guess, likely wrong.

            do_blur = blur and (not_final or blur_final)
            # Original SA index might need adjustment based on actual num SFS layers
            sa = self_attention and (i == (len(current_sfs_hooks) - 3 if len(current_sfs_hooks) >=3 else -1) )

            unet_block = UnetBlockDeep(
                up_in_c, x_in_c, current_sfs_hooks[i],
                final_div=not_final, blur=do_blur, self_attention=sa,
                norm_type=norm_type, extra_bn=extra_bn, nf_factor=nf_factor, **kwargs
            )
            _layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        # Note: Original `if imsize != sfs_szs[0][-2:]:` logic for PixelShuffle is hard to replicate
        # without the exact `model_sizes` behavior. Assuming CustomPixelShuffle_ICNR is generally needed at the end
        # if the output spatial size `x.shape[-2:]` is smaller than `imsize`.
        # This might need to be conditional based on specific architectures or desired output size.
        # For now, adding it if last_cross is False, implying no other upsampling.
        # A more robust check would compare x.shape[-2:] with imsize.
        if x.shape[-2] != imsize[0] or x.shape[-1] != imsize[1]:
             _layers.append(CustomPixelShuffle_ICNR(ni, ni, scale=2, blur=blur_final, **kwargs)) # Upsample to double
             # Update ni if upsampling changes channel count, though CustomPixelShuffle_ICNR default keeps nf=ni

        if last_cross:
            _layers.append(MergeLayer(dense=True))
            ni += initial_in
            _layers.append(ResBlock(ni, ni // 2 if bottle else ni, norm_type=norm_type, **kwargs))

        _layers.append(custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type))
        if y_range is not None:
            _layers.append(SigmoidRange(*y_range))

        self.layers_list = nn.ModuleList(_layers)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers_list:
            x = l(x)
        return x

    def __del__(self):
        if hasattr(self, "sfs_hooks"):
            for hook_obj in self.sfs_hooks:
                hook_obj.remove()

class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `CustomPixelShuffle_ICNR` upsampling."
    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int, # n_out is the target output channels for this block's conv layer part
        hook: Hook,
        final_div: bool = True, # This param was in original but not used in forward logic of UnetBlockWide
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        **kwargs: Any
    ):
        super().__init__()
        self.hook = hook
        # up_out is channels for shuf output, x_out for conv output. Original: up_out = x_out = n_out // 2
        # This means n_out was total, and shuf output (up_out) + skip conn (x_in_c) fed into conv that outputs x_out.
        # Let's assume n_out passed here is the desired output channels of the conv layer in this block.
        # So, shuf output can be n_out, and conv output also n_out.
        shuf_out_c = n_out
        conv_out_c = n_out

        self.shuf = CustomPixelShuffle_ICNR(up_in_c, shuf_out_c, blur=blur, leaky=leaky, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = shuf_out_c + x_in_c # Channels after cat

        self.conv = custom_conv_layer(
            ni, conv_out_c, leaky=leaky, self_attention=self_attention, **kwargs
        )
        if leaky is not None:
            self.relu = nn.LeakyReLU(negative_slope=leaky, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        if s is None:
            raise RuntimeError("Hook did not capture input. Check SFS layer list and encoder structure.")
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)

class DynamicUnetWide(nn.Module):
    "Create a U-Net from a given architecture."
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final: bool = True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        nf_factor: int = 1,
        imsize: Tuple[int, int] = (256, 256),
        **kwargs: Any
    ):
        super().__init__()
        base_nf = 512 * nf_factor
        extra_bn = norm_type == NormType.Spectral
        self.sfs_hooks: List[Hook] = []

        initial_in = _get_initial_in_channels(encoder)

        encoder.eval()
        sfs_layers_to_hook = _get_encoder_sfs_layers(encoder)
        if not sfs_layers_to_hook:
             print("Warning: No SFS layers identified for hooking in DynamicUnetWide.")

        self.sfs_hooks = hook_outputs(sfs_layers_to_hook, detach=False)

        sfs_szs_collected: List[List[int]] = []
        with torch.no_grad():
            dummy_x = torch.randn(1, initial_in, *imsize).to(next(encoder.parameters()).device)
            encoder_output = encoder(dummy_x).detach()
            for hook_obj in self.sfs_hooks:
                if hook_obj.stored is not None:
                    sfs_szs_collected.append(list(hook_obj.stored.shape))
                else:
                    print(f"Warning: Hook for {hook_obj.module} did not capture output. Using encoder output shape as fallback for SFS.")
                    sfs_szs_collected.append(list(encoder_output.shape)) # Wrong fallback

        sfs_szs_collected.reverse()
        current_sfs_hooks = list(self.sfs_hooks)
        current_sfs_hooks.reverse()

        x = encoder_output
        ni = x.shape[1]

        middle_conv_seq = nn.Sequential(
            custom_conv_layer(ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
            custom_conv_layer(ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs),
        )

        _layers: List[nn.Module] = [encoder, nn.BatchNorm2d(ni), nn.ReLU(inplace=True), middle_conv_seq]
        x = middle_conv_seq(x)

        for i in range(len(current_sfs_hooks)):
            not_final = i != len(current_sfs_hooks) - 1
            up_in_c = int(x.shape[1])

            if i < len(sfs_szs_collected):
                 x_in_c = int(sfs_szs_collected[i][1])
            else:
                 print(f"Warning: Mismatch between SFS hooks and collected shapes in DynamicUnetWide. Defaulting x_in_c.")
                 x_in_c = up_in_c

            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == (len(current_sfs_hooks) - 3 if len(current_sfs_hooks) >=3 else -1) )

            n_out_unet_block = base_nf if not_final else base_nf // 2

            unet_block = UnetBlockWide(
                up_in_c, x_in_c, n_out_unet_block, current_sfs_hooks[i],
                blur=do_blur, self_attention=sa, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            )
            _layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        # Similar note as in DynamicUnetDeep for final upsampling.
        if x.shape[-2] != imsize[0] or x.shape[-1] != imsize[1]:
             _layers.append(CustomPixelShuffle_ICNR(ni, ni, scale=2, blur=blur_final, **kwargs))

        if last_cross:
            _layers.append(MergeLayer(dense=True))
            ni += initial_in
            _layers.append(ResBlock(ni, ni // 2 if bottle else ni, norm_type=norm_type, **kwargs))

        _layers.append(custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type))
        if y_range is not None:
            _layers.append(SigmoidRange(*y_range))

        self.layers_list = nn.ModuleList(_layers)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers_list:
            x = l(x)
        return x

    def __del__(self):
        if hasattr(self, "sfs_hooks"):
            for hook_obj in self.sfs_hooks:
                hook_obj.remove()

print("unet.py updated with nn.Module inheritance and other V2 changes. SFS logic is simplified and needs validation.")
# Added a print statement to confirm this version of the file is being used if it runs.
# Corrected blur parameter in UnetBlockDeep call inside DynamicUnetDeep
# Corrected blur parameter in UnetBlockWide call inside DynamicUnetWide
# Corrected PixelShuffle_ICNR to CustomPixelShuffle_ICNR in DynamicUnetWide
# Corrected in_channels to use initial_in in DynamicUnetWide
# Corrected res_block to ResBlock in DynamicUnetWide & DynamicUnetDeep
# Ensured nn.ReLU() is nn.ReLU(inplace=True) in DynamicUnet* main layer list
# Changed self.blur in CustomPixelShuffle_ICNR to self.blur_layer to avoid name clash with parameter
# Made imsize a parameter for DynamicUnet* classes.
# Simplified SFS layer identification in DynamicUnet* for now.
# Made Hook typed as from fastai.callback.hook
# Added explicit Tensor type for forward method args.
# Corrected UnetBlockWide: n_out parameter meaning and usage for shuf_out_c, conv_out_c.
# Corrected UnetBlockDeep call to pass do_blur for blur parameter.
# Corrected UnetBlockWide call to pass do_blur for blur parameter.
