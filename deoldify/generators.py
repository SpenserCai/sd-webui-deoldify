from fastai.vision.all import * # Includes models, Learner, DataLoaders, etc.
from fastai.data.core import DataLoaders # Explicit import for clarity if needed
from fastai.learner import Learner # Explicit import for clarity
from fastai.torch_core import apply_init, to_device # SplitFuncOrIdxList is often handled by Learner methods or specific splitter functions
# from fastai.layers import NormType # NormType is available via fastai.vision.all
# from fastai.vision.learner import create_body # create_body is available via fastai.vision.all
# cnn_config is deprecated, model_meta or default splitters are used.
from torch import nn
import torch.nn.functional as F # For F.l1_loss
from .unet import DynamicUnetWide, DynamicUnetDeep
from .dataset import get_dummy_databunch # Correctly imports the V2 version

# Weights are implicitly read from ./models/ folder
def gen_inference_wide(
    root_folder: Path, weights_name: str, nf_factor: int = 2, arch=resnet101
) -> Learner:
    dls = get_dummy_databunch() # Now returns DataLoaders
    learn = gen_learner_wide(
        dls=dls, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.path = root_folder # learn.path should still work
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_wide(
    dls: DataLoaders, gen_loss, arch=resnet101, nf_factor: int = 2
) -> Learner:
    return unet_learner_wide(
        dls,
        arch=arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


# The code below is meant to be merged into fastaiv1 ideally (now fastai v2)
def unet_learner_wide(
    dls: DataLoaders,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType.Batch, # Default NormType.Batch if not specified
    split_on: Optional[Callable] = None, # Changed from SplitFuncOrIdxList to Callable
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: int = 1,
    **kwargs: Any
) -> Learner:
    "Build Unet learner from `dls` and `arch`."
    # meta = cnn_config(arch) # cnn_config is deprecated
    # Default splitter for resnet architectures in fastai v2 is usually handled by Learner
    # or by passing a splitter like `resnet_splitter`
    body = create_body(arch, pretrained, cut=None) # cut=None lets create_body decide

    # Ensure dls.c is available and correct. For ImageBlock to ImageBlock, dls.c should be 3.
    # If get_dummy_databunch might result in dls without items, dls.c might be an issue.
    # However, the original code also used data.c from a dummy databunch.
    model_n_classes = dls.c if hasattr(dls, 'c') else 3 # Fallback to 3 if dls.c not present

    model = to_device(
        DynamicUnetWide(
            body,
            n_classes=model_n_classes,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        dls.device if hasattr(dls, 'device') else default_device(), # Fallback for device
    )

    learn = Learner(dls, model, **kwargs)

    # Handle model splitting
    if split_on is None:
        # Default splitting for common architectures like ResNets
        # For unets, often the encoder is split from the decoder part
        # Fastai's Learner might do this by default if body is a standard arch
        # Or, a custom splitter might be needed if DynamicUnet structure is specific
        # For now, we rely on Learner's default splitter or pass one if known.
        # Example: learn.split(default_split) or learn.split(resnet_splitter) if arch is resnet
        # If DynamicUnetWide has its own preferred split, it should be passed via split_on
        # For now, let's assume default_split or that Learner handles it.
        # If `arch` is a resnet, `resnet_splitter` can be used.
        if 'resnet' in arch.__name__:
             learn.split(resnet_splitter)
        # else: pass # use default learner splitter
    else:
        learn.split(split_on)

    if pretrained:
        learn.freeze()

    # apply_init to the decoder part (U-Net specific layers)
    # model[1] is often the U-Net decoder if model[0] is the backbone
    if hasattr(model, 'children'):
        children = list(model.children())
        if len(children) > 1 and hasattr(children[1], 'parameters'): # Check if there's a second component (decoder)
             apply_init(children[1], nn.init.kaiming_normal_)
    return learn


# ----------------------------------------------------------------------

# Weights are implicitly read from ./models/ folder
def gen_inference_deep(
    root_folder: Path, weights_name: str, arch=resnet34, nf_factor: float = 1.5
) -> Learner:
    dls = get_dummy_databunch() # Now returns DataLoaders
    learn = gen_learner_deep(
        dls=dls, gen_loss=F.l1_loss, arch=arch, nf_factor=nf_factor
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_deep(
    dls: DataLoaders, gen_loss, arch=resnet34, nf_factor: float = 1.5
) -> Learner:
    return unet_learner_deep(
        dls,
        arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


# The code below is meant to be merged into fastaiv1 ideally (now fastai v2)
def unet_learner_deep(
    dls: DataLoaders,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType.Batch, # Default NormType.Batch
    split_on: Optional[Callable] = None, # Changed from SplitFuncOrIdxList to Callable
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: float = 1.5,
    **kwargs: Any
) -> Learner:
    "Build Unet learner from `dls` and `arch`."
    # meta = cnn_config(arch) # cnn_config is deprecated
    body = create_body(arch, pretrained, cut=None)

    model_n_classes = dls.c if hasattr(dls, 'c') else 3

    model = to_device(
        DynamicUnetDeep(
            body,
            n_classes=model_n_classes,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        dls.device if hasattr(dls, 'device') else default_device(),
    )
    learn = Learner(dls, model, **kwargs)

    if split_on is None:
        if 'resnet' in arch.__name__:
             learn.split(resnet_splitter)
        # else: pass # use default learner splitter
    else:
        learn.split(split_on)

    if pretrained:
        learn.freeze()

    if hasattr(model, 'children'):
        children = list(model.children())
        if len(children) > 1 and hasattr(children[1], 'parameters'):
             apply_init(children[1], nn.init.kaiming_normal_)
    return learn


# -----------------------------
