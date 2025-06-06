from fastai.learner import Learner # Changed
from abc import ABC, abstractmethod
from fastcore.all import * # Changed: From fastai.core
from fastai.vision.all import * # Consolidated vision imports
# from fastai.vision.image import * # Covered by vision.all
# from fastai.vision.data import * # Covered by vision.all
# from fastai import * # Covered by vision.all and learner
import cv2
from PIL import Image as PilImage # PIL is already imported by fastai.vision.all as PILImage
from deoldify import device as device_settings
import logging
import torch # Explicit import for torch.no_grad()

class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass


class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        
        if not device_settings.is_gpu():
            self.learn.model = self.learn.model.cpu()
        
        self.device = learn.dls.device if hasattr(learn, 'dls') and hasattr(learn.dls, 'device') else default_device()
        # self.norm, self.denorm = normalize_funcs(*stats) # normalize_funcs still exists
        # Denormalization might not be needed if learn.predict gives decoded output.
        # If raw tensor output is used, then denorm is needed.
        # For now, let's assume learn.predict provides what we need.
        # Storing stats might be useful if manual denorm is required later.
        self.stats = stats


    def _transform(self, image: PilImage) -> PilImage:
        return image

    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PilImage.BILINEAR) # PIL.Image needs to be PilImage

    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz) # This is a PIL Image
        
        try:
            # learn.predict handles transformations and moving to device
            # It returns a tuple: (decoded_input, raw_pred_tensor, decoded_output)
            # We want the decoded_output, which is usually a PILImage for vision tasks.
            with torch.no_grad(): # Ensure no gradients are computed during prediction
                _, _, pred_decoded = self.learn.predict(model_image)

            # pred_decoded should be the final processed image (e.g., PIL Image)
            # If it's a tensor, it needs conversion and denormalization.
            # FastAI v2 predict usually returns a high-level object like TensorImage or PILImage.
            # If pred_decoded is a TensorImage:
            if isinstance(pred_decoded, TensorImage):
                # Convert to PIL Image. This TensorImage should already be denormalized and on CPU.
                out_pil = ToPIL()(pred_decoded)
            elif isinstance(pred_decoded, PilImage):
                out_pil = pred_decoded
            elif isinstance(pred_decoded, torch.Tensor):
                # This case implies predict didn't fully decode to PILImage.
                # We might need to denormalize and convert manually.
                # This would be similar to the old path:
                # 1. Ensure tensor is on CPU
                pred_tensor = pred_decoded.cpu()
                # 2. Denormalize (assuming it's in 0-1 range after sigmoid/tanh if applicable, or needs full denorm)
                # If the model outputs normalized values, we need self.denorm.
                # Let's get denorm from normalize_funcs if needed.
                # For now, assume predict gives something that can be converted to PIL directly or is already PIL.
                # If it's a raw normalized tensor, the following would be needed:
                # _, denorm_func = normalize_funcs(*self.stats)
                # pred_tensor_denorm = denorm_func(pred_tensor)
                # out_pil = ToPIL()(pred_tensor_denorm)
                # This path is less likely for learn.predict's third output element.
                # For safety, let's assume it's a PIL image or TensorImage.
                # If it's a raw tensor, this will need adjustment.
                # For now, convert directly assuming it's in displayable range (e.g. 0-1 or 0-255)
                out_pil = ToPIL()(pred_decoded)
            else:
                # Unexpected type
                logging.error(f"Unexpected prediction output type: {type(pred_decoded)}")
                return model_image # return original on error

        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            logging.warn('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image # Return the pre-processed model_image (PIL)
            
        # out_pil is now expected to be a PIL Image
        # The old code did: image2np(out * 255).astype(np.uint8)
        # If out_pil is a PIL Image, image2np is sufficient.
        # The *255 was because denorm(out.px) resulted in a 0-1 float tensor.
        # ToPIL should handle this conversion correctly from a TensorImage.
        # If pred_decoded was a raw tensor and manually converted, *255 might be needed if it was 0-1.
        # Assuming ToPIL from TensorImage gives a standard PIL image:
        # out_np = image2np(out_pil).astype(np.uint8) # image2np is from fastai.vision.all
        # return PilImage.fromarray(out_np)
        return out_pil # Should be a PIL Image

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PilImage.BILINEAR) # PIL.Image needs to be PilImage
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz) # Returns PIL image
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        # Ensure orig_np is RGB before converting to YUV
        if orig_np.shape[2] == 4: # RGBA
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)

        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)

        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
        final = PilImage.fromarray(final)
        return final


class MasterFilter(BaseFilter):
    def __init__(self, filters: List[IFilter], render_factor: int):
        # This class doesn't call super().__init__(learn, stats) which might be an issue
        # if it's intended to be a BaseFilter itself. However, it only implements IFilter.
        # For now, assuming this is as intended. If it needs its own learner, this needs change.
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for item_filter in self.filters: # renamed 'filter' to 'item_filter' to avoid conflict with method name
            filtered_image = item_filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image
