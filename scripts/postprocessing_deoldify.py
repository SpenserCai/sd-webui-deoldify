'''
Author: SpenserCai
Date: 2023-07-28 14:41:28
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-09 10:11:23
Description: file content
'''
# DeOldify UI & Processing
from modules import scripts_postprocessing, paths_internal
from modules.ui_components import FormRow
from scripts.deoldify_base import *
import gradio as gr

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Deoldify"
    order = 20001

    def ui(self):
        with FormRow():
            is_enabled = gr.Checkbox(label="Deoldify")
            is_enabled.value = False
            render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="Render Factor")
            render_factor.value = 35
            # 一个名为artistic的复选框，初始值是False
            artistic = gr.Checkbox(label="Artistic")
            artistic.value = False

        return {
            "is_enabled": is_enabled,
            "render_factor": render_factor,
            "artistic": artistic,
        }
    
    def process_image(self, image, render_factor, artistic):
        vis = get_image_colorizer(root_folder=Path(paths_internal.models_path),render_factor=render_factor, artistic=artistic)
        outImg = vis.get_transformed_image_from_image(image, render_factor=render_factor)
        return outImg

    def process(self, pp: scripts_postprocessing.PostprocessedImage, is_enabled, render_factor, artistic):
        if not is_enabled or is_enabled is False:
            return

        pp.image = self.process_image(pp.image, render_factor, artistic)
        pp.info["deoldify"] = f"render_factor={render_factor}, artistic={artistic}"