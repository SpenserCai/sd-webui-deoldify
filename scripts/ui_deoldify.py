'''
Author: SpenserCai
Date: 2023-08-06 20:15:12
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-06 21:42:47
Description: file content
'''
from modules import script_callbacks, paths_internal
import gradio as gr
from deoldify.visualize import *
import tempfile

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")

def process_image(video, render_factor):
    colorizer = get_video_colorizer(workfolder=Path(tempfile.gettempdir() + '/deoldify'))
    # 把video复制到缓存路径
    
    out_video = colorizer.colorize_from_file_name(video, render_factor=render_factor)
    return out_video

def deoldify_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        # 多个tab第一个是video
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.inputs.Video(label="原视频")
                    # 一个名为render_factor的滑块，范围是1-50，初始值是35，步长是1
                    render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="Render Factor")
                    render_factor.value = 35
                with gr.Column():
                    video_output = gr.outputs.Video(label="修复后的视频")
            run_button = gr.Button(label="Run")
            run_button.click(inputs=[video_input,render_factor],outputs=[video_output],fn=process_image)

    return [(ui,"DeOldify","DeOldify")]

script_callbacks.on_ui_tabs(deoldify_tab)