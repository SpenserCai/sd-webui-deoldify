'''
Author: SpenserCai
Date: 2023-08-06 20:15:12
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-07 21:57:30
Description: file content
'''
from modules import script_callbacks,shared, paths_internal
import gradio as gr

from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
import tempfile
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")

def process_image(video, render_factor):
    wkfolder = Path(tempfile.gettempdir() + '/deoldify')
    if not wkfolder.exists():
        wkfolder.mkdir()
    colorizer = get_stable_video_colorizer(root_folder=Path(paths_internal.models_path) ,workfolder=wkfolder)
    video_name = os.path.basename(video)
    # 把video复制到临时文件夹
    source_path = wkfolder/"source"
    if not source_path.exists():
        source_path.mkdir()
    shutil.copy(video, source_path/video_name)
    out_video = colorizer.colorize_from_file_name(video_name, render_factor=render_factor)
    # 删除wkfolder中除了result以外的目录
    for dir in wkfolder.iterdir():
        if dir.name != 'result':
            shutil.rmtree(dir)
    # 把out_video从Path对象转换为str
    out_video = str(out_video)
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