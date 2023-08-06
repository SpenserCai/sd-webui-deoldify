'''
Author: SpenserCai
Date: 2023-08-06 20:15:12
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-06 20:44:00
Description: file content
'''
from modules import script_callbacks, paths_internal
import gradio as gr
import string

def process_image(video, render_factor, artistic):
    return video

def deoldify_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        # 多个tab第一个是video
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.inputs.Video(label="原视频",type="file")
                    # 一个名为render_factor的滑块，范围是1-50，初始值是35，步长是1
                    render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="Render Factor")
                    render_factor.value = 35
                    # 一个名为artistic的复选框，初始值是False
                    artistic = gr.Checkbox(label="Rrtistic")
                    artistic.value = False
                with gr.Column():
                    video_output = gr.outputs.Video(label="修复后的视频",type="auto")
                run_button = gr.Button(label="Run")
                run_button.click(inputs=[video_input,render_factor,artistic],outputs=[video_output],fn=process_image)

    return [(ui,"DeOldify","DeOldify")]

script_callbacks.on_ui_tabs(deoldify_tab)