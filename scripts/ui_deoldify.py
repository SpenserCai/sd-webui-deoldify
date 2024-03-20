'''
Author: SpenserCai
Date: 2023-08-06 20:15:12
version: 
LastEditors: SpenserCai
LastEditTime: 2023-09-07 10:21:59
Description: file content
'''
from modules import script_callbacks, paths_internal, shared
from modules.call_queue import wrap_queued_call
from scripts.deoldify_base import *
import gradio as gr
import tempfile
import os
import shutil

def process_image(video, render_factor,process=gr.Progress()):
    restoreList = []
    try:
        old_disable_safe_unpickle = shared.cmd_opts.disable_safe_unpickle
        def restore():
            shared.cmd_opts.disable_safe_unpickle = old_disable_safe_unpickle
        restoreList.append(restore)
        shared.cmd_opts.disable_safe_unpickle = True

        wkfolder = Path(tempfile.gettempdir() + '/deoldify')
        if not wkfolder.exists():
            wkfolder.mkdir()
        colorizer = get_stable_video_colorizer(root_folder=Path(paths_internal.models_path) ,workfolder=wkfolder)
        video_name = os.path.basename(video)
        process(0,"Copying video to temp folder...")
        # 把video复制到临时文件夹
        source_path = wkfolder/"source"
        if not source_path.exists():
            source_path.mkdir()
        shutil.copy(video, source_path/video_name)
        out_video = colorizer.colorize_from_file_name(video_name, render_factor=render_factor,g_process_bar=process)
        # 删除wkfolder中除了result以外的目录
        for dir in wkfolder.iterdir():
            if dir.name != 'result':
                shutil.rmtree(dir)
        # 把out_video从Path对象转换为str
        out_video = str(out_video)
        colorizer.vis._clean_mem()
        return out_video
    finally:
        for restore in restoreList:
            restore()


isChineese = shared.opts.localization.lower().startswith('zh')

def deoldify_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        # 多个tab第一个是video
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="原视频" if isChineese else "Original Video")
                    # 一个名为render_factor的滑块，范围是1-50，初始值是35，步长是1
                    render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="Render Factor")
                    render_factor.value = 35
                with gr.Column():
                    video_output = gr.Video(label="修复后的视频" if isChineese else "Repaired video", interactive=False)
            run_button = gr.Button(label="Run")
            run_button.click(inputs=[video_input,render_factor],outputs=[video_output],fn=wrap_queued_call(process_image))

    return [(ui,"DeOldify","DeOldify")]

script_callbacks.on_ui_tabs(deoldify_tab)