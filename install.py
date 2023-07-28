'''
Author: SpenserCai
Date: 2023-07-28 14:37:09
version: 
LastEditors: SpenserCai
LastEditTime: 2023-07-28 15:39:22
Description: file content
'''
import launch
# 从huggingface下载权重

for dep in ['wandb','fastai==1.0.60', 'tensorboardX', 'ffmpeg', 'ffmpeg-python', 'yt-dlp', 'opencv-python','Pillow']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for DeOldify extension")
