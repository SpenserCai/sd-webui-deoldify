'''
Author: SpenserCai
Date: 2023-07-28 15:49:52
version: 
LastEditors: SpenserCai
LastEditTime: 2023-07-28 15:52:12
Description: file content
'''
from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
render_factor=50

vis = get_image_colorizer(render_factor=render_factor, artistic=True)

x = vis.get_transformed_image('./test_images/1.png', render_factor=35, compare=True)