<!--
 * @Author: SpenserCai
 * @Date: 2023-07-28 14:35:35
 * @version: 
 * @LastEditors: SpenserCai
 * @LastEditTime: 2023-08-07 17:00:21
 * @Description: file content
-->
# DeOldify for Stable Diffusion WebUI

This is an extension for StableDiffusion's [AUTOMATIC1111 web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows colorize of old photos and old video. It is based on [deoldify](https://github.com/jantic/DeOldify).

![example](examples/demo.jpeg)

## Compatibility

### OS

<!--制作一个表格显示操作系统的兼容性，Windows不确定，linux兼容-->
| OS | Compatibility | Remark |
| :----: | :----: | :----: |
| Windows 11 | ✅ | Thank for [@w-e-w](https://github.com/w-e-w) test |
| Linux | ✅ | |


### Pytorch
<!--制作一个表格显示Pytorch版本的兼容性-->
| Version | Compatibility | Remark |
| :----: | :----: | :----: |
| <=1.13.1+cu117 | ✅ | |
| 2.1.0.dev20230711+rocm5.5  | ✅ | Thanks for [@fgtm2023](https://github.com/fgtm2023) test | 
| 2.0.1+cu118 | ✅ | Thank for [@w-e-w](https://github.com/w-e-w) test |

### Other
If you have tested other systems or Pytorch during use, please submit a PR and attach a screenshot of the successful operation. Thank you

## Installation
In web-ui, go to the "Extensions" tab and use this URL https://github.com/SpenserCai/sd-webui-deoldify in the "install from URL" tab.

2023-08-05：Support install from Extensions list！！！

If your network is not good, you can download the extension from [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/spensercai/DeOldify)

## Usage
1.To completely exit webui, you need to add `--disable-safe-unpickle` at startup.

2.In web-ui, go to the "Extra" tab and select "DeOldify" checkbox.

3.Upload the old photo you want to colorize.

## Application Scenario
Combining Upscale, GFPGAN, and Denoldify for old photo restoration effects

| Before | After |
| :----: | :----: |
| <img src="examples/before.jpeg" alt="before" align=center /> | <img src="examples/after.jpeg" alt="after" align=center /> |

## Video Colorization

<img src="examples/video_demo.jpeg" alt="video" align=center />

<hr/>

In "DeOldify" tab, upload the video you want to colorize,set "Render Factor" and click "Run".
Now,colorization need long time,please wait patiently.

You need install `ffmpeg`
```bash
sudo apt install ffmpeg
```

### Before
<video src="examples/video_before.mp4" controls="controls" width="500" height="300">Your browser does not support the video tag.</video>

### After
<video src="examples/video_after.mp4" controls="controls" width="500" height="300">Your browser does not support the video tag.</video>


## TODO
- [x] Support video colorization
- [ ] Support repair options
- [ ] Support for simultaneous generation of images with different Render Factor values and Artistic on/off like “X/Y/Z Script” [#2](https://github.com/SpenserCai/sd-webui-deoldify/issues/2)
- [ ] Support need not to add `--disable-safe-unpickle` at startup [#5](https://github.com/SpenserCai/sd-webui-deoldify/issues/5)



