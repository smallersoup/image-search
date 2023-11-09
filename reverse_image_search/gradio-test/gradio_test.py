# -*- coding: utf-8 -*-
# @Time : 2023/11/07 13:48

import os
import gradio as gr

def get_img_lits(img_dir):
        imgs_List=[ os.path.join(img_dir,name) for name in sorted(os.listdir(img_dir)) if  name.endswith(('.png','.jpg','.webp','.tif','.jpeg'))]
        return imgs_List


def input_text(dir):
    img_paths_list=get_img_lits(dir) #  注意传入自定义的web
    # 结果为 list,里面对象可以为
    return img_paths_list

'''
 gr.Gallery()
 必须要使用.style()才能控制图片布局
'''
demo = gr.Interface(
    fn=input_text,
    inputs=gr.inputs.Textbox(default='./images'),
    # inputs=gr.inputs.Image(type="pil"),
    outputs=gr.Gallery(
            label="最终的结果图片").style(height='auto',columns=10),
    title='显示某路径下的所有图片的缩略图23.10.12',
)
if __name__ == "__main__":

    demo.launch()