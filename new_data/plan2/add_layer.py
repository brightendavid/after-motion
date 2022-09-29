#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Import a image as a artLayer.


from photoshop import Session

with Session() as ps:
    layer = ps.active_document.activeLayer
    print(f"old name: {layer.name}")
    layer.name = "new name"
    print(f"new name: {layer.name}")

"""

# Import local modules
# 这个脚本可以作为生成动作记录的前置步骤
# 这个文件的作用就是输入2张图   src  和mask       输出
import os
# 输入为图片
from photoshop import Session
from PIL import Image
import cv2 as cv
from photoshop.api import SaveOptions
from win32com.client import Dispatch, GetActiveObject, GetObject


def add_layers(src, mask, save_dir):
    # 运行中会输出PSD名称，交互性必须要有
    images = {
        "src": src,
        "mask": mask
    }
    name = images["src"].split("\\")[-1][:-4]
    name = name + ".psd"
    print(name)
    save_dir = save_dir
    # src
    src = Image.open(images["src"])
    # print(src.size)
    a, b = src.size

    src.save("../pic/src.png")

    mask = Image.open(images["mask"])
    mask.save("../pic/mask.png")

    images = {
        # 保存到固定目录   实际的目的是图片的改名  从而改变图层的名称
        "src": r"C:\Users\brighten\Desktop\后处理工作\new_data\pic\src.png",
        "mask": r"C:\Users\brighten\Desktop\后处理工作\new_data\pic\mask.png"
    }

    with Session(action="new_document") as ps:
        # 添加两个图层  最下层为背景图层
        doc = ps.active_document
        ps.app.preferences.rulerUnits = ps.Units.Pixels
        ps.app.documents.add(a, b, name="my_new_document")
        desc = ps.ActionDescriptor
        desc.putPath(ps.app.charIDToTypeID("null"), images["src"])  # src

        ps.app.executeAction(ps.app.charIDToTypeID("Plc "), desc)

        desc2 = ps.ActionDescriptor
        desc2.putPath(ps.app.charIDToTypeID("null"), images["mask"])  # 蒙版
        event_id = ps.app.charIDToTypeID("Plc ")  # `Plc` need one space in here.
        ps.app.executeAction(ps.app.charIDToTypeID("Plc "), desc2)

        # 保存方式   为png    最终步骤   还可为pdf   psd 格式
        # png = r'C:\Users\brighten\Desktop\test\src\1.png'
        # options = ps.PNGSaveOptions()
        # image_path = png
        # doc.saveAs(image_path, options, True)

        # 保存为psd 格式

        psd_file = os.path.join(save_dir, name)
        options = ps.PhotoshopSaveOptions()
        doc.saveAs(psd_file, options, True)
        doc.close()
        ps.app.activeDocument.close(SaveOptions.DoNotSaveChanges)
        # ps.alert("Task done!")
        # ps.app.activeDocument.close()
        # ps.echo(doc.activeLayer)


if __name__ == "__main__":
    src = r"C:\Users\brighten\Desktop\test\src\COD10K-CAM-3-Flying-61-Katydid-4038.jpg"
    mask = r"C:\Users\brighten\Desktop\test\src\COD10K-CAM-3-Flying-61-Katydid-4038.png"
    save_dir = r"C:\Users\brighten\Desktop\test\src"
    add_layers(src, mask, save_dir)
