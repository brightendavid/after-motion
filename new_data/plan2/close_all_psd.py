#!/usr/bin/env python
# -*- coding:utf-8 -*-
from photoshop.api import SaveOptions
from win32com.client import Dispatch, GetActiveObject, GetObject

app = GetActiveObject("Photoshop.Application")
while len(app.documents) > 0:
    # 调用[document]对象的[close]方法，关闭文档。[close]方法里的参数保证关闭文档时，不再保存文档。
    app.activeDocument.close(SaveOptions.DoNotSaveChanges)
