import base64

import cv2
import numpy as np


# 将base64字符串转化为np数组，hwc格式
def convert_bs64_to_array(bs64):
    if bs64.startswith("data:image/png;base64,"):
        bs64 = bs64.replace("data:image/png;base64,", "")
    img_data = np.frombuffer(base64.b64decode(bs64), np.uint8)
    img_data = cv2.imdecode(img_data, cv2.COLOR_RGB2BGR)
    # return hwc
    return img_data

