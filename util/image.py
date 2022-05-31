import base64
import time

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


# 图片归一化
def NormalizeImage(image, resize_size, show_image=False, save_image=True):
    """
    图片归一化
    :param image: hwc
    :param resize_size: wh
    :param show_image: debug
    :param save_image: debug
    :return: 1,1,w,h
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_size)
    if show_image:
        cv2.imshow("test", image)
        cv2.waitKey(1)
    if save_image:
        cv2.imwrite("dump/{}.png".format(time.time()), image)
    image = image[None, None, :, :].astype(np.float32) / 255
    # print(image.shape)
    return image
