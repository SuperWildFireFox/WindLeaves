import os
import time

import cv2
from selenium import webdriver
from util.net import get_chromedriver
from util.image import convert_bs64_to_array

fps = 1 / 24
js1 = "return hack_canvas_base64"
js2 = "hack_canvas_require_flag = true"

get_chromedriver()

root_dir = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
chop = webdriver.ChromeOptions()
chop.add_argument("--disable-web-security")

driver = webdriver.Chrome(options=chop)

OFFLINE_GAME_URL = "file:///{}/offline_game/game/bczhc.github.io-master/wind-game/index.html".format(root_dir)
driver.get(OFFLINE_GAME_URL)

i = 0
while True:
    # 约0.03-0.04s
    driver.execute_script(js2)
    time.sleep(1 / 61)
    bs64 = driver.execute_script(js1)
    i += 1
    if bs64:
        img = convert_bs64_to_array(bs64)
        cv2.imwrite("dump/{}.png".format(i), img)
    time.sleep(fps)
