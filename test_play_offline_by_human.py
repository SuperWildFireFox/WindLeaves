import os

from selenium import webdriver
from util.net import get_chromedriver

get_chromedriver()

root_dir = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
chop = webdriver.ChromeOptions()
chop.add_argument("--disable-web-security")

driver = webdriver.Chrome(options=chop)

OFFLINE_GAME_URL = "file:///{}/offline_game/game/bczhc.github.io-master/wind-game/index.html".format(root_dir)
driver.get(OFFLINE_GAME_URL)
input()
