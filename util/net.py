import time
import traceback

import chromedriver_autoinstaller
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from wenv.hack_code import refound_js_code

BILI_URL = "https://www.bilibili.com/"
BILI_URL2 = "https://www.bilibili.com/v/ent/"


def get_chromedriver():
    try:
        chromedriver_autoinstaller.install()
    except:
        pass


def expand_shadow_element(driver, element):
    shadow_root = driver.execute_script('return arguments[0].shadowRoot', element)
    return shadow_root


def enter_online_game(driver):
    # driver.get(BILI_URL)
    driver.get(BILI_URL2)
    WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'login-btn'))
    )
    lg_ui = driver.find_element(By.CLASS_NAME, "v-popover-content")
    other_ui = driver.find_element(By.CLASS_NAME, "bili-header__channel")
    ActionChains(driver).move_to_element(lg_ui).perform()
    ActionChains(driver).move_to_element(other_ui).perform()
    time.sleep(1)
    bf_game = driver.find_element(by=By.XPATH, value='//*[@id="i_cecream"]/div/div[1]/div/div[2]/div[3]')
    bf_game.click()

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "banner-game"))
    )
    game_root = expand_shadow_element(driver, driver.find_element(by=By.CLASS_NAME, value="banner-game"))
    cls = 0
    while cls < 11:
        try:
            fst_buttons = game_root.find_element(by=By.CLASS_NAME, value="guide-container").find_element(
                by=By.CLASS_NAME,
                value="option-bubble")
            fst_buttons.click()
            cls += 1
        except NoSuchElementException:
            pass


def refound_game(driver):
    js = refound_js_code
    driver.execute_script(js)
