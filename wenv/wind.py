import os
import random
import time

import cv2
import numpy as np
from selenium import webdriver

from util.net import get_chromedriver
from util.image import convert_bs64_to_array
from wenv.hack_code import get_keydown_js, get_keyup_js

ACTIONS = [
    ['noop'],
    ['right'],
    ['right', 'up'],
    ['right', 'catch'],
    ['left'],
    ['left', 'up'],
    ['left', 'catch'],
    ['up'],
    ['catch'],
]

ACTION_KET_MAP = {"noop": None,
                  "left": 'a',
                  "right": 'd',
                  "up": 'w',
                  "down": 's',
                  "catch": 'k',
                  }
# 游戏原始图像大小
GAME_ORIGIN_SIZE = (1920, 360)
# 决定了离线训练时chrome窗口大小，美观起见
OFFLINE_CHROME_WIDTH = 640
OFFLINE_CHROME_HEIGHT = 120 + 100  # 100 is fix
PLAYER_VIEW_SHAPE = (3 * 360, 360)
PLAYER_VIEW_SPLIT = (1, 3)
# 缩小采样比例
DOWNSAMPLE_SCALE = 4


class GameState:
    CurrentStates = ["Air", "Climb", "Ground"]
    PositionStates = ["LeftTorch", "Mid", "RightTorch"]
    StemTypes = ["stem", "stem_trap", "stem_lucky", "stem_lucky_end"]
    GroundTypes = ["leaf", "leaf_trap", "leaf_lucky", "leaf_lucky_end"]
    GameStates = ['MainMenu', 'InGame', 'Paused', 'EndPage']
    PlayerPositionXRange = [-980, 980]  # 非强制性指标
    PlayerPositionYRange = [-220, 220]
    PlayerVelocityXRange = [-3, 5.5]
    BaseSpeeds = [-0.06, -0.09, -0.12]

    class Leaves:
        LeafTypes = ["leaf", "leaf_trap", "leaf_lucky"]
        LeafVelocityRange = [-1, 1]

        def __init__(self, leaf_type, position_x, position_y, velocity_y):
            if leaf_type in GameState.Leaves.LeafTypes:
                self.leaf_type = leaf_type
            else:
                raise Exception("Unknown leaf_type {}".format(leaf_type))
            self.position_x, self.position_y = position_x, position_y
            self.velocity_y = velocity_y

        def is_visible(self, view_range):
            return view_range[0] < self.position_x < view_range[
                1] and -GAME_ORIGIN_SIZE[0] / 2 < self.position_x < GAME_ORIGIN_SIZE[0] / 2

    def __init__(self,
                 current_state,
                 position_x,
                 position_y,
                 ground_id,
                 ground_type,
                 lock_ground,
                 stem_id,
                 stem_type,
                 lock_stem,
                 velocity_x,
                 velocity_y,
                 score,
                 base_speed,
                 game_state,
                 leaves_array):
        if current_state in GameState.CurrentStates:
            self.current_state = current_state
        else:
            raise Exception("Unknown current_state {}".format(current_state))
        self.posx, self.posy = position_x, position_y
        self.ground_id = ground_id
        if self.ground_id is not None:
            if ground_type in GameState.GroundTypes:
                self.ground_type = ground_type
            else:
                raise Exception("Unknown ground_type {}".format(ground_type))
            if lock_ground == False and self.ground_type == "leaf_lucky":
                self.ground_type = "leaf_lucky_end"
        self.stem_id = stem_id
        if self.stem_id is not None:
            if stem_type in GameState.StemTypes:
                self.stem_type = stem_type
            else:
                raise Exception("Unknown stem_type {}".format(stem_type))
            if lock_stem == False and self.stem_type == "stem_lucky":
                self.stem_lucky = "stem_lucky_end"
        self.velocity_x, self.velocity_y = velocity_x, velocity_y
        self.score = score
        if base_speed in GameState.BaseSpeeds:
            self.base_speed_idx = GameState.BaseSpeeds.index(base_speed)
        else:
            raise Exception("Unknown base_speed {}".format(base_speed))
        if game_state in GameState.GameStates:
            self.game_state = game_state
        else:
            raise Exception("Unknown game_state {}".format(game_state))
        self.leaves_array = leaves_array
        self.leaves_obj = []
        view_range = self.get_view_range()
        self.pad_size = 0
        if view_range[0] < -GAME_ORIGIN_SIZE[0] / 2:
            self.position_state = "LeftTorch"
            self.pad_size = abs(view_range[0] - (-GAME_ORIGIN_SIZE[0] / 2))
        elif view_range[1] > GAME_ORIGIN_SIZE[0] / 2:
            self.position_state = "RightTorch"
            self.pad_size = abs(view_range[1] - (GAME_ORIGIN_SIZE[0] / 2))
        else:
            self.position_state = "Mid"
        self.pad_size = int(self.pad_size)
        for leaf in leaves_array:
            leaf_obj = GameState.Leaves(leaf[0], leaf[1], leaf[2], leaf[3])
            if leaf_obj.is_visible(view_range):
                self.leaves_obj.append(leaf)

    # 根据视窗大小与左右视窗比例计算画面裁剪范围
    def get_view_range(self):
        left_length = PLAYER_VIEW_SHAPE[0] * PLAYER_VIEW_SPLIT[0] / (PLAYER_VIEW_SPLIT[0] + PLAYER_VIEW_SPLIT[1])
        right_length = PLAYER_VIEW_SHAPE[0] - left_length
        return [self.posx - left_length, self.posx + right_length]


class Game:
    OFFLINE = "offline"
    ONLINE = "online"
    ACTION_RANDOM = "action_random"
    ACTION_PPO = "action_ppo"

    def __init__(self, mode=OFFLINE, action=ACTION_RANDOM, layout_pos=[0, 0]):
        if mode in [Game.OFFLINE, Game.ONLINE]:
            self.mode = mode
        else:
            raise NotImplemented
        if action in [Game.ACTION_RANDOM, Game.ACTION_PPO]:
            self.action = action
        else:
            raise NotImplemented
        self.layout_pos = layout_pos

        # 核心参数
        # fps仅供参考，实际fps会比设定值小，大约小一倍
        self.fps_r = 1 / 24

        # 全局对象
        self.driver = None
        self.game_state = None
        self.image_data = None
        self.pre_step_terminal = False

    # 初始化离线环境
    def init_offline_env(self):
        self.init_offline_driver()
        # 只有当在WindLeaves为根目录的情况下这个才正确
        root_dir = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
        game_dir = "file:///{}/offline_game/game/bczhc.github.io-master/wind-game/index.html".format(root_dir)
        self.driver.get(game_dir)
        time.sleep(0.5)
        self.game_state = self.info_get_game_state()
        self.image_data = self.info_get_game_image(self.game_state)
        self.control_pause_game()
        # while True:
        #     self.game_state = self.info_get_game_state()
        #     self.info_get_game_image(self.game_state)
        #     if self.game_state.game_state == "EndPage":
        #         self.reset_offline_env()
        #     else:
        #         time.sleep(self.fps_r)

    # 重置离线版游戏环境
    def reset_offline_env(self):
        self.pre_step_terminal = False
        self.control_reset_game()
        time.sleep(0.1)
        self.game_state = self.info_get_game_state()
        assert self.game_state.game_state == "InGame"
        self.image_data = self.info_get_game_image(self.game_state)
        self.control_pause_game()

    # 初始化离线driver
    def init_offline_driver(self):
        get_chromedriver()
        chop = webdriver.ChromeOptions()
        chop.add_argument("--disable-web-security")
        # 关闭测试提示
        chop.add_experimental_option("excludeSwitches", ['enable-automation'])
        self.driver = webdriver.Chrome(options=chop)
        # 设置离线游戏窗口位置
        self.driver.set_window_size(width=OFFLINE_CHROME_WIDTH, height=OFFLINE_CHROME_HEIGHT)
        self.driver.set_window_position(*self.layout_pos)

    # 通过调用该函数操控游戏
    # action为单一数值，0开始
    def step(self, action):
        if self.pre_step_terminal:
            self.reset_offline_env()
        self.control_resume_game()
        action_chars = []
        for act in ACTIONS[action]:
            action_chars.append(ACTION_KET_MAP[act])
        js1 = ""
        js2 = ""
        if action_chars[0] is not None:
            for c in action_chars:
                js1 += get_keydown_js(c)
                js2 += get_keyup_js(c)
            self.tool_execute_script(js1)
        time.sleep(self.fps_r)
        if action_chars[0] is not None:
            self.tool_execute_script(js2)
        self.game_state = self.info_get_game_state()
        self.image_data = self.info_get_game_image(self.game_state)
        if self.game_state.game_state == "EndPage":
            self.pre_step_terminal = True
        self.control_pause_game()
        return self.game_state, self.image_data

    # 一个娱乐的，随机操控的策略
    def random_move(self):
        assert self.action == Game.ACTION_RANDOM
        actions_num = [i for i in range(len(ACTIONS))]
        while True:
            action = random.choice(actions_num)
            self.step(action)

    # 工具函数，对driver执行js（因为可能会涉及到锁所以单独拎出来）
    def tool_execute_script(self, js, obj=None):
        if obj is None:
            res = self.driver.execute_script(js)
        else:
            res = self.driver.execute_script(js, obj)
        return res

    # 暂停游戏
    def control_pause_game(self):
        js = "gameInst.pause();"
        self.tool_execute_script(js)

    # 恢复游戏
    def control_resume_game(self):
        js = "gameInst.resume();"
        self.tool_execute_script(js)

    # 重启游戏
    def control_reset_game(self):
        js = "gameInst.start();"
        self.tool_execute_script(js)

    # 获取游戏全局信息
    def info_get_game_state(self):
        js = """
        return [gameInst.player.state['currentState'],
        gameInst.player._position,
        gameInst.player.ground?.id,
        gameInst.player.ground?.label,
        gameInst.player.stem?.id,
        gameInst.player.stem?.label,
        gameInst.player.lockGround,
        gameInst.player.lockStem,
        gameInst.player.body.velocity,
        hack_scoreboard[0].score,
        hack_scoreboard[0].baseSpeed,
        gameInst.gameState.currentState,
        gameInst.shamrocks.hack_all_array
        ];
        """
        states = self.tool_execute_script(js)
        game_state = GameState(current_state=states[0],
                               position_x=states[1][0],
                               position_y=states[1][1],
                               ground_id=states[2],
                               ground_type=states[3],
                               stem_id=states[4],
                               stem_type=states[5],
                               lock_ground=states[6],
                               lock_stem=states[7],
                               velocity_x=states[8]['x'],
                               velocity_y=states[8]['y'],
                               score=states[9],
                               base_speed=states[10],
                               game_state=states[11],
                               leaves_array=states[12],
                               )
        return game_state

    def info_get_game_image(self, game_state, show_image=False):
        # 获取图像
        js1 = "hack_canvas_require_flag = true"
        js2 = "return hack_canvas_base64"
        self.driver.execute_script(js1)
        time.sleep(1 / 30)
        bs64 = self.driver.execute_script(js2)
        img = convert_bs64_to_array(bs64)
        # 裁剪图像
        view_range = self.game_state.get_view_range()
        if game_state.position_state == "LeftTorch":
            view_range[0] = -GAME_ORIGIN_SIZE[0] / 2
        elif game_state.position_state == "RightTorch":
            view_range[1] = GAME_ORIGIN_SIZE[0] / 2
        view_range[0] = int(view_range[0] + GAME_ORIGIN_SIZE[0] / 2)
        view_range[1] = int(view_range[1] + GAME_ORIGIN_SIZE[0] / 2)
        img = img[:, view_range[0]:view_range[1]]
        if game_state.position_state == "LeftTorch":
            pad_data = np.zeros((img.shape[0], game_state.pad_size, img.shape[2]), np.uint8)
            img = np.concatenate((pad_data, img), 1)
        elif game_state.position_state == "RightTorch":
            pad_data = np.zeros((img.shape[0], game_state.pad_size, img.shape[2]), np.uint8)
            img = np.concatenate((img, pad_data), 1)
        if show_image:
            # 360,1080,3
            cv2.imshow("test", img)
            cv2.waitKey(1)
        return img
