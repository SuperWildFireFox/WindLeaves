import os
import time

from selenium import webdriver

from util.net import get_chromedriver

ACTIONS = [
    ['noop'],
    ['right'],
    ['left'],
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


class PlayerState:
    CurrentStates = ["Air", "Clime", "Ground"]
    StemTypes = ["stem", "stem_trap", "stem_lucky", "stem_lucky_end"]
    GroundTypes = ["leaf", "leaf_lucky", "leaf_lucky_end"]
    GameStates = ['MainMenu', 'InGame', 'Paused', 'EndPage']
    PlayerPositionXRange = [-980, 980]  # 非强制性指标
    PlayerPositionYRange = [-220, 220]
    PlayerVelocityXRange = [-3, 5.5]
    BaseSpeeds = [-0.06, -0.09, -0.12]

    def __init__(self, current_state, position, ground_id, ground_type, stem_id, stem_type, lock_ground, lock_stem, ):
        if current_state in PlayerState.CurrentStates:
            self.current_state = current_state
        else:
            raise NotImplemented
        self.posx, self.posy = position
        # self.


class Game:
    OFFLINE = "offline"
    ONLINE = "online"
    ACTION_RANDOM = "action_random"
    ACTION_PPO = "action_ppo"

    def __init__(self, mode=OFFLINE, action=ACTION_RANDOM, layout_pos=[0, 0], pipe=None):
        if mode in [Game.OFFLINE, Game.ONLINE]:
            self.mode = mode
        else:
            raise NotImplemented
        if action in [Game.ACTION_RANDOM, Game.ACTION_PPO]:
            self.action = action
        else:
            raise NotImplemented
        self.layout_pos = layout_pos
        if self.action == Game.ACTION_PPO:
            assert pipe is not None
            self.inner_pipe, self.outer_pipe = pipe

        # 全局对象
        self.driver = None

    # 初始化离线环境
    def init_offline_env(self):
        self.init_offline_driver()
        # 只有当在WindLeaves为根目录的情况下这个才正确
        root_dir = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
        game_dir = "file:///{}/offline_game/game/bczhc.github.io-master/wind-game/index.html".format(root_dir)
        self.driver.get(game_dir)
        time.sleep(5)
        while True:
            tt = time.time()
            print(self.info_get_player_info())
            print("0", time.time() - tt)

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

    def info_get_player_info(self):
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
        hack_scoreboard[0].base_speed,
        gameInst.gameState.currentState,
        gameInst.shamrocks.hack_all_array
        ];
        """
        return self.tool_execute_script(js)
