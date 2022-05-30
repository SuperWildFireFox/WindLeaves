import time

from screeninfo import get_monitors
from wenv.wind import Game, OFFLINE_CHROME_WIDTH, OFFLINE_CHROME_HEIGHT
import multiprocessing as mp
from multiprocessing import Lock as mpLock


class LayoutManager:
    def __init__(self, nums, width, height, sp=10, monitor=2, display_mode="top"):
        self.nums = nums
        self.width = width
        self.height = height
        self._monitor = [m for m in get_monitors()]
        self.monitor = self._monitor[monitor]
        # print("LayoutManager: monitor is", self.monitor)
        self.window_w = self.monitor.width
        self.window_h = self.monitor.height
        self.ori_point = [self.monitor.x, self.monitor.y]
        self.n = 0
        self.sp = sp
        if nums == 1:
            self.layout = (1, 1)
        elif nums == 2:
            self.layout = (2, 1)
        elif nums == 4:
            self.layout = (4, 1)
        elif nums == 8:
            self.layout = (4, 2)
        elif nums == 16:
            self.layout = (8, 2)
        self.start_pos = [-1, -1]
        if display_mode.lower() == "top":
            self.display_mode = "top"
        elif display_mode.lower() == "mid":
            self.display_mode = "mid"
        else:
            raise Exception("Display mode {} not avail".format(display_mode))
        self.calc()

    def calc(self):
        total_x = self.layout[1] * self.width + (self.layout[1] - 1) * self.sp
        step_x = (self.window_w - total_x) // 2
        self.start_pos[0] = self.ori_point[0] + step_x
        if self.display_mode == "top":
            self.start_pos[1] = self.ori_point[1] + step_x
        elif self.display_mode == "mid":
            total_y = self.layout[0] * self.height + (self.layout[0] - 1) * self.sp
            self.start_pos[1] = self.ori_point[1] + (self.window_h - total_y) // 2

    def step(self):
        pos = [0, 0]
        pos[0] = self.start_pos[0] + (self.n % self.layout[1]) * (self.width + self.sp)
        pos[1] = self.start_pos[1] + (self.n // self.layout[1]) * (self.height + self.sp)
        self.n += 1
        return pos


class EnvironmentControl:
    def __init__(self, num_envs, action=Game.ACTION_PPO, use_layout=True, monitor=2):
        self.num_envs = num_envs
        self.action = action
        if use_layout:
            self.layout_manager = LayoutManager(self.num_envs,
                                                OFFLINE_CHROME_WIDTH,
                                                OFFLINE_CHROME_HEIGHT,
                                                monitor=monitor)
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        for idx in range(num_envs):
            if use_layout:
                layout_pos = self.layout_manager.step()
            else:
                layout_pos = (0, 0)
            process = mp.Process(target=self.run, args=(idx, layout_pos,))
            process.start()
            time.sleep(1)

    def run(self, idx, layout_pos):
        game = Game(mode="offline",
                    action=self.action,
                    layout_pos=layout_pos)
        game.init_offline_env()
        while True:
            action = self.env_conns[idx].recv()
            game_state, image_data = game.step(action)
            self.env_conns[idx].send([game_state, image_data])


if __name__ == '__main__':
    EnvironmentControl(16)
