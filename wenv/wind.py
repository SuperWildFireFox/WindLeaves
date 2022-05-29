from util.net import *

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
