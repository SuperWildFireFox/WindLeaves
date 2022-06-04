# -*- coding: UTF-8 -*-
import json

import numpy as np
from tensorboardX import SummaryWriter

from model import Actor
from util.data import generate_feature1, generate_feature2
from util.image import NormalizeImage
from util.log import Logger
from wenv.control import EnvironmentControl
from wenv.wind import Game
import torch
import multiprocessing as mp


class GameTest:
    def __init__(self, mode=Game.OFFLINE, training=True, **kwargs):
        self.mode = mode
        if self.mode == Game.OFFLINE:
            if training:
                self.test_on_training(**kwargs)
            else:
                pass
        elif self.mode == Game.ONLINE:
            pass
        else:
            raise NotImplemented

    def test_on_training(self, args):
        self.args = args
        self.avg_num = args.test_round
        self.best_score = 0
        self.inner_channel, self.outer_channel = mp.Pipe()
        mp.Process(target=self.test_on_training_loop).start()

    def test_on_training_loop(self):
        writer = SummaryWriter(self.args.summary_path)
        logger = Logger(self.args.log_path_testing)
        self.game_conns, self.control_conns = zip(*[mp.Pipe() for _ in range(self.avg_num)])
        for idx in range(self.avg_num):
            mp.Process(target=self.test_on_training_game_loop, args=(idx,)).start()
        while True:
            episode, model_dict = self.inner_channel.recv()
            [control_conn.send(model_dict) for control_conn in self.control_conns]
            score_list = [control_conn.recv() for control_conn in self.control_conns]
            score_list = np.array(score_list)
            mean_score = score_list.mean()
            max_score = score_list.max()
            min_score = score_list.min()
            std_score = score_list.std()
            writer.add_scalar('TEST_mean_score', mean_score, episode)
            writer.add_scalar('TEST_max_score', max_score, episode)
            writer.add_scalar('TEST_min_score', min_score, episode)
            writer.add_scalar('TEST_score_std', std_score, episode)
            mean_score = int(mean_score)
            if mean_score > self.best_score:
                self.best_score = mean_score
                writer.add_scalar("Best Episode", episode, mean_score)
            logger.write("Test: episode: {}, mean score: {}, max score: {}".
                         format(episode, mean_score, max_score))
            logger.write("scores list: {}".format(str(score_list)),print_=False)

    def test_on_training_game_loop(self, idx):
        game = Game(mode="offline", action=Game.ACTION_PPO)
        game.init_offline_env()
        with open("{}/model_input_shape.json".format(self.args.weight_path), 'r') as fp:
            model_input_shape = json.loads(fp.read())
        actor = Actor(model_input_shape["NUM_ACTION"],
                      model_input_shape["IMAGE_SIZE"],
                      model_input_shape["feature1_length"],
                      model_input_shape["feature2_length"])
        while True:
            model_dict = self.game_conns[idx].recv()
            actor.load_state_dict(model_dict)
            actor.eval()
            image, game_state, reward = game.step(0)
            terminal = False
            while not terminal:
                image = torch.from_numpy(NormalizeImage(image, model_input_shape["IMAGE_SIZE"]))
                feature1 = generate_feature1([game_state], model_input_shape["feature1_length"])
                feature2 = generate_feature2([game_state], model_input_shape["feature2_length"])
                mask = torch.cat(
                    [torch.from_numpy(gs.get_action_mask()[None, :]).type(torch.BoolTensor) for gs in [game_state]])
                logic = actor(image, feature1, feature2, mask)
                action = torch.argmax(logic).item()
                image, game_state, reward = game.step(action)
                terminal = game_state.game_state == "EndPage"
                if terminal:
                    self.game_conns[idx].send(game_state.score)
