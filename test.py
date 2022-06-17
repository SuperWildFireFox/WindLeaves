import json
import os
import pickle
import shutil
import time
import traceback
from threading import Thread

import cv2
import numpy as np
from tensorboardX import SummaryWriter

from model import Actor
from util.data import generate_feature1, generate_feature2
from util.image import NormalizeImage
from util.log import Logger
from wenv.wind import Game
import torch
import multiprocessing as mp
from mss.windows import MSS as mss
import torch.nn.functional as F


class GameTest:
    def __init__(self, mode=Game.OFFLINE, training=True, **kwargs):
        self.mode = mode
        if self.mode == Game.OFFLINE:
            if training:
                self.test_on_training(**kwargs)
            else:
                self.test_offline(**kwargs)
        elif self.mode == Game.ONLINE:
            self.can_capture = False
            self.test_online_(**kwargs)
        else:
            raise NotImplemented

    def test_online(self, weight_shape_path, weight_actor_path):
        game = Game(mode="online", action=Game.ACTION_PPO)
        with open(weight_shape_path, 'r') as fp:
            model_input_shape = json.loads(fp.read())
        actor = Actor(model_input_shape["NUM_ACTION"],
                      model_input_shape["IMAGE_SIZE"],
                      model_input_shape["feature1_length"],
                      model_input_shape["feature2_length"])
        actor.load_state_dict(torch.load(weight_actor_path, map_location=torch.device('cpu')))
        actor.eval()
        game.init_online_env()
        image, game_state, reward = game.step(0)
        terminal = False
        while not terminal:
            image = torch.from_numpy(NormalizeImage(image, model_input_shape["IMAGE_SIZE"], show_image=True))
            feature1 = generate_feature1([game_state], model_input_shape["feature1_length"])
            feature2 = generate_feature2([game_state], model_input_shape["feature2_length"])
            mask = torch.cat(
                [torch.from_numpy(gs.get_action_mask()[None, :]).type(torch.BoolTensor) for gs in [game_state]])
            logic = actor(image, feature1, feature2, mask)
            action = torch.argmax(logic).item()
            image, game_state, reward = game.step(action)
            terminal = game_state.game_state == "EndPage"
            if terminal:
                print("Score is {}！".format(game_state.score))
        input()

    def test_online_(self, weight_shape_path, weight_actor_path, goal):
        game = Game(mode="online", action=Game.ACTION_PPO)
        with open(weight_shape_path, 'r') as fp:
            model_input_shape = json.loads(fp.read())
        actor = Actor(model_input_shape["NUM_ACTION"],
                      model_input_shape["IMAGE_SIZE"],
                      model_input_shape["feature1_length"],
                      model_input_shape["feature2_length"])
        actor.load_state_dict(torch.load(weight_actor_path, map_location=torch.device('cpu')))
        actor.eval()
        game.init_online_env()
        game.wait_online_ready()
        print("now")
        monitor = game.get_online_monitor()
        cap_thread = Thread(target=self.capture_loop, args=(monitor,))
        cap_thread.start()
        image, game_state, reward = game.step(0)
        start_time = time.time()
        if os.path.exists("dump"):
            shutil.rmtree("dump")
        os.makedirs("dump/norm")
        os.makedirs("dump/ori")
        self.can_capture = True
        self.ori_buff = []
        pro_buff = []
        act_list = []
        inputs = []
        while True:
            input_t = []
            image, image2 = NormalizeImage(image, model_input_shape["IMAGE_SIZE"], debug=True)
            pro_buff.append(image2)
            image = torch.from_numpy(image)
            feature1 = generate_feature1([game_state], model_input_shape["feature1_length"])
            input_t.append(feature1.numpy())
            feature2 = generate_feature2([game_state], model_input_shape["feature2_length"])
            input_t.append(feature2.numpy())
            mask = torch.cat(
                [torch.from_numpy(gs.get_action_mask()[None, :]).type(torch.BoolTensor) for gs in [game_state]])
            input_t.append(mask.numpy())
            logic = actor(image, feature1, feature2, mask)
            act_list.append(F.softmax(logic, dim=1).detach().numpy())
            inputs.append(input_t)
            action = torch.argmax(logic).item()
            image, game_state, reward = game.step(action)
            terminal = game_state.game_state == "EndPage"
            if terminal:
                print("{} Score is {}！".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                                               game_state.score))
                if game_state.score > goal:
                    print("We did it!")
                    time.sleep(3)
                    self.can_capture = False
                    if os.path.exists("dump"):
                        shutil.rmtree("dump")
                    os.makedirs("dump/norm")
                    os.makedirs("dump/ori")
                    for i, img in enumerate(self.ori_buff):
                        cv2.imwrite("dump/ori/{}.png".format(i), img)
                    for i, img in enumerate(pro_buff):
                        cv2.imwrite("dump/norm/{}.png".format(i), img)
                    with open("dump/actions.pkl", 'wb') as fp:
                        pickle.dump(act_list, fp)
                    with open("dump/inputs.pkl", 'wb') as fp:
                        pickle.dump(inputs, fp)
                    return
                else:
                    self.can_capture = False
                    self.ori_buff = []
                    pro_buff = []
                    act_list = []
                    inputs = []
                    time.sleep(1)
                    done = False
                    while not done:
                        try:
                            shutil.rmtree("dump")
                            time.sleep(1)
                            os.makedirs("dump/norm")
                            os.makedirs("dump/ori")
                            done = True
                        except:
                            traceback.print_exc()
                            pass
                    self.can_capture = True
                    game.init_online_env()
                game.wait_online_ready()
                image, game_state, reward = game.step(0)

    def capture_loop(self, monitor):
        sct = mss()
        while True:
            if self.can_capture:
                img = np.array(sct.grab(monitor))
                self.ori_buff.append(img)
                time.sleep(1 / 36)
            else:
                time.sleep(1 / 12)

    def test_offline(self, weight_shape_path, weight_actor_path):
        game = Game(mode="offline", action=Game.ACTION_PPO)
        game.init_offline_env()
        with open(weight_shape_path, 'r') as fp:
            model_input_shape = json.loads(fp.read())
        actor = Actor(model_input_shape["NUM_ACTION"],
                      model_input_shape["IMAGE_SIZE"],
                      model_input_shape["feature1_length"],
                      model_input_shape["feature2_length"])
        actor.load_state_dict(torch.load(weight_actor_path, map_location=torch.device('cpu')))
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
                print(game_state.score)

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
            logger.write("scores list: {}".format(str(score_list)), print_=False)

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
