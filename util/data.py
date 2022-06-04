import io
import json
import os
import sys

import numpy as np
import torch

from wenv.wind import GameState


def generate_feature1(game_state_list, feature1_length):
    res_list = []
    for game_state in game_state_list:
        states = np.zeros(feature1_length, dtype=np.float32)
        states[0] = game_state.posx_normalized
        states[1] = game_state.posy_normalized
        states[2] = game_state.velocity_x_normalized
        states[3] = game_state.velocity_y_normalized
        states[4 + game_state.current_state_idx] = 1  # 3
        states[7 + game_state.position_state_idx] = 1  # 3
        if game_state.ground_type_idx is not None:
            states[10 + game_state.ground_type_idx] = 1  # 4
        if game_state.stem_type_idx is not None:
            states[14 + game_state.stem_type_idx] = 1  # 4
        states[18 + game_state.base_speed_idx] = 1  # 3
        res_list.append(states)
    return torch.from_numpy(np.array(res_list))


def generate_feature2(game_state_list, feature2_length, fix_length=16):
    res_list = []
    for game_state in game_state_list:
        group_list = []
        pad_size = fix_length - len(game_state.leaves_obj)
        assert pad_size >= 0
        for leaf in game_state.leaves_obj:
            states = np.zeros(feature2_length, dtype=np.float32)
            states[0] = leaf.distance_x_normalized
            states[1] = leaf.distance_y_normalized
            states[2] = leaf.velocity_y_normalized
            states[3 + leaf.leaf_type_idx] = 1  # 3
            group_list.append(states)
        for i in range(pad_size):
            states = np.zeros(feature2_length, dtype=np.float32)
            group_list.append(states)
        res_list.append(torch.from_numpy(np.array(group_list)[None, :, :]))
    return torch.cat(res_list)


def compute_gae(value, next_value, reward, done, gamma, gae_lambda):
    # shape of array: [T, B]
    # return returns, advantange, mask
    # code from https://github.com/sail-sg/envpool/
    T, B = value.shape
    mask = (1.0 - done) * (gamma * gae_lambda)
    flag = False
    value_tp1 = next_value
    gae_tp1 = 0
    delta = reward - value
    adv = []
    for t in range(T - 1, -1, -1):
        adv.append(delta[t] + gamma * value_tp1 * (1 - done[t]) + mask[t] * gae_tp1)
        mask[t] = (done[t] == 1 | flag)
        gae_tp1 = adv[-1]
        value_tp1 = value[t]
        flag = True
    return torch.cat(adv[::-1])


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def lr_decay(args, episode, actor_optimizer, critic_optimizer):
    lr_now = args.lr * (1 - episode / args.max_train_episode)
    for p in actor_optimizer.param_groups:
        p['lr'] = lr_now
    for p in critic_optimizer.param_groups:
        p['lr'] = lr_now
