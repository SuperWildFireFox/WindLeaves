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
