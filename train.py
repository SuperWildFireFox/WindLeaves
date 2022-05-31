import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from model import Actor, Critic
from util.data import generate_feature1, generate_feature2, compute_gae
from util.image import NormalizeImage
from wenv.control import EnvironmentControl
from wenv.wind import ACTIONS, GameState, GAME_ORIGIN_SIZE, DOWNSAMPLE_SCALE, PLAYER_VIEW_SHAPE

# 调试变量
DELETE_WEIGHT = True
DELETE_SUMMARY = True
DUMP_PATH = "dump"  # 垃圾桶区域
NUM_ACTION = len(ACTIONS)
WEIGHT_PATH_ACTOR = ""
WEIGHT_PATH_CRITIC = ""
START_EPISODE = 0
IMAGE_PROCESS_SIZE = PLAYER_VIEW_SHAPE[0] // DOWNSAMPLE_SCALE, PLAYER_VIEW_SHAPE[1] // DOWNSAMPLE_SCALE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='随机数种子（虽然没什么用，程序无法控制javascript内部随机数）')
    parser.add_argument("--weight_path", type=str, default="model", help="权重保存路径")
    parser.add_argument("--summary_path", type=str, default="summary", help="log文件保存路径")
    parser.add_argument("--num_env", type=int, default=4, help="同时训练的环境数")
    parser.add_argument("--monitor", type=int, default=2, help="可视化环境显示的显示屏位置")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--test_step', type=int, default=10, help="训练中测试的间隔")
    parser.add_argument('--max_train_episode', type=int, default=9600, help="最大训练轮次")
    parser.add_argument('--num_collection_steps', type=int, default=32, help="每episode收集数据步数")
    parser.add_argument('--gamma', type=float, default=0.975, help="奖励折扣因子")
    parser.add_argument('--gae_lambda', type=float, default=0.98, help="gae参数")
    parser.add_argument('--train_epoch', type=int, default=4, help="每episode数据的训练轮数")
    parser.add_argument('--batch_split', type=int, default=2, help="每epoch训练时切分数")
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.01, help='计算actor loss时熵的系数')
    parser.add_argument('--mode_save_episode', type=int, default=100, help="训练模型保存间隔")

    # parser.add_argument("--stage", type=int, default=1)epsilon
    # parser.add_argument("--action_type", type=str, default="simple")
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    # parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    # parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    # parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--num_epochs', type=int, default=10)
    # parser.add_argument("--num_local_steps", type=int, default=128)
    # parser.add_argument("--num_global_steps", type=int, default=5e6)
    # parser.add_argument("--num_processes", type=int, default=8)
    # parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    # parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    # parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    # parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置随机数种子
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 如果设置过删除权重，则先把权重文件夹删了
    if DELETE_WEIGHT and os.path.isdir(args.weight_path):
        shutil.rmtree(args.weight_path)
    if not os.path.isdir(args.weight_path):
        os.makedirs(args.weight_path)
    # 如果设置过删除log，删了log文件夹
    if DELETE_SUMMARY and os.path.isdir(args.summary_path):
        shutil.rmtree(args.summary_path)
    if os.path.isdir(DUMP_PATH):
        shutil.rmtree(DUMP_PATH)
    os.makedirs(DUMP_PATH)

    writer = SummaryWriter(args.summary_path)
    envs = EnvironmentControl(args.num_env, monitor=args.monitor)

    # 生成模型与优化器
    feature1_length = 2 + 2 + len(GameState.CurrentStates) + len(GameState.PositionStates) + len(
        GameState.GroundTypes) + len(GameState.StemTypes) + len(GameState.BaseSpeeds)  # 21
    feature2_length = 2 + 1 + len(GameState.Leaves.LeafTypes)  # 6
    actor = Actor(NUM_ACTION, IMAGE_PROCESS_SIZE, feature1_length, feature2_length).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic = Critic(IMAGE_PROCESS_SIZE, feature1_length, feature2_length).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    if WEIGHT_PATH_ACTOR and WEIGHT_PATH_CRITIC and os.path.exists(WEIGHT_PATH_ACTOR) and os.path.exists(
            WEIGHT_PATH_CRITIC):
        print("load weight from {},{}".format(WEIGHT_PATH_ACTOR, WEIGHT_PATH_CRITIC))
        actor.load_state_dict(torch.load(WEIGHT_PATH_ACTOR, map_location=torch.device(device)))
        critic.load_state_dict(torch.load(WEIGHT_PATH_CRITIC, map_location=torch.device(device)))

    # 初始化测试环境
    # p_test = PPOTest(STEP_FRAME_NUM,
    #                  action_type,
    #                  num_actions,
    #                  writer_path=summary_path,
    #                  test_num=5,
    #                  log_path="test_monitor.txt")
    # p_test.start_test_mp()

    # 获得初始特征
    curr_image = []
    curr_game_state = []
    [agent_conn.send(0) for agent_conn in envs.agent_conns]
    for agent_conn in envs.agent_conns:
        data = agent_conn.recv()
        curr_image.append(NormalizeImage(data[0], IMAGE_PROCESS_SIZE))
        curr_game_state.append(data[1])

    # num_env*1*w*h
    curr_image = np.concatenate(curr_image, 0)
    curr_image = torch.from_numpy(curr_image).to(device)
    # num_env*feature1_length
    curr_feature1 = generate_feature1(curr_game_state, feature1_length).to(device)
    # num_env*x*feature1_length
    curr_feature2 = generate_feature2(curr_game_state, feature2_length).to(device)

    curr_mask = torch.cat([torch.from_numpy(game_state.get_action_mask()[None, :]).type(torch.BoolTensor) for
                           game_state in curr_game_state]).to(device)

    curr_episode = 0 if DELETE_WEIGHT else START_EPISODE
    start_time = time.time()

    while curr_episode < args.max_train_episode:
        curr_episode += 1
        imp_log_probs = []
        actions = []
        values = []
        images = []
        features1 = []
        features2 = []
        masks = []
        rewards = []
        terminals = []

        for step in range(args.num_collection_steps):
            images.append(curr_image)
            features1.append(curr_feature1)
            features2.append(curr_feature2)
            masks.append(curr_mask)
            logic = actor(curr_image, curr_feature1, curr_feature2, curr_mask)
            # num_envs*1
            value = critic(curr_image, curr_feature1, curr_feature2)

            values.append(value.squeeze())
            policy = F.softmax(logic, dim=1)
            imp_sample = Categorical(policy)
            # num_envs,
            action = imp_sample.sample()
            imp_log_prob = imp_sample.log_prob(action)
            imp_log_probs.append(imp_log_prob)
            if device == "cuda":
                [agent_conn.send(atc) for agent_conn, atc in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(atc) for agent_conn, atc in zip(envs.agent_conns, action)]
            image, game_state, reward = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            terminal = [gs.game_state == "EndPage" for gs in game_state]
            image = np.concatenate([NormalizeImage(im, IMAGE_PROCESS_SIZE) for im in image], 0)
            # n*c*w*h
            image = torch.from_numpy(image).to(device)
            # num_envs,
            reward = torch.FloatTensor(reward).to(device)
            # num_envs,
            terminal = torch.FloatTensor(terminal).to(device)
            # num_envs*fea1
            feature1 = generate_feature1(game_state, feature1_length).to(device)
            # num_envs*10*fea2
            feature2 = generate_feature2(game_state, feature2_length).to(device)
            # num_env*len(action)
            mask = torch.cat(
                [torch.from_numpy(gs.get_action_mask()[None, :]).type(torch.BoolTensor) for gs in game_state]).to(
                device)

            rewards.append(reward)
            terminals.append(terminal)
            actions.append(action)

            curr_image = image
            curr_mask = mask
            curr_feature1 = feature1
            curr_feature2 = feature2

        _, next_value = actor(curr_image, curr_feature1, curr_feature2, curr_mask), critic(curr_image, curr_feature1,
                                                                                           curr_feature2)
        next_value = next_value.squeeze()
        images = torch.cat(images)
        actions = torch.cat(actions)
        features1 = torch.cat(features1)
        features2 = torch.cat(features2)
        masks = torch.cat(masks)
        imp_log_probs = torch.cat(imp_log_probs).detach()
        values = torch.cat(values).view(-1).detach()

        values_group = values.view(-1, args.num_env).detach()
        rewards = torch.cat(rewards).view(-1, args.num_env).detach()
        terminals = torch.cat(terminals).view(-1, args.num_env).detach()

        adv = compute_gae(values_group, next_value, rewards, terminals, args.gamma, args.gae_lambda).detach()
        R = adv + values

        a_loss_list = []
        c_loss_list = []
        entropy_list = []
        for epoch in range(args.train_epoch):
            indices = torch.randperm(args.num_env * args.num_collection_steps)
            for bs in range(args.batch_split):
                batch_indices = indices[int(bs * (args.num_env * args.num_collection_steps / args.batch_split)): int(
                    (bs + 1) * (args.num_env * args.num_collection_steps / args.batch_split))]
                logic = actor(images[batch_indices], features1[batch_indices], features2[batch_indices],
                              masks[batch_indices])
                value = critic(images[batch_indices], features1[batch_indices], features2[batch_indices])
                policy = F.softmax(logic, dim=1)
                imp_sample = Categorical(policy)
                imp_log_prob = imp_sample.log_prob(actions[batch_indices])
                ratio = torch.exp(imp_log_prob - imp_log_probs[batch_indices])
                adv_batch = adv[batch_indices]
                actor_loss = -torch.mean(torch.min(ratio * adv_batch, torch.clamp(ratio, 1.0 - args.epsilon,
                                                                                  1.0 + args.epsilon) * adv_batch))
                entropy_loss = -torch.mean(imp_sample.entropy())
                entropy_list.append(entropy_loss.item())
                actor_loss = actor_loss + args.beta * entropy_loss
                a_loss_list.append(actor_loss.item())
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                c_loss_list.append(critic_loss.item())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()
        a_loss = np.array(a_loss_list).mean()
        c_loss = np.array(c_loss_list).mean()
        e_loss = np.array(entropy_list).mean()
        writer.add_scalar("TRAIN_actor_loss", a_loss, curr_episode)
        writer.add_scalar("TRAIN_critic_loss", c_loss, curr_episode)
        writer.add_scalar("TRAIN_entropy_loss", e_loss, curr_episode)
        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
              "Episode: {}. Actor loss: {:.4f}. Critic loss: {:.4f}".format(curr_episode, a_loss, c_loss))
        if curr_episode % args.mode_save_episode == 0:
            torch.save(actor.state_dict(), "{}/actor_episode_{}.pth".format(args.weight_path, curr_episode))
            torch.save(critic.state_dict(), "{}/critic_episode_{}.pth".format(args.weight_path, curr_episode))
            # if TEST and curr_episode and curr_episode % TEST_STEP == 0:
            #     state_dict = actor.state_dict()
            #     for k, v in state_dict.items():
            #         state_dict[k] = v.cpu()
            #     p_test.outer_channel.send((curr_episode,
            #                                state_dict,
            #                                normalization.running_ms.mean,
            #                                normalization.running_ms.std))
        writer.close()


if __name__ == '__main__':
    train(get_args())
