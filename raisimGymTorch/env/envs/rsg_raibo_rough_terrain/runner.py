# task specification
task_name = "rsg_raibo_rough_terrain"

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import RaisimGymRaiboRoughTerrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
if mode == 'retrain':
    cfg['environment']['curriculum']['initial_factor'] = 1
env = VecEnv(RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/RaiboController.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.995,
              lam=0.95,
              num_mini_batches=4,
              learning_rate=2e-4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              desired_kl=0.006,
              )

iteration_number = 0

if mode == 'retrain':
    iteration_number = load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(iteration_number, 1000000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    # if update % cfg['environment']['eval_every_n'] == 0:
    #     print("Visualizing and evaluating the current policy")
    #     torch.save({
    #         'actor_architecture_state_dict': actor.architecture.state_dict(),
    #         'actor_distribution_state_dict': actor.distribution.state_dict(),
    #         'critic_architecture_state_dict': critic.architecture.state_dict(),
    #         'optimizer_state_dict': ppo.optimizer.state_dict(),
    #     }, saver.data_dir+"/full_"+str(update)+'.pt')
    #
    #     data_tags = env.get_step_data_tag()
    #     data_size = 0
    #     data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
    #     data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
    #     data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
    #     data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
    #
    #     for step in range(n_steps):
    #         with torch.no_grad():
    #             obs = env.observe(False)
    #             actions, actions_log_prob = actor.sample(torch.from_numpy(obs).to(device))
    #             reward, dones = env.step_visualize(actions)
    #             data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)
    #
    #     data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
    #
    #     for data_id in range(len(data_tags)):
    #         ppo.writer.add_scalar(data_tags[data_id]+'/mean', data_mean[data_id], global_step=update)
    #         ppo.writer.add_scalar(data_tags[data_id]+'/std', data_std[data_id], global_step=update)
    #         ppo.writer.add_scalar(data_tags[data_id]+'/min', data_min[data_id], global_step=update)
    #         ppo.writer.add_scalar(data_tags[data_id]+'/max', data_max[data_id], global_step=update)
    #
    #     env.reset()
    #     env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)
            action = ppo.act(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)

    # take st step to get value obs
    obs = env.observe(update < 10000)
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.distribution.enforce_minimum_std((torch.ones(12)*(0.6*math.exp(-0.0002*update) + 0.4)).to(device))
    actor.update()

    if update % 100 == 0:
        env.curriculum_callback()

    if update % 10 == 0:
        ppo.writer.add_scalar('Training/average_reward', average_ll_performance, global_step=update)
        ppo.writer.add_scalar('Training/dones', average_dones, global_step=update)
        ppo.writer.add_scalar('Training/learning_rate', ppo.learning_rate, global_step=update)

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("learning rate: ", '{:0.6f}'.format(ppo.learning_rate)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
