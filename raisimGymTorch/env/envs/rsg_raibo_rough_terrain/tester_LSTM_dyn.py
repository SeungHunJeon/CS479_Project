import time

import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_raibo_rough_terrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.bin import rsg_raibo_rough_terrain
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import torch
import argparse
import collections
import torch.nn as nn
import raisimGymTorch.algo.MPPI.mppi as mppi
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')
device = torch.device('cuda:0')
# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# MPPI
n_samples = cfg['MPPI']['nSamples_']
n_horizon = cfg['MPPI']['nHorizon_']
gamma = cfg['MPPI']['gamma_']
use_dynamics = cfg['MPPI']['use_dynamics_']

# create environment from the configuration file
# cfg['environment']['num_envs'] = 1 + n_samples
cfg['environment']['num_envs'] = 1

cfg['environment']['render'] = True
cfg['environment']['curriculum']['initial_factor'] = 1.
is_rollout = cfg['environment']['Rollout']
# create environment from the configuration file
# cfg['environment']['num_envs'] = 1 + n_samples
if(is_rollout):
    num_env = cfg['environment']['num_envs']
else:
    num_env = 100
    cfg['environment']['num_envs'] = 100


#
env = VecEnv(rsg_raibo_rough_terrain.RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

print('env create success')
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

# Encoding
historyNum = cfg['environment']['dimension']['historyNum_']
pro_dim = cfg['environment']['dimension']['proprioceptiveDim_']
ext_dim = 0
inertial_dim = 0
dynamics_info_dim = 0
dynamics_predict_dim = 0
ROA_ext_dim = ext_dim - inertial_dim

# shortcuts
act_dim = env.num_acts
# Encoder_ob_dim = historyNum * inertial_dim
Encoder_ob_dim = historyNum * (pro_dim + act_dim)


# LSTM
hidden_dim = cfg['LSTM']['hiddendim_']
batchNum = cfg['LSTM']['batchNum_']
layerNum = cfg['LSTM']['numLayer_']
is_decouple = cfg['LSTM']['isDecouple_']

# ROA Encoding
ROA_Encoder_ob_dim = historyNum * (pro_dim + act_dim)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
num_learning_epochs = 16
num_mini_batches = 1
Domain_Randomization = cfg['environment']['Domain_Randomization']

# PPO coeff
entropy_coeff_ = cfg['environment']['entropy_coeff']

def actor_input_concat(encoder, latent, obs):
    actor_input = []

    actor_input.apend(obs[...,
                      0
                      :
                      encoder.architecture.block_dim])
    actor_input.append(obs[...,
    (encoder.architecture.block_dim)*(historyNum-1)
                       :
    (encoder.architecture.block_dim)*(historyNum-1)
                       + encoder.architecture.pro_dim])

    actor_input = np.concatenate(actor_input, axis=-1)

    return torch.cat((latent, torch.Tensor(actor_input)), dim=-1)

def obs_post_process(encoder, obs_batch):
    obs = []


    for i in range(historyNum):
        # Get proprioceptive part of observation
        obs.append(obs_batch[...,
                             (encoder.architecture.block_dim)*i:
                             (encoder.architecture.block_dim)*i
                             + encoder.architecture.pro_dim + encoder.architecture.act_dim])

    obs = np.concatenate(obs, axis=-1)

    return obs

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 3
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)

    Estimator = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'],
                                                    nn.LeakyReLU,
                                                    hidden_dim + pro_dim * 2 + act_dim,
                                                    24), device=device)

    Estimator_cov = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'],
                                                        nn.LeakyReLU,
                                                        hidden_dim + pro_dim * 2 + act_dim,
                                                        24), device=device)
#

    # Encoder_ROA = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ROA_Encoder_ob_dim/historyNum),
    #                                                               hidden_dim=hidden_dim,
    #                                                               ext_dim=ROA_ext_dim,
    #                                                               pro_dim=pro_dim,
    #                                                               act_dim=act_dim,
    #                                                               dyn_info_dim=dynamics_info_dim,
    #                                                               dyn_predict_dim=dynamics_predict_dim,
    #                                                               hist_num=historyNum,
    #                                                               device=device,
    #                                                               batch_num=batchNum,
    #                                                               layer_num=layerNum,
    #                                                               num_minibatch = num_mini_batches,
    #                                                               num_env=num_env), device=device)

    Encoder = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(Encoder_ob_dim/historyNum),
                                                              hidden_dim=hidden_dim,
                                                              ext_dim=ext_dim,
                                                              pro_dim=pro_dim,
                                                              act_dim=act_dim,
                                                              dyn_info_dim=dynamics_info_dim,
                                                              dyn_predict_dim=dynamics_predict_dim,
                                                              hist_num=historyNum,
                                                              batch_num=batchNum,
                                                              layer_num=layerNum,
                                                              device=device,
                                                              num_minibatch = num_mini_batches,
                                                              num_env=num_env), device=device)


    # actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
    #                          ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
    #                                                                            num_env,
    #                                                                            1.0,
    #                                                                            NormalSampler(act_dim),
    #                                                                            cfg['seed']),
    #                          device)

    # Encoder_Rollout = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ROA_Encoder_ob_dim/historyNum),
    #                                                                   hidden_dim=hidden_dim,
    #                                                                   ext_dim=ROA_ext_dim,
    #                                                                   pro_dim=pro_dim,
    #                                                                   act_dim=act_dim,
    #                                                                   dyn_info_dim=dynamics_info_dim,
    #                                                                   dyn_predict_dim=dynamics_predict_dim,
    #                                                                   hist_num=historyNum,
    #                                                                   device=device,
    #                                                                   batch_num=batchNum,
    #                                                                   layer_num=layerNum,
    #                                                                   num_minibatch = num_mini_batches,
    #                                                                   num_env=num_env), device=device)
    #
    # actor_Rollout = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
    #                          ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
    #                                                                            n_samples,
    #                                                                            1.0,
    #                                                                            NormalSampler(act_dim),
    #                                                                            cfg['seed']),
    #                          device)

    Encoder.architecture.load_state_dict(torch.load(weight_path)['Encoder_state_dict'])
    # Encoder_ROA.architecture.load_state_dict(torch.load(weight_path)['Encoder_ROA_state_dict'])
    Estimator.architecture.load_state_dict(torch.load(weight_path)['Estimator_state_dict'])
    Estimator_cov.architecture.load_state_dict(torch.load(weight_path)['Estimator_cov_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    # for i in range (int(int(iteration_number) / 100)):
    #     env.curriculum_callback()

    # for i in range(100):
    #     env.curriculum_callback()
    #
    # traj_sampler = mppi.MPPI(latent_f_dynamics=latent_f_dynamics,
    #                          obj_f_dynamics=obj_f_dynamics,
    #                          encoder=Encoder,
    #                          encoder_ROA=Encoder_Rollout,
    #                          actor=actor_Rollout,
    #                          environment=env,
    #                          n_samples=n_samples,
    #                          horizon=n_horizon,
    #                          gamma=gamma,
    #                          device=device,
    #                          use_dynamics=use_dynamics,
    #                          inertial_dim=inertial_dim)



    success_batch = []
    success = None




    for i in range (100):

        env.reset()

        # env_value = env.get_envrionmental_value()
        # print(env_value.shape)
        Encoder.architecture.reset()
        # Encoder_ROA.architecture.reset()

        # For action plotting
        idx = 0
        x = []
        y = []

        x = np.linspace(0, total_steps, total_steps*5)
        y = np.zeros(total_steps*5)
        y2 = np.zeros(total_steps*5)
        y3 = np.zeros(total_steps*5)
        plt.ion()
        figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
        line1, = ax[0].plot(x,y)
        line2, = ax[1].plot(x,y2)
        line3, = ax[2].plot(x,y3)
        ax[0].set_xlim(0, total_steps)
        ax[1].set_xlim(0, total_steps)
        ax[2].set_xlim(0, total_steps)
        ax[0].set_ylim(-5, 5)
        ax[1].set_ylim(-5, 5)
        ax[2].set_ylim(-5, 5)


        if(is_rollout):
            target_pos = env.get_target_pos()

        actions = np.zeros((env.num_envs, 3), dtype=np.float32)
        for step in range(total_steps):
            with torch.no_grad():
                obs = env.observe(False)
                obs_processed = obs_post_process(Encoder, obs)
                latent = Encoder.evaluate(torch.from_numpy(obs_processed).to(device))
                actor_input = actor_input_concat(Encoder, latent, obs)

                estimated_anchors = Estimator.predict(actor_input)

                # latent_ROA = Encoder_ROA.evaluate(torch.from_numpy(obs_ROA).to(device))
                # action_ll = actor.architecture(latent_ROA, actor=True).cpu().numpy()
                # action_ll, actions_log_prob = actor.sample(latent_ROA)

                # For action plotting
                # y[idx] = action_ll[0][0]
                # y2[idx] = action_ll[0][1]
                # y3[idx] = action_ll[0][2]
                # idx+=1
                # line1.set_xdata(x)
                # line1.set_ydata(y)
                # line2.set_xdata(x)
                # line2.set_ydata(y2)
                # line3.set_xdata(x)
                # line3.set_ydata(y3)
                # figure.canvas.draw()
                # figure.canvas.flush_events()

                # success = torch.Tensor(env.get_success_state()).unsqueeze(-1)
                env.step_visualize(actions)

                '''
        # For action plotting
        # plt.close(figure)
        # print( "Trial : {} ".format(i))
        # print(success.sum().item())
        # success_npy = success.numpy()
        # env_value = np.concatenate([env_value, success_npy], axis=-1)
        # print(env_value.shape)

        # env_value
        # env_value_data_frame = pd.DataFrame(env_value)
        # env_value_data_frame.to_csv("env_value.csv", index=False)

        # if int(success) == 0:
        #     print("failed")
        # else:
        #     print("success")
        # success = torch.Tensor(env.get_success_state()).unsqueeze(-1)
        # success_sum = torch.sum(success, dim=0) / success.shape(0)
        # print(success_sum)
        # success_batch.append(success)
    # success_batch = torch.cat(success_batch, dim=0)
    # print(torch.sum(success_batch, dim=0))
    # 
    # 
    # 
    # env.turn_off_visualization()
    # env.reset()
    '''


