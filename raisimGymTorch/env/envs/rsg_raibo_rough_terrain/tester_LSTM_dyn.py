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

device = torch.device('cpu')
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
print(1)
is_rollout = cfg['environment']['Rollout']
#
if (is_rollout):
    print(1)
    env = VecEnv(rsg_raibo_rough_terrain.RaisimGymRaiboRoughTerrain_ROLLOUT(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
else:
    env = VecEnv(rsg_raibo_rough_terrain.RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
print(2)
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

# Encoding
historyNum = cfg['environment']['dimension']['historyNum_']
actionhistoryNum = cfg['environment']['dimension']['actionhistoryNum_']
pro_dim = cfg['environment']['dimension']['proprioceptiveDim_']
ext_dim = cfg['environment']['dimension']['exteroceptiveDim_']
inertial_dim = cfg['environment']['dimension']['inertialparamDim_']
dynamics_dim = cfg['environment']['dimension']['dynamicsDim_']
ROA_ext_dim = cfg['environment']['ROA_dimension']['exteroceptiveDim_']

# shortcuts
act_dim = env.num_acts
Encoder_ob_dim = historyNum * (pro_dim + ext_dim)

# LSTM
hidden_dim = cfg['LSTM']['hiddendim_']
batchNum = cfg['LSTM']['batchNum_']
is_decouple = cfg['LSTM']['is_decouple_']

# ROA Encoding
ROA_Encoder_ob_dim = historyNum * (pro_dim + ROA_ext_dim)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

# PPO coeff
entropy_coeff_ = cfg['environment']['entropy_coeff']


def get_obs_ROA(encoder, obs_batch):
    obs_ROA_batch = []


    for i in range(historyNum):
        # Get proprioceptive part of observation
        obs_ROA_batch.append(obs_batch[...,
                             (encoder.architecture.block_dim)*i:
                             (encoder.architecture.block_dim)*i
                             + encoder.architecture.pro_dim])

        # Get Exteroceptive part of observation except inertial parameter
        obs_ROA_batch.append(obs_batch[...,
                             (encoder.architecture.block_dim)*i
                             + encoder.architecture.pro_dim:
                             (encoder.architecture.block_dim)*i
                             + encoder.architecture.pro_dim
                             + encoder.architecture.ext_dim - inertial_dim])


    # estimator_true_data = (obs_batch[...,
    #                        (encoder.architecture.pro_dim +
    #                         encoder.architecture.ext_dim +
    #                         encoder.architecture.act_dim)*(historyNum-1)
    #                        + encoder.architecture.pro_dim +15:
    #                        (encoder.architecture.pro_dim +
    #                         encoder.architecture.ext_dim +
    #                         encoder.architecture.act_dim)*(historyNum-1)
    #                        + encoder.architecture.pro_dim +15+13
    #                        ])

    obs_ROA_batch = np.concatenate(obs_ROA_batch, axis=-1)
    # estimator_true_data = estimator_true_data.reshape(-1, 13)

    return obs_ROA_batch


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 3
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)

    obs_f_dynamics_input_dim = pro_dim + ROA_ext_dim + act_dim

    obs_f_dynamics = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['obs_f_dynamics']['net'],
                                                         nn.LeakyReLU,
                                                         obs_f_dynamics_input_dim,
                                                         pro_dim + ROA_ext_dim),
                                          device=device)

    obj_f_dynamics_input_dim = hidden_dim + act_dim

    obj_f_dynamics = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['obj_f_dynamics']['net'],
                                                         nn.LeakyReLU,
                                                         obj_f_dynamics_input_dim,
                                                         dynamics_dim),
                                          device=device)

    Estimator = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'],
                                                    nn.LeakyReLU,
                                                    int(Encoder_ob_dim/historyNum),
                                                    int((Encoder_ob_dim-ROA_Encoder_ob_dim)/historyNum)), device=device)

    Encoder_ROA = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ROA_Encoder_ob_dim/historyNum),
                                                                  hidden_dim=hidden_dim,
                                                                  ext_dim=ROA_ext_dim,
                                                                  pro_dim=pro_dim,
                                                                  act_dim=act_dim,
                                                                  dyn_dim=dynamics_dim,
                                                                  hist_num=historyNum,
                                                                  device=device,
                                                                  batch_num=batchNum,
                                                                  num_env=env.num_envs,
                                                                  is_decouple=is_decouple), device=device)

    Encoder = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(Encoder_ob_dim/historyNum),
                                                              hidden_dim=hidden_dim,
                                                              ext_dim=ext_dim,
                                                              pro_dim=pro_dim,
                                                              act_dim=act_dim,
                                                              dyn_dim=dynamics_dim,
                                                              hist_num=historyNum,
                                                              batch_num=batchNum,
                                                              device=device,
                                                              num_env=env.num_envs,
                                                              is_decouple=is_decouple), device=device)

    actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                               env.num_envs,
                                                                               1.0,
                                                                               NormalSampler(act_dim),
                                                                               cfg['seed']),
                             device)

    Encoder_Rollout = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(Encoder_ob_dim/historyNum),
                                                              hidden_dim=hidden_dim,
                                                              ext_dim=ext_dim,
                                                              pro_dim=pro_dim,
                                                              act_dim=act_dim,
                                                              dyn_dim=dynamics_dim,
                                                              hist_num=historyNum,
                                                              batch_num=batchNum,
                                                              device=device,
                                                              num_env=n_samples,
                                                              is_decouple=is_decouple), device=device)

    actor_Rollout = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                               n_samples,
                                                                               1.0,
                                                                               NormalSampler(act_dim),
                                                                               cfg['seed']),
                             device)

    actor.architecture.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(torch.load(weight_path)['actor_distribution_state_dict'])
    Encoder.architecture.load_state_dict(torch.load(weight_path)['Encoder_state_dict'])
    Encoder_ROA.architecture.load_state_dict(torch.load(weight_path)['Encoder_ROA_state_dict'])
    Estimator.architecture.load_state_dict(torch.load(weight_path)['Inertial_estimator'])
    obj_f_dynamics.architecture.load_state_dict(torch.load(weight_path)['obj_f_dynamics_state_dict'])
    obs_f_dynamics.architecture.load_state_dict(torch.load(weight_path)['obs_f_dynamics_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    traj_sampler = mppi.MPPI(dynamics=obj_f_dynamics,
                             encoder=Encoder_Rollout,
                             actor=actor_Rollout,
                             environment=env,
                             n_samples=n_samples,
                             horizon=n_horizon,
                             gamma=gamma,
                             device=device,
                             use_dynamics=use_dynamics)

    for i in range (int(int(iteration_number) / 100)):
        env.curriculum_callback()

    success_batch = []

    for i in range (100):
        env.curriculum_callback()
        env.reset()
        Encoder.architecture.reset()
        Encoder_ROA.architecture.reset()
        if(is_rollout):
            target_pos = env.get_target_pos()
        for step in range(total_steps):
            with torch.no_grad():
                if(is_rollout == False):
                    obs = env.observe(False)
                    obs_ROA = get_obs_ROA(Encoder, obs)
                    latent_ROA = Encoder_ROA.evaluate(torch.from_numpy(obs_ROA).to(device))
                    action_ll, actions_log_prob = actor.sample(latent_ROA)
                    success = torch.Tensor(env.get_success_state()).unsqueeze(-1)
                    env.step_visualize_success(action_ll, success)
                    print(action_ll)

                if (is_rollout == True):
                    obs = env.observe(False)
                    obs_ROA = get_obs_ROA(Encoder, obs)
                    latent_ROA = Encoder_ROA.evaluate(torch.from_numpy(obs_ROA).to(device))
                    cur_pos = env.get_obj_pos()
                    tic = time.time()
                    cur_observation = env.observe_Rollout(False)

                    action_rollout, predict_states = traj_sampler.compute_rollout(goal_state=target_pos, cur_state=cur_pos, cur_observation=cur_observation)
                    toc = time.time()
                    print("time consuming : ", toc - tic)
                    env.predict_obj_update(predict_states.numpy())
                    env.step_visualize(action_rollout.numpy())
                    env.synchronize()

                    # gc = np.array([1]*19, dtype=np.float32)
                    # gv = np.array([1]*18, dtype=np.float32)
                    # env.get_state(gc, gv)
                    # print(gc)
                    # print(gv)

                    # gc_batch = np.ones((n_samples,19), dtype=np.float32)
                    # gv_batch = np.ones((n_samples,18), dtype=np.float32)
                    # env.get_state_rollout(gc_batch, gv_batch)
                    # print(gc_batch)
                    # print(gv_batch)

                    """
                    Encoder 넣을 때 적절한 시기에 clone해서 넣어줘야 할 듯 ? LSTM이라
                    """



                    # gc_batch = np.ones((n_samples,19), dtype=np.float32)
                    # gv_batch = np.ones((n_samples,18), dtype=np.float32)
                    # env.get_state_rollout(gc_batch, gv_batch)
                    # print(gc_batch)
                    # print(gv_batch)

                # success = torch.Tensor(env.get_success_state()).unsqueeze(-1)
                # TODO add MPPI here
                '''
                action_traj = MPPI(state, e, o)
                '''
        # success = torch.Tensor(env.get_success_state()).unsqueeze(-1)
        # success_sum = torch.sum(success, dim=0) / success.shape(0)
        # print(success_sum)
        # success_batch.append(success)
    success_batch = torch.cat(success_batch, dim=0)
    print(torch.sum(success_batch, dim=0))



    env.turn_off_visualization()
    env.reset()


