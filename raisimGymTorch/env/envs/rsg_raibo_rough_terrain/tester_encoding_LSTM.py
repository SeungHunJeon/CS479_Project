import time

import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_raibo_rough_terrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import torch
import argparse
import collections
import torch.nn as nn

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

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
cfg['environment']['curriculum']['initial_factor'] = 1.


env = VecEnv(rsg_raibo_rough_terrain.RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# env.set_command(0)  # ensures that the initial command is zero

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

obs_pro_dim = pro_dim
obs_ext_dim = ext_dim
obs_act_dim = act_dim

# LSTM
hidden_dim = cfg['LSTM']['hiddendim_']
batchNum = cfg['LSTM']['batchNum_']

# ROA Encoding
ROA_ext_dim = cfg['environment']['ROA_dimension']['exteroceptiveDim_']
ROA_ob_dim = historyNum * (pro_dim + act_dim + ROA_ext_dim)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

total_steps = n_steps * env.num_envs



def get_obs_ROA(encoder, obs_batch):
    obs_ROA_batch = []


    for i in range(historyNum):
        obs_ROA_batch.append(obs_batch[...,
                             (encoder.architecture.pro_dim + encoder.architecture.ext_dim + encoder.architecture.act_dim)*i
                             :(encoder.architecture.pro_dim + encoder.architecture.ext_dim + encoder.architecture.act_dim)*i
                              + encoder.architecture.pro_dim])
        obs_ROA_batch.append(obs_batch[...,
                             (encoder.architecture.pro_dim + encoder.architecture.ext_dim + encoder.architecture.act_dim)*i
                             + encoder.architecture.pro_dim:
                             (encoder.architecture.pro_dim +
                              encoder.architecture.ext_dim +
                              encoder.architecture.act_dim)*i
                             + encoder.architecture.pro_dim +15])


        obs_ROA_batch.append(obs_batch[...,
                             (encoder.architecture.pro_dim + encoder.architecture.ext_dim + encoder.architecture.act_dim)*i
                             + encoder.architecture.pro_dim+15+13:(encoder.architecture.pro_dim + encoder.architecture.ext_dim + encoder.architecture.act_dim)*i
                                                                       + encoder.architecture.pro_dim+15+13 +12])

    estimator_true_data = (obs_batch[...,
                           (encoder.architecture.pro_dim +
                            encoder.architecture.ext_dim +
                            encoder.architecture.act_dim)*(historyNum-1)
                           + encoder.architecture.pro_dim +15:
                           (encoder.architecture.pro_dim +
                            encoder.architecture.ext_dim +
                            encoder.architecture.act_dim)*(historyNum-1)
                           + encoder.architecture.pro_dim +15+13
                           ])

    obs_ROA_batch = np.concatenate(obs_ROA_batch, axis=-1)
    estimator_true_data = estimator_true_data.reshape(-1, 13)

    return obs_ROA_batch, estimator_true_data


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                               env.num_envs,
                                                                               1.0,
                                                                               NormalSampler(act_dim),
                                                                               cfg['seed']),
                             device)
    Encoder = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ob_dim/batchNum),
                                                              hidden_dim=hidden_dim,
                                                              ext_dim=ext_dim,
                                                              pro_dim=pro_dim,
                                                              act_dim=act_dim,
                                                              hist_num=historyNum,
                                                              batch_num=batchNum,
                                                              device=device,
                                                              num_env=env.num_envs), device=device)

    Estimator = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'], nn.LeakyReLU, int(ob_dim/historyNum),
                                                    int((ob_dim-ROA_ob_dim)/historyNum)), device=device)

    Encoder_ROA = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ROA_ob_dim/batchNum),
                                                                  hidden_dim=hidden_dim,
                                                                  ext_dim=ROA_ext_dim,
                                                                  pro_dim=pro_dim,
                                                                  act_dim=act_dim,
                                                                  hist_num=historyNum,
                                                                  device=device,
                                                                  batch_num=batchNum,
                                                                  num_env=env.num_envs), device=device)


    actor.architecture.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(torch.load(weight_path)['actor_distribution_state_dict'])
    Encoder.architecture.load_state_dict(torch.load(weight_path)['Encoder_state_dict'])
    Encoder_ROA.architecture.load_state_dict(torch.load(weight_path)['Encoder_ROA_state_dict'])
    Estimator.architecture.load_state_dict(torch.load(weight_path)['Inertial_estimator'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    for i in range (int(int(iteration_number) / 100)):
        env.curriculum_callback()

    for i in range (200):
        env.curriculum_callback()
        env.reset()
        Encoder.architecture.reset()
        Encoder_ROA.architecture.reset()
        for step in range(total_steps):
            with torch.no_grad():
                obs = env.observe(False)

                # latent = Encoder.evaluate(torch.from_numpy(obs).to(device))
                obs_ROA, _ = get_obs_ROA(Encoder, obs)

                latent_ROA = Encoder_ROA.evaluate(torch.from_numpy(obs_ROA).to(device))

                action_ll, actions_log_prob = actor.sample(latent_ROA)

                # print(action_ll)
                env.step_visualize(action_ll)

    env.turn_off_visualization()
    env.reset()


