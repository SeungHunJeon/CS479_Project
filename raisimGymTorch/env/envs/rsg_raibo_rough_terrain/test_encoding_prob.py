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

pro_latent_dim = cfg['encoder']['proprioceptivelatentDim_']
ext_latent_dim = cfg['encoder']['exteroceptivelatentDim_']
act_latent_dim = cfg['encoder']['actionlatentDim_']

obs_pro_dim = pro_dim*(historyNum)
obs_ext_dim = ext_dim*(historyNum)
obs_act_dim = act_dim*(actionhistoryNum)

@staticmethod
def latent_concat(obs):
    with torch.no_grad():
        obs_proprioceptive = obs[:, :pro_dim*(historyNum)]
        obs_exteroceptive = obs[:, pro_dim*(historyNum) : (pro_dim+ext_dim)*(historyNum)]
        obs_action = obs[:, (pro_dim+ext_dim)*(historyNum):]

        pro_latent, pro_mu, pro_logvar = pro_encoder.evaluate(torch.from_numpy(obs_proprioceptive).to(device))
        ext_latent, ext_mu, ext_logvar = ext_encoder.evaluate(torch.from_numpy(obs_exteroceptive).to(device))
        act_latent, act_mu, act_logvar = act_encoder.evaluate(torch.from_numpy(obs_action).to(device))

        obs_concat = torch.cat((pro_latent,ext_latent,act_latent), 1)
    return obs_concat

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                               env.num_envs,
                                                                               1.0,
                                                                               NormalSampler(act_dim),
                                                                               cfg['seed']), device)

    pro_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['pro_encoder_net'],
                                                         torch.nn.LeakyReLU,
                                                         obs_pro_dim,
                                                         pro_latent_dim
                                                         ),
                                     device)

    ext_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['ext_encoder_net'],
                                                         torch.nn.LeakyReLU,
                                                         obs_ext_dim,
                                                         ext_latent_dim),
                                     device)

    act_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['act_encoder_net'],
                                                         torch.nn.LeakyReLU,
                                                         obs_act_dim,
                                                         act_latent_dim),
                                     device)


    actor.architecture.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(torch.load(weight_path)['actor_distribution_state_dict'])
    pro_encoder.architecture.load_state_dict(torch.load(weight_path)['pro_encoder_state_dict'])
    ext_encoder.architecture.load_state_dict(torch.load(weight_path)['ext_encoder_state_dict'])
    act_encoder.architecture.load_state_dict(torch.load(weight_path)['act_encoder_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    for i in range (200):
        env.reset()
        time.sleep(1)
        for step in range(total_steps):
            with torch.no_grad():
                obs = env.observe(False)

                obs_concat = latent_concat(obs)

                action_ll, actions_log_prob = actor.sample(obs_concat)

                print(action_ll)
                env.step_visualize(action_ll)

    env.turn_off_visualization()
    env.reset()


