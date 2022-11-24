# task specification
task_name = "Random_Object_encod_prob"

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import RaisimGymRaiboRoughTerrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo_encoding as PPO_Encod
import torch.nn as nn
import numpy as np
import torch
import argparse
import wandb
import datetime

# task specification

# initialize wandb
wandb.init(group="jsh",project=task_name)

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

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

total_steps = n_steps * env.num_envs

pro_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['pro_encoder_net'],
                                                nn.LeakyReLU,
                                                obs_pro_dim,
                                                pro_latent_dim
                                                ),
                                 device)

ext_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['ext_encoder_net'],
                                                nn.LeakyReLU,
                                                obs_ext_dim,
                                                ext_latent_dim),
                                 device)

act_encoder = ppo_module.Encoder(ppo_module.MLP_Prob(cfg['architecture']['encoding']['act_encoder_net'],
                                                nn.LeakyReLU,
                                                obs_act_dim,
                                                act_latent_dim),
                                 device)

encoders = [pro_encoder, ext_encoder, act_encoder]

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, pro_latent_dim+ext_latent_dim+act_latent_dim, act_dim, actor=True),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['encoding']['value_net'], nn.LeakyReLU, pro_latent_dim+ext_latent_dim+act_latent_dim, 1, actor=False),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/RaiboController.hpp"])

ppo = PPO_Encod.PPO(actor=actor,
              critic=critic,
              encoder=encoders,
              encoder_deterministic=False,
              num_envs=cfg['environment']['num_envs'],
              obs_shape=[env.num_obs],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.995,
              lam=0.95,
              num_mini_batches=4,
              learning_rate=5e-5,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              desired_kl=0.006,
              )

iteration_number = 0

@staticmethod
def latent_concat(obs):
    obs_proprioceptive = obs[:, :pro_dim*(historyNum)]
    obs_exteroceptive = obs[:, pro_dim*(historyNum) : (pro_dim+ext_dim)*(historyNum)]
    obs_action = obs[:, (pro_dim+ext_dim)*(historyNum):]

    pro_latent, pro_mu, pro_logvar = pro_encoder.evaluate(torch.from_numpy(obs_proprioceptive).to(device))
    ext_latent, ext_mu, ext_logvar = ext_encoder.evaluate(torch.from_numpy(obs_exteroceptive).to(device))
    act_latent, act_mu, act_logvar = act_encoder.evaluate(torch.from_numpy(obs_action).to(device))

    obs_concat = torch.cat((pro_latent,ext_latent,act_latent), 1)
    return obs_concat

if mode == 'retrain':
    iteration_number = load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(iteration_number, 1000000):
    torch.cuda.empty_cache()
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
#
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")

        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'pro_encoder_state_dict' : pro_encoder.architecture.state_dict(),
            'ext_encoder_state_dict' : ext_encoder.architecture.state_dict(),
            'act_encoder_state_dict' : act_encoder.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

        # env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):
            with torch.no_grad():
                obs = env.observe(False)

                obs_concat = latent_concat(obs)

                actions, actions_log_prob = actor.sample(obs_concat)
                reward, dones = env.step_visualize(actions)
                # data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        # data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))

        env.stop_video_recording()
        # env.turn_off_visualization()
        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    data_log = {}
    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)

            obs_concat = latent_concat(obs)
            obs_concat = obs_concat.detach().cpu().numpy()

            action = ppo.act(obs_concat)

            reward, dones = env.step(action)
            ppo.step(value_obs=obs_concat, obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
            data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

    data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
    # take st step to get value obs
    obs = env.observe(update < 10000)

    obs_concat = latent_concat(obs)
    obs_concat = obs_concat.detach().cpu().numpy()

    ppo.update(actor_obs=obs_concat, value_obs=obs_concat, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.distribution.enforce_minimum_std((torch.ones(2)*(0.6*math.exp(-0.0002*update) + 0.4)).to(device))
    # actor.distribution.enforce_minimum_std((torch.ones(1)*(0.06*math.exp(-0.0002*update) + 0.04)).to(device))
    actor.update()

    if update % 100 == 0:
        env.curriculum_callback()

    if update % 10 == 0:
        data_log['Training/average_reward'] = average_ll_performance
        data_log['Training/dones'] = average_dones
        data_log['Training/learning_rate'] = ppo.learning_rate
        data_log['PPO/value_function'] = ppo.mean_value_loss
        data_log['PPO/surrogate'] = ppo.mean_surrogate_loss
        data_log['PPO/mean_noise_std'] = ppo.mean_noise_std

        for id, data_name in enumerate(data_tags):
            data_log[data_name + '/mean'] = data_mean[id]
            data_log[data_name + '/std'] = data_std[id]

    end = time.time()

    wandb.log(data_log)

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
