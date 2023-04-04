# task specification


from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import RaisimGymRaiboRoughTerrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo_LSTM as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
import wandb
import datetime

# task specification

# os.environ["WANDB_API_KEY"] = '3bdcf6389f74ce8110d7914041ec50f6771bbee8'
# os.environ["WANDB_MODE"] = "dryrun"

# initialize wandb


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
print("env create pass")
# wandb name
name = cfg['name']
task_name = cfg['task_name'] + ' Transformer'


# Encoding
historyNum = cfg['environment']['dimension']['historyNum_']
pro_dim = cfg['environment']['dimension']['proprioceptiveDim_']
ext_dim = cfg['environment']['dimension']['exteroceptiveDim_']
inertial_dim = cfg['environment']['dimension']['inertialparamDim_']
dynamics_info_dim = cfg['environment']['dimension']['dynamicsInfoDim_']
dynamics_input_dim = cfg['environment']['dimension']['dynamicsInputDim_']
dynamics_predict_dim = cfg['environment']['dimension']['dynamicsPredictDim_']
ROA_ext_dim = ext_dim - inertial_dim

# shortcuts
act_dim = env.num_acts
Encoder_ob_dim = historyNum * (pro_dim + ext_dim + act_dim)

# LSTM
hidden_dim = cfg['LSTM']['hiddendim_']
batchNum = cfg['LSTM']['batchNum_']
layerNum = cfg['LSTM']['numLayer_']

# Transformer
n_head = cfg['Transformer']['n_head_']
layerNum = cfg['Transformer']['layerNum_']
dim_feedforward = cfg['Transformer']['dim_feedforward_']

# ROA Encoding
ROA_Encoder_ob_dim = historyNum * (pro_dim + ROA_ext_dim + act_dim)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

num_learning_epochs = cfg['PPO']['num_learning_epoch']
num_mini_batches = cfg['PPO']['num_mini_batches']

# PPO coeff
entropy_coeff_ = cfg['environment']['entropy_coeff']

obj_f_dynamics_input_dim = hidden_dim + act_dim + dynamics_input_dim

obj_f_dynamics = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['obj_f_dynamics']['net'],
                                                     nn.LeakyReLU,
                                                     obj_f_dynamics_input_dim,
                                                     dynamics_predict_dim),
                                      device=device)

latent_f_dynamics_input_dim = hidden_dim + act_dim

latent_f_dynamics = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['obj_f_dynamics']['net'],
                                                     nn.LeakyReLU,
                                                     latent_f_dynamics_input_dim,
                                                     hidden_dim),
                                      device=device)


Estimator = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'],
                                                nn.LeakyReLU,
                                                hidden_dim,
                                                inertial_dim), device=device)

Encoder_ROA = ppo_module.Encoder(architecture=ppo_module.Transformer(input_dim=int(ROA_Encoder_ob_dim/historyNum),
                                                          hidden_dim=hidden_dim,
                                                          ext_dim=ROA_ext_dim,
                                                          pro_dim=pro_dim,
                                                          act_dim=act_dim,
                                                          dyn_info_dim=dynamics_info_dim,
                                                          dyn_predict_dim=dynamics_predict_dim,
                                                          hist_num=historyNum,
                                                          device=device,
                                                          batch_num=batchNum,
                                                          num_env=env.num_envs,
                                                          d_model=hidden_dim,
                                                          num_minibatch = num_mini_batches,
                                                          dim_feedforward = dim_feedforward,
                                                          layerNum = layerNum,
                                                          nhead = n_head,
                                                          max_len=historyNum), device=device)

Encoder = ppo_module.Encoder(architecture=ppo_module.Transformer(input_dim=int(Encoder_ob_dim/historyNum),
                          hidden_dim=hidden_dim,
                          ext_dim=ext_dim,
                          pro_dim=pro_dim,
                          act_dim=act_dim,
                          dyn_info_dim=dynamics_info_dim,
                          dyn_predict_dim=dynamics_predict_dim,
                          hist_num=historyNum,
                          batch_num=batchNum,
                          device=device,
                          num_env=env.num_envs,
                          d_model=hidden_dim,
                          num_minibatch = num_mini_batches,
                          dim_feedforward = dim_feedforward,
                          layerNum = layerNum,
                          nhead = n_head,
                          max_len=historyNum), device=device)

pytorch_total_params = sum(p.numel() for p in Encoder.architecture.parameters())

print(pytorch_total_params)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim, act_dim, actor=True),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['encoding']['value_net'], nn.LeakyReLU, hidden_dim, 1, actor=False),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/RaiboController.hpp"])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              encoder=Encoder,
              obj_f_dynamics=obj_f_dynamics,
              latent_f_dynamics=latent_f_dynamics,
              num_envs=cfg['environment']['num_envs'],
              obs_shape=[env.num_obs],
              num_transitions_per_env=n_steps,
              num_learning_epochs=num_learning_epochs,
              gamma=0.995,
              lam=0.95,
              num_mini_batches=num_mini_batches,
              learning_rate=5e-5,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              encoder_ROA=Encoder_ROA,
              estimator=Estimator,
              desired_kl=0.006,
              num_history_batch=historyNum,
              inertial_dim=inertial_dim,
              entropy_coef=entropy_coeff_
              )

iteration_number = 0


wandb.init(group="jsh",project=task_name,name=name)

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
            'Encoder_state_dict' : Encoder.architecture.state_dict(),
            'Encoder_ROA_state_dict' : Encoder_ROA.architecture.state_dict(),
            'Inertial_estimator': Estimator.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
            'obj_f_dynamics_state_dict': obj_f_dynamics.architecture.state_dict(),
            'latent_f_dynamics_state_dict': latent_f_dynamics.architecture.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

        # env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):
            with torch.no_grad():
                obs = env.observe(False)

                # latent = Encoder.evaluate(torch.from_numpy(obs).to(device))
                latent = Encoder.evaluate(ppo.filter_for_encode_from_obs(obs).to(device))
                filtered_obs = ppo.filter_for_encode_from_obs(obs)
                # print(latent)
                # print(latent.shape)
                actions, actions_log_prob = actor.sample(latent)
                reward, dones = env.step_visualize(actions)
                data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
        # env.stop_video_recording()
        # env.turn_off_visualization()
        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    data_log = {}
    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)

            # latent = Encoder.evaluate(torch.from_numpy(obs).to(device))
            latent = Encoder.evaluate(ppo.filter_for_encode_from_obs(obs).to(device))

            latent = latent.detach().cpu().numpy()

            action = ppo.act(latent)

            reward, dones = env.step(action)

            ppo.step(value_obs=latent, obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
            # data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

    # data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
    # take st step to get value obs
    obs2 = env.observe(update < 10000)
    with torch.no_grad():
        # latent = Encoder.evaluate(torch.from_numpy(obs2).to(device))
        latent = Encoder.evaluate(ppo.filter_for_encode_from_obs(obs2).to(device))
        latent = latent.detach().cpu().numpy()

    ppo.update(actor_obs=latent, value_obs=latent, log_this_iteration=update % 10 == 0, update=update)


    ### For logging encoder (LSTM)
    # wandb.watch(Encoder.architecture)


    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*(0.6*math.exp(-0.0002*update) + 0.4)).to(device))
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
        data_log['PPO/loss_ROA'] = ppo.loss_ROA
        data_log['PPO/lambda_loss_ROA'] = ppo.lambda_loss_ROA
        data_log['PPO/estimator_loss'] = ppo.estimator_loss
        data_log['PPO/obj_f_dynamics_loss'] = ppo.obj_f_dynamics_loss
        data_log['PPO/entropy'] = ppo.entropy_mean
        data_log['PPO/latent_f_dynamics_loss'] = ppo.latent_f_dyn_loss
        data_log['PPO/obs_decoder_loss'] = ppo.decoder_loss

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
