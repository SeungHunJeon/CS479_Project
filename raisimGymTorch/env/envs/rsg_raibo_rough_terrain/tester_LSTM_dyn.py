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

from estimator_helper import state_estimator, get_img_process
from nerf_helper import load_intrinsic_params
from nerf.utils import *
from nerf.provider import NeRFDataset

# dddd

matplotlib.use('tkagg')
device = torch.device('cuda:0')
# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')

parser.add_argument('path', type=str)
parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--workspace', type=str, default='ngp')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=30000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

### network backbone options
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
# (the default value is for the fox dataset)
parser.add_argument('--bound', type=float, default=4.0, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
parser.add_argument('--dt_gamma', type=float, default=0.02, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=1920, help="GUI width")
parser.add_argument('--H', type=int, default=1080, help="GUI height")
parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")


### experimental
parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

opt = parser.parse_args()
opt.path = os.path.dirname(os.path.realpath(__file__)) + "/" + opt.path
opt.workspace = os.path.dirname(os.path.realpath(__file__)) + "/" + opt.workspace
if opt.O:
    opt.fp16 = True
    opt.cuda_ray = False
    opt.preload = False

if opt.ff:
    opt.fp16 = False
    assert opt.bg_radius <= 0, "background model is not implemented for --ff"
    from nerf.network_ff import NeRFNetwork
elif opt.tcnn:
    opt.fp16 = False
    assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    from nerf.network_tcnn import NeRFNetwork
else:
    from nerf.network import NeRFNetwork



# torch.set_default_tensor_type(torch.cuda.FloatTensor)
seed_everything(opt.seed)

nerf_model = NeRFNetwork(
    encoding="hashgrid",
    bound=opt.bound,
    cuda_ray=opt.cuda_ray,
    density_scale=1,
    min_near=opt.min_near,
    density_thresh=opt.density_thresh,
    bg_radius=opt.bg_radius,
)

nerf_model.eval()
metrics = [PSNRMeter(),]
criterion = torch.nn.MSELoss(reduction='none')



# trainer = Trainer('ngp', opt, nerf_model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)


intrinsics, img_W, img_H = load_intrinsic_params(opt)

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

cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
cfg['environment']['curriculum']['initial_factor'] = 1.
is_rollout = cfg['environment']['Rollout']
num_env = 1

#
env = VecEnv(rsg_raibo_rough_terrain.RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])


nerf_model = nerf_model.to(device)

ckpt_path = os.path.join(opt.workspace, 'checkpoints')
checkpoint = None
if checkpoint is None:
    checkpoint_list = sorted(glob.glob(f'{ckpt_path}/ngp_ep*.pth'))
if checkpoint_list:
    checkpoint = checkpoint_list[-1]
    print(f"[INFO] Latest checkpoint is {checkpoint}")
else:
    print("[WARN] No checkpoint found, model randomly initialized.")

checkpoint_dict = torch.load(checkpoint, map_location='cpu')
if 'model' not in checkpoint_dict:
    nerf_model.load_state_dict(checkpoint_dict)
    print("[INFO] loaded model.")

else:
    missing_keys, unexpected_keys = nerf_model.load_state_dict(checkpoint_dict['model'], strict=False)
    print("[INFO] loaded model.")
    if len(missing_keys) > 0:
        print(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"[WARN] unexpected keys: {unexpected_keys}")

print('env create success')
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = opt.weight
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

    actor_input.append(obs[...,
                      0
                      :
                      encoder.architecture.block_dim])
    actor_input.append(obs[...,
    (encoder.architecture.block_dim)*(historyNum-1)
                       :
    (encoder.architecture.block_dim)*(historyNum-1)
                       + encoder.architecture.pro_dim])

    actor_input = np.concatenate(actor_input, axis=-1)

    return torch.cat((latent, torch.Tensor(actor_input).to(device)), dim=-1)

def estimator_pre_process(anchor_points):
    return anchor_points[..., -24:] - anchor_points[..., :24]

def obs_post_process(encoder, obs_batch):
    obs = []


    for i in range(historyNum):
        # Get proprioceptive part of observation
        obs.append(obs_batch[...,
                             (encoder.architecture.block_dim)*i:
                             (encoder.architecture.block_dim)*i
                             + encoder.architecture.pro_dim + encoder.architecture.act_dim])

    obs = np.concatenate(obs, axis=0)

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
                                                              num_env=num_env,
                                                              inertial_dim= 0
                                                            ), device=device)

    filter_cfg = {
        'dil_iter': 3,
        'batch_size': 1024,
        'kernel_size': 5,
        'lrate': 1e-3,
        'N_iter': 300,
        'render_viz': True,
        'show_rate': [20, 100]
    }



    render_fn = lambda rays_o, rays_d: nerf_model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
    get_rays_fn = lambda pose: get_rays(pose, intrinsics, img_H, img_W)

    filter = state_estimator(filter_cfg, get_rays_fn=get_rays_fn, render_fn=render_fn, device=device)

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

        plt.ion()
        fig, ax = plt.subplots()

        x_data = []
        y_data = []

        if(is_rollout):
            target_pos = env.get_target_pos()

        actions = np.zeros((env.num_envs, 3), dtype=np.float32)
        for step in range(total_steps):
            torch.cuda.empty_cache()
            with torch.no_grad():
                x = step
                obs = env.observe(False)
                obs_processed = obs_post_process(Encoder, obs)
                Encoder.architecture.reset()
                latent = Encoder.evaluate(torch.from_numpy(obs_processed).to(device))
                actor_input = actor_input_concat(Encoder, latent, obs)
                anchors = env.getAnchorHistory()

                target = estimator_pre_process(anchors)
                estimated_anchors = Estimator.predict(actor_input)
                estimation_cov = torch.exp_(2*Estimator_cov.predict(actor_input))

                mse = np.mean(np.sqrt((torch.Tensor(target).cpu().numpy()-estimated_anchors.detach().cpu().numpy())**2))
                print("estimation error: ", mse)
                print("estimation COV :  ",estimation_cov)
                cam_pos, cam_rot = env.get_camera_pose()
                # img = torch.from_numpy(env.get_color_image()[0])

                # img = get_img_process(env.get_color_image()[0], False).to(device)

                img = torch.load('sample_img.pt')

                # img = get_img_process(torch.ones((720,1080,3)),False).to(device)

                gt_pose = torch.eye(4)
                gt_pose[:3, :3] = torch.from_numpy(cam_rot)
                gt_pose[:3, 3] = torch.from_numpy(cam_pos)
                gt_pose = gt_pose.to(device)

                # dummy function for get camera_img from raisim_env
                # img = env.get_color_image()
                # processed_img = get_img_process(img, white_bg=True)
                # gt_anchor
                anchors_w = env.getAnchorHistory(robotFrame=False)
                prev_anchor_w = anchors_w[..., :24]
            current_anchor = filter.estimate_state_fusion(img, estimated_anchors, estimation_cov, prev_anchor_w, gt_pose)
            # prev_anchor_w = current_anchor
            cam_pose, cam_rot = env.get_camera_pose()



            # TODO : Here, replace estimated_anchors to current_anchor. However, the represented frame mismatches.
            env.step_evaluate(actions, estimated_anchors.detach().cpu().numpy())
            y = env.get_error(step >= 5, estimated_anchors.detach().cpu().numpy().transpose(1,0))/8

            x_data.append(x)
            y_data.append(y)

            ax.clear()
            plt.rc('font', size=40)
            ax.plot(x_data, y_data,linewidth =10)
            plt.xlabel('num of time step(0.2sec)')
            plt.ylabel('estimation error of SE(3) projected on 8 anchor point (meter)')
            plt.show()
            plt.pause(0.2)

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


