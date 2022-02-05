import time

import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_raibo_rough_terrain
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import torch
import argparse
import pygame
import collections
import matplotlib.pyplot as plt


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
cfg['environment']['curriculum']['initial_factor'] = .5

env = VecEnv(rsg_raibo_rough_terrain.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# env.set_command(0)  # ensures that the initial command is zero

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

command = np.zeros(2, dtype=np.float32)
gc = np.zeros(19, dtype=np.float32)
gv = np.zeros(18, dtype=np.float32)

# plotting
gcHistory = []
gvHistory = []
buffer_size = 200

for i in range(3):
    gcHistory.append(collections.deque(np.zeros(buffer_size)))
    gvHistory.append(collections.deque(np.zeros(buffer_size)))

plt.ion()
fig = plt.figure(figsize=(12, 6), facecolor='#DEDEDE')
ax = plt.subplot(121)
ax1 = plt.subplot(122)
ax.set_facecolor('#DEDEDE')
ax1.set_facecolor('#DEDEDE')

ax.cla()
ax1.cla()
ax.set_ylim(-25, 25)
ax1.set_ylim(-25, 25)

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print(joystick.get_name())

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    for step in range(total_steps):
        for event in pygame.event.get():  # User did something.

            if event.type == pygame.JOYBUTTONDOWN:  # If user clicked close.
                if event.button == 0:
                    env.set_command(0)
                    print("set new goal")
                elif event.button == 1:
                    env.reset()
                    print("env reset")
                elif event.button == 2:
                    env.curriculum_callback()
                    print("change env")

        if len(joysticks) > 0:
            if abs(joysticks[0].get_axis(0)) > 0.05:
                command.flat[0] = command.flat[0] + joysticks[0].get_axis(0)*0.02

            if abs(joysticks[0].get_axis(1)) > 0.05:
                command.flat[1] = command.flat[1] + joysticks[0].get_axis(1)*-0.02

        env.move_controller_cursor(0, command)
        obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        env.step(action_ll.cpu().detach().numpy())

        # plotting
        env.get_state(gc, gv)

        for i in range(3):
            gcHistory[i].popleft()
            gcHistory[i].append(gc[7+i])
            gvHistory[i].popleft()
            gvHistory[i].append(gv[6+i])

            # plot gc
            ax.plot(gcHistory[i])

            # plot memory
            ax1.plot(gvHistory[i])

        plt.pause(cfg['environment']['control_dt'])

    plt.pause(100000)
    env.turn_off_visualization()
    env.reset()
