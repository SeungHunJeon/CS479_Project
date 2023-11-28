import os, sys
import numpy as np
import torch
import shutil
import pathlib
import subprocess
from tqdm import trange
import argparse

from raisimGymTorch.nav import state_estimator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, density_fn, render_fn, get_rays_fn):
    '''
    Main loop that iterates between planning and estimation.
    '''

    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']

    # Creates a workspace to hold all the trajectory data
    basefolder = "paths" / pathlib.Path(planner_cfg['exp_name'])
    if basefolder.exists():
        print(basefolder, "already exists!")
        # if input("Clear it before continuing? [y/N]:").lower() == "y":
        shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "init_poses").mkdir()
    (basefolder / "init_costs").mkdir()
    (basefolder / "replan_poses").mkdir()
    (basefolder / "replan_costs").mkdir()
    (basefolder / "estimator_data").mkdir()
    print("created", basefolder)

    # Initialize Planner
    traj = Planner(start_state, end_state, planner_cfg, density_fn)

    traj.basefolder = basefolder

    # Create a coarse trajectory to initialize the planner by using A*.
    traj.a_star_init()

    # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
    # that minimizes collision and control effort.
    traj.learn_init()

    #Change start state from 18-vector (with rotation as a rotation matrix) to 12 vector (with rotation as a rotation vector)
    start_state = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()

    agent_cfg['x0'] = start_state
    # Initialize the agent. Evolves the agent with time and interacts with the simulator (Blender) to get observations.
    agent = Agent(agent_cfg, camera_cfg, blender_cfg)

    # State estimator. Takes the observations from Agent class and performs filtering to get a state estimate (12-vector)
    filter = Estimator(filter_cfg, agent, start_state, get_rays_fn=get_rays_fn, render_fn=render_fn)
    filter.basefolder = basefolder

    true_states = start_state.cpu().detach().numpy()

    steps = traj.get_actions().shape[0]

    noise_std = extra_cfg['mpc_noise_std']
    noise_mean = extra_cfg['mpc_noise_mean']

    try:
        for iter in trange(steps):
            # In MPC style, take the next action recommended from the planner
            if iter < steps - 5:
                action = traj.get_next_action().clone().detach()
            else:
                action = traj.get_actions()[iter - steps + 5, :]

            noise = torch.normal(noise_mean, noise_std)

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.
            true_pose, true_state, gt_img = agent.step(action, noise=noise)
            true_states = np.vstack((true_states, true_state))

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance.
            state_est = filter.estimate_state(gt_img, true_pose, action)

            if iter < steps - 5:
                #state estimate is 12-vector. Transform to 18-vector
                state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)

                # Let the planner know where the agent is estimated to be
                traj.update_state(state_est)

                # Replan from the state estimate
                traj.learn_update(iter)
        return

    except KeyboardInterrupt:
        return