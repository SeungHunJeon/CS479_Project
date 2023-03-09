import torch
import numpy as np
import torch.nn as nn
import time
class MPPI():

    def __init__(self,
                 dynamics,
                 encoder,
                 encoder_ROA,
                 actor,
                 environment,
                 n_samples,
                 horizon,
                 gamma,
                 device,
                 use_dynamics,
                 inertial_dim):

        self.dynamics = dynamics
        self.n_samples = n_samples
        self.horizon = horizon
        self.device = device
        self.cur_observation = None
        self.cur_latent = None
        self.cur_state = None
        self.encoder = encoder
        self.encoder_ROA = encoder_ROA
        self.encoder_ROA_Rollout = encoder_ROA
        self.env = environment
        self.criteria = nn.MSELoss()
        self.gamma = gamma
        self.actor = actor
        self.use_dynamics = use_dynamics
        self.best_cost = 10000
        self.best_actions = None
        self.best_future_states = None
        self.inertial_dim = inertial_dim
    def smoothing_actions(self, action_batch, return_batch):

        smoothed_action = None

        return smoothed_action

    def filter_for_encode_from_obs(self, obs_batch):
        filtered_obs = []
        for i in range(self.encoder.architecture.hist_num):
            filtered_obs.append(obs_batch[...,
                                (self.encoder.architecture.block_dim)*i:
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.ext_dim - self.inertial_dim])

        if isinstance(filtered_obs[0], np.ndarray):
            filtered_obs = np.concatenate(filtered_obs, axis=-1)
            return torch.Tensor(filtered_obs)
        if isinstance(filtered_obs[0], torch.Tensor):
            filtered_obs = torch.cat(filtered_obs, dim=-1)
            return filtered_obs

    def sampling_actions(self, latent_batch):
        """
        :param state_batch: state batch
        :return: get actions from estimated state batch
        """

        # latent_batch = self.encoder(state_batch)

        """
        action sampling from uniform distribution
        """
        # sampled_action = 6 * (np.random.rand(self.n_samples, 2).astype(np.float32) - 0.5)
        """
        action sampling from uniform distribution & rejection through current policy
        """
        action_ll, actions_log_prob = self.actor.sample(latent_batch)

        """
        
        """

        sampled_action = action_ll

        return sampled_action

    def cost_function(self, goal_state_batch, state_batch):
        dist = torch.square(goal_state_batch[...,:2] - state_batch[...,:2])
        dist=dist.mean(dim=[0,-1])
        cost_batch = dist
        print(torch.min(cost_batch))
        print(self.best_cost)
        if (self.best_cost > torch.min(cost_batch)):
            self.best_cost = torch.min(cost_batch)

        return cost_batch, torch.argmin(cost_batch, dim=0)

    def compute_rollout(self, goal_state, cur_state, cur_observation):
        goal_state = torch.Tensor(goal_state)
        goal_state = goal_state.unsqueeze(0)
        goal_state = goal_state.unsqueeze(0)
        goal_state_batch = goal_state.repeat(self.horizon, self.n_samples, 1)
        # goal_state_batch = goal_state_batch.repeat(self.horizon, 1, 0)

        total_state_batch = []
        total_action_batch = []

        self.cur_state = cur_state
        self.cur_observation = cur_observation

        cur_state_batch = self.cur_state
        cur_observation_batch = self.cur_observation
        filtered_obs = self.filter_for_encode_from_obs(cur_observation_batch)
        cur_latent_batch = self.encoder_ROA.evaluate(filtered_obs)
        # self.encoder_ROA_Rollout = self.encoder_ROA
        for i in range(self.horizon):
            actions = self.sampling_actions(cur_latent_batch)
            total_action_batch.append(torch.Tensor(actions))

            # Dynamics model
            # dynamics_input = torch.Tensor(np.concatenate([actions, cur_latent_batch], axis=-1))
            # predicted_state = self.dynamics.predict(dynamics_input)
            # print(predicted_state.shape)
            # predicted_state_z = torch.Tensor(np.zeros([300])).unsqueeze(-1)
            # predicted_state = torch.cat([predicted_state, predicted_state_z],dim=-1)
            # next_state_batch = predicted_state + cur_state_batch
            # next_observation_batch = self.env.step_rollout(actions)

            # Rollout
            toc = time.time()
            self.env.step_rollout(actions)
            tic = time.time()
            print("time consuming for step rollout : ", tic-toc)
            next_observation_batch = self.env.observe_Rollout(False)
            next_state_batch = self.env.get_obj_pos()

            total_state_batch.append(torch.Tensor(next_state_batch))
            cur_state_batch = next_state_batch
            cur_observation_batch = next_observation_batch
            cur_latent_batch = self.encoder_ROA.evaluate(self.filter_for_encode_from_obs(cur_observation_batch))

        # I don't know exact dimension ..
        total_state_batch = torch.stack(total_state_batch, dim=0)
        total_action_batch = torch.stack(total_action_batch, dim=0)

        # get cost compare between total_state_batch & goal_state_batch
        # TODO
        cost_batch, min_idx = self.cost_function(goal_state_batch, total_state_batch)

        if(self.best_cost == torch.min(cost_batch) or self.best_actions.shape[0] < 2):
            self.best_actions = total_action_batch[:, min_idx, :]
            self.best_cost = torch.min(cost_batch)
            self.best_future_states = total_state_batch[:, min_idx, :]

        optimal_action = self.best_actions[0, ...]

        self.best_actions = self.best_actions[1:, ...]

        # action smoothing via cost batch
        smoothed_action = self.smoothing_actions(total_action_batch, cost_batch)

        return optimal_action.unsqueeze(0), self.best_future_states