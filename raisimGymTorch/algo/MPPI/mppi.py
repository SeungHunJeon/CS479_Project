import torch
import numpy as np
import torch.nn as nn

class MPPI():

    def __init__(self,
                 dynamics,
                 encoder,
                 actor,
                 environment,
                 n_samples,
                 horizon,
                 gamma,
                 device):

        self.dynamics = dynamics
        self.n_samples = n_samples
        self.horizon = horizon
        self.device = device
        self.cur_observation = None
        self.cur_latent = None
        self.cur_state = None
        self.encoder = encoder
        self.env = environment
        self.criteria = nn.MSELoss()
        self.gamma = gamma
        self.actor = actor

    def smoothing_actions(self, action_batch, return_batch):

        smoothed_action = None

        return smoothed_action

    def sampling_actions(self, state_batch):
        """
        :param state_batch: state batch
        :return: get actions from estimated state batch
        """

        # latent_batch = self.encoder(state_batch)

        return True

    def compute_rollout_cost(self, goal_state):
        goal_state = goal_state.unsqueeze(0)
        goal_state = goal_state.unsqueeze(0)
        goal_state_batch = goal_state.repeat(self.n_samples, 1)
        goal_state_batch = goal_state_batch.repeat(self.horizon, 0)

        total_state_batch = []
        total_action_batch = []

        cur_state_batch = self.cur_state.repeat(self.n_samples, 0)
        cur_observation_batch = self.cur_observation.repeat(self.n_samples, 0)
        cur_latent_batch = self.encoder(cur_observation_batch)

        for i in range(self.horizon):
            actions = self.sampling_actions(cur_latent_batch)
            next_state_batch = self.dynamics.predict(cur_state_batch, actions, cur_latent_batch)

            # environment rollout or dynamics model
            next_observation_batch = self.env.step(cur_observation_batch, actions)

            total_state_batch.append(next_state_batch)
            cur_state_batch = next_state_batch
            cur_observation_batch = next_observation_batch
            cur_latent_batch = self.encoder(cur_observation_batch)

        # I don't know exact dimension ..
        total_state_batch = torch.cat(total_state_batch, dim=-1)

        # get cost compare between total_state_batch & goal_state_batch
        # TODO
        cost_batch = None

        # action smoothing via cost batch
        smoothed_action = self.smoothing_actions(total_action_batch, cost_batch)

        return True