import torch
import numpy as np
import torch.nn as nn
import time
import copy

class MPPI():

    def __init__(self,
                 latent_f_dynamics,
                 obj_f_dynamics,
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

        self.latent_f_dynamics = latent_f_dynamics
        self.obj_f_dynamics = obj_f_dynamics
        self.n_samples = n_samples
        self.horizon = horizon
        self.device = device
        self.cur_observation = None
        self.cur_latent = None
        self.cur_state = None
        self.encoder = encoder
        self.encoder_ROA = encoder_ROA
        self.encoder_ROA_Rollout = copy.deepcopy(self.encoder_ROA)
        self.env = environment
        self.criteria = nn.MSELoss()
        self.gamma = gamma
        self.actor = actor
        self.use_dynamics = use_dynamics
        self.best_cost = 10000
        self.best_actions = None
        self.best_future_states = None
        self.inertial_dim = inertial_dim
        self.action_mean = self.env.mean[self.encoder.architecture.block_dim * self.encoder.architecture.hist_num
                                         - self.encoder.architecture.act_dim
                                         - self.encoder.architecture.dyn_info_dim:
                                         self.encoder.architecture.block_dim * self.encoder.architecture.hist_num
                                         - self.encoder.architecture.dyn_info_dim]
        self.action_var = self.env.var[self.encoder.architecture.block_dim * self.encoder.architecture.hist_num
                                       - self.encoder.architecture.act_dim
                                       - self.encoder.architecture.dyn_info_dim:
                                       self.encoder.architecture.block_dim * self.encoder.architecture.hist_num
                                       - self.encoder.architecture.dyn_info_dim]

    def smoothing_actions(self, action_batch, return_batch):

        smoothed_action = None

        return smoothed_action

    def filter_for_obj_f_dynamics_from_obs(self, obs_batch, latent_batch_, action_batch):
        obj_f_dynamics_obs = []


        self.env.mean
        self.env.var
        # TODO (Need Denormalize) action_batch is from random sampling ..
        action_info = action_batch
        # action_info *= 2
        # denormalize using obMean obsVar
        #
        #

        dyn_obs = obs_batch[..., self.encoder.architecture.block_dim
                                       - self.encoder.architecture.dyn_info_dim:
                                       self.encoder.architecture.block_dim]

        latent_batch = latent_batch_.reshape((-1, self.n_samples, self.encoder.architecture.hidden_dim))

        latent_batch = latent_batch[1:-1]

        Obj_Pos = dyn_obs[..., :3].unsqueeze(-1)
        ee_Pos = dyn_obs[..., 3:6].unsqueeze(-1)
        Obj_Vel = dyn_obs[..., 6:9].unsqueeze(-1)
        Robot_Vel = dyn_obs[..., 9:12].unsqueeze(-1)
        Obj_AVel = dyn_obs[..., 12:15].unsqueeze(-1)
        Robot_AVel = dyn_obs[..., 15:18].unsqueeze(-1)
        Obj_RotMat = dyn_obs[..., 18:27].unsqueeze(-1)
        Obj_RotMat = Obj_RotMat.reshape(Obj_RotMat.shape[0], Obj_RotMat.shape[1], -1, 3)
        Robot_RotMat = dyn_obs[..., 27:36].unsqueeze(-1)
        Robot_RotMat = Robot_RotMat.reshape(Obj_RotMat.shape[0], Obj_RotMat.shape[1], -1, 3)
        Robot_RotMat_transpose = torch.transpose(Robot_RotMat, -2, -1)
        Obj_Geometry = dyn_obs[..., 36:].unsqueeze(-1)

        obj_f_dynamics_obs.append((Robot_RotMat_transpose @ (Obj_Vel - Robot_Vel)).squeeze(-1))
        obj_f_dynamics_obs.append((Robot_RotMat_transpose @ (Obj_AVel - Robot_AVel)).squeeze(-1))
        obj_f_dynamics_obs.append((Robot_RotMat_transpose @ (Obj_Pos - ee_Pos)).squeeze(-1))
        obj_f_dynamics_obs.append((Robot_RotMat[..., 0, :] - Obj_RotMat[..., 0, :]).squeeze(-1))
        obj_f_dynamics_obs.append((Obj_Geometry).squeeze(-1))
        obj_f_dynamics_obs.append(action_info)
        obj_f_dynamics_obs.append(latent_batch)

        obj_f_dynamics_obs = torch.cat(obj_f_dynamics_obs, dim=-1)

        return obj_f_dynamics_obs

    def filter_for_encode_from_obs(self, obs_batch):
        filtered_obs = []
        for i in range(self.encoder.architecture.hist_num):
            filtered_obs.append(obs_batch[...,
                                (self.encoder.architecture.block_dim)*i:
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.ext_dim - self.inertial_dim])

            filtered_obs.append(obs_batch[...,
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.ext_dim:
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.ext_dim
                                + self.encoder.architecture.act_dim])

        if isinstance(filtered_obs[0], np.ndarray):
            filtered_obs = np.concatenate(filtered_obs, axis=-1)
            return torch.Tensor(filtered_obs).to(self.device)
        if isinstance(filtered_obs[0], torch.Tensor):
            filtered_obs = torch.cat(filtered_obs, dim=-1)
            return filtered_obs.to(self.device)

    def sampling_actions(self, latent_batch):
        """
        :param state_batch: state batch
        :return: get actions from estimated state batch
        """

        # latent_batch = self.encoder(state_batch)

        """
        action sampling from uniform distribution
        """
        # sampled_action = 5 * (np.random.rand(self.n_samples, 2).astype(np.float32) - 0.5)
        """
        action sampling from uniform distribution & rejection through current policy
        """
        sampled_action, actions_log_prob = self.actor.sample(latent_batch)

        """
        
        """

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
        print("cur latent batch : ", cur_latent_batch[0])
        # cur_latent_batch = cur_latent_batch.repeat(self.n_samples, 1)
        self.encoder_ROA_Rollout = copy.deepcopy(self.encoder_ROA)

        for i in range(self.horizon):
            actions = self.sampling_actions(cur_latent_batch)
            total_action_batch.append(torch.Tensor(actions))

            if(self.use_dynamics):
                # Latent forward dynamics model
                # cur_late
                # latent_dynamics_input = torch.cat(cur_latent_batch)
                latent_dynamics_input = torch.Tensor(np.concatenate([cur_latent_batch.cpu().numpy(), actions], axis=-1)).to(self.device)
                predicted_latent_batch = self.latent_f_dynamics.predict(latent_dynamics_input)
                # TODO (denormalize obj batch)
                cur_latent_batch = predicted_latent_batch

            else:
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
                cur_latent_batch = self.encoder_ROA_Rollout.evaluate(self.filter_for_encode_from_obs(cur_observation_batch))



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

        # Validate latent forward dynamics model is feasible
        normalized_optimal_action = (torch.Tensor(optimal_action).unsqueeze(0).numpy() - self.action_mean) / self.action_var
        latent_dynamics_input = torch.Tensor(np.concatenate([cur_latent_batch[0].unsqueeze(0).cpu().numpy(), normalized_optimal_action], axis=-1)).to(self.device)
        predicted_latent = self.latent_f_dynamics.predict(latent_dynamics_input)
        print("predicted latent :", predicted_latent)

        # action smoothing via cost batch
        smoothed_action = self.smoothing_actions(total_action_batch, cost_batch)

        return optimal_action.unsqueeze(0), self.best_future_states