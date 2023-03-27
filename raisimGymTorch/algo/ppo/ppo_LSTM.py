from datetime import datetime
import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage_encoding import RolloutStorage
# import matplotlib.pyplot as plt



class PPO:
    def __init__(self,
                 actor,
                 critic,
                 encoder,
                 decoder,
                 obj_f_dynamics,
                 obs_f_dyanmics,
                 latent_f_dynamics,
                 obs_shape,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 encoder_ROA=None,
                 estimator=None,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 lambda_ROA=1e-4,
                 lambdaDecayFactor=0.999,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 num_history_batch = 4,
                 inertial_dim=15):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.encoder_ROA = encoder_ROA
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, obs_shape, actor.action_shape, device)
        self.lambda_ROA = lambda_ROA
        self.lambdaDecayFactor = lambdaDecayFactor
        self.inertial_dim = inertial_dim
        self.estimator = estimator
        self.obj_f_dynamics = obj_f_dynamics
        self.obs_f_dynamics = obs_f_dyanmics
        self.latent_f_dynamics = latent_f_dynamics
        self.decoder = decoder

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.encoder_input_dim = self.encoder.architecture.input_shape[0]
        self.optimizer = optim.Adam([*self.actor.parameters(),
                                     *self.critic.parameters(),
                                     *self.encoder.parameters(),
                                     *self.encoder_ROA.parameters(),
                                     *self.estimator.parameters()
                                     # *self.obs_f_dynamics.parameters()
                                     # *self.decoder.parameters()
                                     ], lr=learning_rate)

        self.dynamics_optimizer = optim.Adam([*self.obj_f_dynamics.parameters(),
                                              *self.latent_f_dynamics.parameters()],
                                             lr=learning_rate)

        # remove estimator
        # self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters()], lr=learning_rate)
        self.device = device
        self.criteria = nn.MSELoss()

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.mean_value_loss = None
        self.mean_surrogate_loss = None
        self.mean_noise_std = None

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

        # Encoder param
        self.num_history_batch = num_history_batch

        # LSTM param

        # ROA loss
        self.loss_ROA = None
        self.lambda_loss_ROA = None

        # Estimator loss
        self.estimator_loss = None

        # Obj F model lsos
        self.obj_f_dynamics_loss = None

        # Entropy
        self.entropy_mean = None

        # latent f dynamics loss
        self.latent_f_dyn_loss = None

        # Decoder loss
        self.decoder_loss = None

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.inference_mode():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions

    def step(self, value_obs, obs, rews, dones):

        """
        remove actor_obs, value_obs -> into encoder_obs
        """
        self.storage.add_transitions(self.actor_obs, value_obs, obs, self.actions, self.actor.action_mean, self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):

        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(log_this_iteration)
        self.mean_value_loss = infos['mean_value_loss']
        self.mean_surrogate_loss = infos['mean_surrogate_loss']
        self.mean_noise_std = self.actor.distribution.std.mean()
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])


    def filter_for_decoder_from_obs(self, obs_batch, latent_batch):

        decoder_output = []

        decoder_input = latent_batch.reshape((-1, self.num_envs, self.encoder.architecture.hidden_dim))

        decoder_output.append(obs_batch[..., self.encoder.architecture.block_dim * (self.num_history_batch - 1)
                                            :
                                            self.encoder.architecture.block_dim * (self.num_history_batch - 1)
                                            + self.encoder.architecture.pro_dim
                                            + self.encoder.architecture.ext_dim
                                            - self.inertial_dim])

        decoder_output.append(obs_batch[..., self.encoder.architecture.block_dim * (self.num_history_batch - 1)
                                            + self.encoder.architecture.pro_dim
                                            + self.encoder.architecture.ext_dim
                                            :
                                            self.encoder.architecture.block_dim * (self.num_history_batch - 1)
                                            + self.encoder.architecture.pro_dim
                                            + self.encoder.architecture.ext_dim
                                            + self.encoder.architecture.act_dim])

        decoder_output = torch.cat(decoder_output, dim=-1)

        return decoder_input, decoder_output


    def filter_for_latent_f_dynamics_from_obs(self, obs_batch, latent_batch_):
        latent_f_dynamics_input = []

        latent_batch = latent_batch_.clone().detach()

        latent_batch = latent_batch.reshape((-1, self.num_envs, self.encoder.architecture.hidden_dim))

        latent_batch_input = latent_batch[:-1, ...]

        latent_predict_true = latent_batch[1:, ...]

        action_batch_f = obs_batch[:-1, ..., self.encoder.architecture.block_dim * self.num_history_batch
                                          - self.encoder.architecture.act_dim
                                          - self.encoder.architecture.dyn_info_dim:
                                          self.encoder.architecture.block_dim * self.num_history_batch
                                          - self.encoder.architecture.dyn_info_dim]

        latent_f_dynamics_input.append(latent_batch_input)
        latent_f_dynamics_input.append(action_batch_f)

        latent_f_dynamics_input = torch.cat(latent_f_dynamics_input, dim=-1)

        return latent_f_dynamics_input, latent_predict_true

    def filter_for_obj_f_dynamics_from_obs(self, obs_batch, latent_batch_):

        obj_f_dynamics_obs = []

        obj_f_dynamics_true = []

        dyn_obs_current = obs_batch[1:-1, ..., self.encoder.architecture.block_dim * self.num_history_batch
                                       - self.encoder.architecture.dyn_info_dim:
                                       self.encoder.architecture.block_dim * self.num_history_batch]

        dyn_obs_next = obs_batch[2:, ..., self.encoder.architecture.block_dim * self.num_history_batch
                                           - self.encoder.architecture.dyn_info_dim:
                                           self.encoder.architecture.block_dim * self.num_history_batch]

        action_info = obs_batch[2:, ..., self.encoder.architecture.block_dim * self.num_history_batch
                                         - self.encoder.architecture.act_dim
                                         - self.encoder.architecture.dyn_info_dim:
                                         self.encoder.architecture.block_dim * self.num_history_batch
                                         - self.encoder.architecture.dyn_info_dim]

        latent_batch = latent_batch_.reshape((-1, self.num_envs, self.encoder.architecture.hidden_dim))

        latent_batch = latent_batch[1:-1]

        obj_pos_res = (dyn_obs_next[..., :3] - dyn_obs_current[..., :3])
        robot_pos_res = (dyn_obs_next[..., 3:6] - dyn_obs_current[..., 3:6])
        obj_vel_res = (dyn_obs_next[..., 6:9] - dyn_obs_current[..., 6:9])
        robot_vel_res = (dyn_obs_next[..., 9:12] - dyn_obs_current[..., 9:12])
        obj_avel_res = (dyn_obs_next[..., 12:15] - dyn_obs_current[..., 12:15])
        robot_avel_res = (dyn_obs_next[..., 15:18] - dyn_obs_current[..., 15:18])
        obj_rot_x_res = (dyn_obs_next[..., 18:21] - dyn_obs_current[..., 18:21])
        robot_rot_x_res = (dyn_obs_next[..., 21:24] - dyn_obs_current[..., 21:24])
        obj_geometry = dyn_obs_current[..., 24:27]
        obj_robot_dist = dyn_obs_current[..., :3] - dyn_obs_current[..., 3:6]
        obj_robot_rot_rel = dyn_obs_current[..., 18:21] - dyn_obs_current[..., 21:24]

        obj_f_dynamics_obs.append(obj_pos_res[:-1])
        obj_f_dynamics_obs.append(robot_pos_res[:-1])
        obj_f_dynamics_obs.append(obj_vel_res[:-1])
        obj_f_dynamics_obs.append(robot_vel_res[:-1])
        obj_f_dynamics_obs.append(obj_avel_res[:-1])
        obj_f_dynamics_obs.append(robot_avel_res[:-1])
        obj_f_dynamics_obs.append(obj_rot_x_res[:-1])
        obj_f_dynamics_obs.append(robot_rot_x_res[:-1])
        obj_f_dynamics_obs.append(obj_robot_dist[:-1])
        obj_f_dynamics_obs.append(obj_robot_rot_rel[:-1])
        obj_f_dynamics_obs.append(obj_geometry[:-1])
        obj_f_dynamics_obs.append(action_info[:-1])
        obj_f_dynamics_obs.append(latent_batch[:-1])

        obj_f_dynamics_true.append(obj_pos_res[1:])
        obj_f_dynamics_true.append(robot_pos_res[1:])
        obj_f_dynamics_true.append(obj_vel_res[1:])
        obj_f_dynamics_true.append(robot_vel_res[1:])
        obj_f_dynamics_true.append(obj_avel_res[1:])
        obj_f_dynamics_true.append(robot_avel_res[1:])
        obj_f_dynamics_true.append(obj_rot_x_res[1:])
        obj_f_dynamics_true.append(robot_rot_x_res[1:])
        obj_f_dynamics_true.append(obj_robot_dist[1:])
        obj_f_dynamics_true.append(obj_robot_rot_rel[1:])
        # Need to make LOG function

        # 임의로 size up 할까? 의미가 있나 근데 ..? Normalize 마렵긴 한데 value를 어케 찾을지 모르겠네 ? prediction 만 하고 gradient 구하는거만 normalize한다? 흠 ..?;
        obj_f_dynamics_obs = torch.cat(obj_f_dynamics_obs, dim=-1)

        obj_f_dynamics_true = torch.cat(obj_f_dynamics_true, dim=-1)

        # print(torch.mean(obj_f_dynamics_obs))

        # print(torch.mean(obj_f_dynamics_true))

        return obj_f_dynamics_obs, obj_f_dynamics_true

    def filter_for_encode_from_obs(self, obs_batch):
        filtered_obs = []
        for i in range(self.num_history_batch):
            # filtered_obs.append(obs_batch[...,
            #                     (self.encoder.architecture.block_dim)*i:
            #                     (self.encoder.architecture.block_dim)*i
            #                     + self.encoder.architecture.pro_dim
            #                     + self.encoder.architecture.ext_dim])

            filtered_obs.append(obs_batch[...,
                                (self.encoder.architecture.block_dim)*i:
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.ext_dim
                                + self.encoder.architecture.act_dim])

        if isinstance(filtered_obs[0], numpy.ndarray):
            filtered_obs = numpy.concatenate(filtered_obs, axis=-1)
            return torch.Tensor(filtered_obs)
        if isinstance(filtered_obs[0], torch.Tensor):
            filtered_obs = torch.cat(filtered_obs, dim=-1)
            return filtered_obs
    def encode(self, obs_batch):
        obs_batch = self.filter_for_encode_from_obs(obs_batch)
        output = self.encoder.evaluate_update(obs_batch)

        return output

    def encode_ROA(self, obs):
        output = self.encoder_ROA.evaluate_update(obs)

        return output
    # def plot_grad_flow(self, named_parameters):
    #     ave_grads = []
    #     layers = []
    #     for n, p in named_parameters:
    #         if(p.requires_grad) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grad = p.grad.clone().detach().to('cpu')
    #             ave_grads.append(ave_grad.abs().mean())
    #     plt.plot(ave_grads, alpha=0.3, color="b")
    #     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(xmin=0, xmax=len(ave_grads))
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.savefig('gradient.png')

    def get_obs_ROA(self, obs_batch):
        obs_ROA_batch = []


        for i in range(self.num_history_batch):
            # Get proprioceptive part of observation
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.block_dim)*i:
                                 (self.encoder.architecture.block_dim)*i
                                                + self.encoder.architecture.pro_dim])

            # Get Exteroceptive part of observation except inertial parameter
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.block_dim)*i
                                 + self.encoder.architecture.pro_dim:
                                 (self.encoder.architecture.block_dim)*i
                                 + self.encoder.architecture.pro_dim
                                 + self.encoder.architecture.ext_dim - self.inertial_dim])

            # Get action part of observation
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.block_dim)*i
                                 + self.encoder.architecture.pro_dim
                                 + self.encoder.architecture.ext_dim:
                                 (self.encoder.architecture.block_dim)*i
                                 + self.encoder.architecture.pro_dim
                                 + self.encoder.architecture.ext_dim
                                 + self.encoder.architecture.act_dim])

        # Distillate the true inertial parameter (oracle)



        estimator_true_data = (obs_batch[-1, :,
                               (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                               + self.encoder.architecture.pro_dim
                               + self.encoder.architecture.ext_dim - self.inertial_dim:
                               (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                               + self.encoder.architecture.pro_dim
                               + self.encoder.architecture.ext_dim
                               ])


        obs_ROA_batch = torch.cat(obs_ROA_batch, dim=-1)
        estimator_true_data = estimator_true_data.reshape(-1, self.inertial_dim)

        return obs_ROA_batch, estimator_true_data.detach()

    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        self.loss_ROA = 0
        self.lambda_loss_ROA = 0
        self.estimator_loss = 0
        self.obj_f_dynamics_loss = 0
        self.entropy_mean = 0
        self.latent_f_dyn_loss = 0
        # self.decoder_loss = 0

        self.lambda_ROA = pow(self.lambda_ROA, self.lambdaDecayFactor)
        print(self.lambda_ROA)
        for epoch in range(self.num_learning_epochs):
            for obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                self.encoder.architecture.reset()
                self.encoder_ROA.architecture.reset()

                obs_ROA_batch, estimator_true_data = self.get_obs_ROA(obs_batch)

                """
                For model update, we'll use
                1. action batch except last action -> splicing
                2. obs_concat_ROA -> splicing
                3. 
                """

                latent = self.encode(obs_batch)
                latent_d = latent.clone().detach()
                latent_ROA = self.encode_ROA(obs_ROA_batch)
                latent_ROA_d = latent_ROA.clone().detach()



                latent_f_dyn_input, latent_f_dyn_predict_true = self.filter_for_latent_f_dynamics_from_obs(obs_batch, latent)

                # decoder_input, decoder_output_true = self.filter_for_decoder_from_obs(obs_batch, latent)
                #
                # decoder_predict = self.decoder.evaluate(decoder_input)

                obj_f_dyn_input, obj_f_dyn_true = self.filter_for_obj_f_dynamics_from_obs(obs_batch, latent)

                # estimator_input = self.encoder_ROA.evaluate_update(obs_ROA_batch[-1, ...]).clone().detach().reshape(-1, self.encoder_ROA.architecture.hidden_dim)

                estimator_input = self.encode(obs_batch[-1, ...]).reshape(-1, self.encoder.architecture.hidden_dim)

                estimator_loss = self.criteria(self.estimator.evaluate(estimator_input), estimator_true_data)

                lambda_loss_ROA = self.lambda_ROA * self.criteria(latent, latent_ROA_d)
                loss_ROA = self.criteria(latent_d, latent_ROA)


                if(self.encoder.architecture.is_decouple):
                    latent_for_update = latent[self.num_history_batch-1::self.num_history_batch, :]
                    actions_log_prob_batch, entropy_batch = self.actor.evaluate(latent_for_update, actions_batch)
                    value_batch = self.critic.evaluate(latent_for_update)

                else:
                    actions_log_prob_batch, entropy_batch = self.actor.evaluate(latent, actions_batch)
                    value_batch = self.critic.evaluate(latent)

                # Adjusting the learning rate using KL divergence
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.distribution.std

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                obj_f_dyn_predict = self.obj_f_dynamics.evaluate(obj_f_dyn_input)

                obj_f_dynamics_loss = self.criteria(obj_f_dyn_predict, obj_f_dyn_true)

                latent_f_dyn_predict = self.latent_f_dynamics.evaluate(latent_f_dyn_input)

                latent_f_dyn_loss = self.criteria(latent_f_dyn_predict, latent_f_dyn_predict_true)



                # decoder_loss = self.criteria(decoder_output_true, decoder_predict)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                       + lambda_loss_ROA \
                       + loss_ROA \
                       + obj_f_dynamics_loss \
                       + latent_f_dyn_loss \
                       + estimator_loss
                       # + decoder_loss

                # dynamics_loss = obj_f_dynamics_loss

                # Add kl divergence term to normalize the latent vector

                # Gradient step
                self.optimizer.zero_grad()
                self.dynamics_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_([*self.actor.parameters(),
                                          *self.critic.parameters(),
                                          *self.encoder.parameters(),
                                          *self.encoder_ROA.parameters(),
                                          # *self.decoder.parameters()
                                          *self.estimator.parameters()
                                          ], self.max_grad_norm)

                nn.utils.clip_grad_norm_([*self.obj_f_dynamics.parameters(),
                                          *self.latent_f_dynamics.parameters()]
                                         ,self.max_grad_norm)

                # remove estimator
                # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters()], self.max_grad_norm)



                self.optimizer.step()
                self.dynamics_optimizer.step()
                # self.plot_grad_flow(self.encoder[0].architecture.named_parameters())

                if log_this_iteration:
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    self.loss_ROA += loss_ROA.item()
                    self.lambda_loss_ROA += lambda_loss_ROA.item()
                    self.estimator_loss += estimator_loss.item()
                    self.obj_f_dynamics_loss += obj_f_dynamics_loss.item()
                    self.entropy_mean = (self.entropy_coef * entropy_batch.mean()).item()
                    self.latent_f_dyn_loss += latent_f_dyn_loss.item()
                    # self.decoder_loss += decoder_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            self.loss_ROA /= num_updates
            self.lambda_loss_ROA /= num_updates
            self.estimator_loss /= num_updates
            self.obj_f_dynamics_loss /= num_updates
            self.entropy_mean /= num_updates
            self.obj_f_dynamics_loss /= num_updates
            # self.decoder_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
