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
                 encoder_DR,
                 obj_f_dynamics,
                 latent_f_dynamics,
                 obs_shape,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 encoder_ROA=None,
                 estimator=None,
                 estimator_cov=None,
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
                 inertial_dim=15,
                 domain_randomization=False):

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
        self.estimator_cov = estimator_cov
        self.obj_f_dynamics = obj_f_dynamics
        self.latent_f_dynamics = latent_f_dynamics

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([
            # *self.actor.parameters(),
            #                          *self.critic.parameters(),
                                     *self.encoder.parameters(),
                                     # *self.encoder_ROA.parameters(),
                                     *self.estimator.parameters(),
                                     *self.estimator_cov.parameters()
                                     # *self.obs_f_dynamics.parameters()
                                     # *self.decoder.parameters()
                                     ], lr=learning_rate)


        # self.dynamics_optimizer = optim.Adam([*self.obj_f_dynamics.parameters(),
        #                                       *self.latent_f_dynamics.parameters()],
        #                                      lr=learning_rate)

        # remove estimator
        self.device = device
        self.GaussianNLLLoss = nn.GaussianNLLLoss()
        self.MSELoss = nn.MSELoss()

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

        # is domain randomization manner
        self.domain_randomization = domain_randomization


    def act(self, actor_obs):
        # self.actor_obs = actor_obs
        # with torch.inference_mode():
        #     self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions

    def estimator_pre_process(self, anchor_points):
        return anchor_points[..., -24:] - anchor_points[..., :24]

    def step(self, value_obs, obs, rews, dones, contact=False, privileged_info=None, anchors=None):

        """
        remove actor_obs, value_obs -> into encoder_obs
        """
        self.storage.add_transitions(self.actor_obs, value_obs, obs, self.actions, self.actor.action_mean, self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob, contact=contact, privileged_info=privileged_info, anchors=anchors)

    def update(self, actor_obs, value_obs, log_this_iteration, update):

        # last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns()
        _, infos = self._train_step(log_this_iteration)
        # self.mean_value_loss = infos['mean_value_loss']
        # self.mean_surrogate_loss = infos['mean_surrogate_loss']
        # self.mean_noise_std = self.actor.distribution.std.mean()
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

    def filter_for_encode_from_obs(self, obs_batch):
        filtered_obs = []
        # print(obs_batch.shape)
        for i in range(self.num_history_batch):
            filtered_obs.append(obs_batch[...,
                                (self.encoder.architecture.block_dim)*i:
                                (self.encoder.architecture.block_dim)*i
                                + self.encoder.architecture.pro_dim
                                + self.encoder.architecture.act_dim])

        if isinstance(filtered_obs[0], numpy.ndarray):
            filtered_obs = numpy.concatenate(filtered_obs, axis=-1)
            return torch.Tensor(filtered_obs)
        if isinstance(filtered_obs[0], torch.Tensor):
            filtered_obs = torch.cat(filtered_obs, dim=-1)
            return filtered_obs
    def encode(self, obs_batch):
        output = self.encoder.evaluate_update(self.filter_for_encode_from_obs(obs_batch))
        return output

    # def encode_ROA(self, obs):
    #     output = self.encoder_ROA.evaluate_update(obs)
    #
    #     return output
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


        if isinstance(obs_ROA_batch[0], numpy.ndarray):
            filtered_obs = numpy.concatenate(obs_ROA_batch, axis=-1)
            return torch.Tensor(filtered_obs)
        if isinstance(obs_ROA_batch[0], torch.Tensor):
            filtered_obs = torch.cat(obs_ROA_batch, dim=-1)
            return filtered_obs

    def filter_for_estimation(self, obs_batch, contact_mask):

        estimator_true_data = obs_batch[:, :,
                              (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                              + self.encoder.architecture.pro_dim
                              + self.encoder.architecture.ext_dim - self.inertial_dim:
                              (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                              + self.encoder.architecture.pro_dim
                              + self.encoder.architecture.ext_dim
                              ]

        estimator_true_data_masked = estimator_true_data[contact_mask]

        estimator_true_data_masked = estimator_true_data_masked.reshape(-1, self.inertial_dim)

        return estimator_true_data_masked

    def filter_for_actor(self, obs_batch, latent_batch):
        filtered_obs = []

        filtered_obs.append(obs_batch[...,
                            0
                            :
                            self.encoder.architecture.block_dim])

        filtered_obs.append(obs_batch[...,
                            (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                            :
                            (self.encoder.architecture.block_dim)*(self.num_history_batch-1)
                            + self.encoder.architecture.pro_dim])

        if isinstance(filtered_obs[0], numpy.ndarray):
            filtered_obs = numpy.concatenate(filtered_obs, axis=-1)
            if isinstance(latent_batch[0], numpy.ndarray):
                numpy.concatenate([latent_batch, filtered_obs], axis=-1)
                # filtered_obs = torch.cat(filtered_obs, dim=-1)
                return numpy.concatenate([latent_batch, filtered_obs], axis=-1)

            if isinstance(latent_batch[0], torch.Tensor):
                filtered_obs = torch.Tensor(filtered_obs)
                filtered_obs = filtered_obs.reshape((-1, self.encoder.architecture.pro_dim * 2 + self.encoder.architecture.act_dim))
                filtered_obs = filtered_obs.to(self.device)

                return torch.cat((latent_batch, filtered_obs), dim=-1)

        if isinstance(filtered_obs[0], torch.Tensor):
            filtered_obs = torch.cat(filtered_obs, dim=-1)

            if isinstance(latent_batch[0], torch.Tensor):
                # filtered_obs = torch.cat(filtered_obs, dim=-1)
                filtered_obs = filtered_obs.reshape((-1, self.encoder.architecture.pro_dim * 2 + self.encoder.architecture.act_dim))
                filtered_obs = filtered_obs.to(self.device)

                return torch.cat((latent_batch, filtered_obs), dim=-1)


    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        self.mean_loss = 0
        self.loss_ROA = 0
        self.lambda_loss_ROA = 0
        self.estimator_loss = 0
        self.obj_f_dynamics_loss = 0
        self.entropy_mean = 0
        self.latent_f_dyn_loss = 0
        # self.mask = numpy.ones([self.num_transitions_per_env, self.num_envs])
        # self.decoder_loss = 0

        self.lambda_ROA = pow(self.lambda_ROA, self.lambdaDecayFactor)
        # print(self.lambda_ROA)
        for epoch in range(self.num_learning_epochs):
            for obs_batch, _, _, _, _, _, _, _, _, _, anchors_batch \
                    in self.batch_sampler(self.num_mini_batches):
                # privileged_batch = privileged_batch.reshape((-1, self.num_envs // self.num_mini_batches, self.inertial_dim))
                # contact_batch = contact_batch.reshape((-1, self.num_envs // self.num_mini_batches, 1))
                # contact_mask = contact_batch.bool().squeeze(-1)

                self.encoder.architecture.reset()
                # self.encoder_ROA.architecture.reset()



                # obs_ROA_batch = self.get_obs_ROA(obs_batch)

                # estimator_true_data = self.filter_for_estimation(obs_batch, contact_mask)
                # estimator_true_data = privileged_batch[contact_mask]
                """
                For model update, we'll use
                1. action batch except last action -> splicing
                2. obs_concat_ROA -> splicing
                3. 
                """

                # if(estimator_true_data.shape[0] != 0):
                #     latent_estimation = self.encode(obs_batch[contact_mask])
                #obs_batch [40, 500, 820]=>
                # print(obs_batch.shape)
                latent = self.encode(obs_batch)
                # latent_estimation = latent.reshape((-1, self.num_envs // self.num_mini_batches, self.encoder.architecture.hidden_dim))[contact_mask]

                # latent_d = latent.clone().detach()
                # latent_ROA = self.encode_ROA(obs_ROA_batch)
                # latent_ROA_d = latent_ROA.clone().detach()



                # latent_f_dyn_input, latent_f_dyn_predict_true = self.filter_for_latent_f_dynamics_from_obs(obs_batch, latent_d)

                # decoder_input, decoder_output_true = self.filter_for_decoder_from_obs(obs_batch, latent)
                #
                # decoder_predict = self.decoder.evaluate(decoder_input)

                # obj_f_dyn_input, obj_f_dyn_true = self.filter_for_obj_f_dynamics_from_obs(obs_batch, latent_d)

                # estimator_input = self.encoder_ROA.evaluate_update(obs_ROA_batch[-1, ...]).clone().detach().reshape(-1, self.encoder_ROA.architecture.hidden_dim)

                # if(estimator_true_data.shape[0] != 0):
                #     estimator_input = latent_estimation
                #     estimator_loss = self.criteria(self.estimator.evaluate(estimator_input), estimator_true_data)
                # else:
                #     estimator_loss = torch.Tensor([0]).to(self.device)

                # lambda_loss_ROA = self.lambda_ROA * self.criteria(latent, latent_ROA_d)
                # loss_ROA = self.criteria(latent_d, latent_ROA)

                actor_input = self.filter_for_actor(obs_batch, latent)
                mean = self.estimator.evaluate(actor_input)
                cov = torch.exp_(2*self.estimator_cov.evaluate(actor_input))
                # print(anchors_batch.shape)
                target = self.estimator_pre_process(anchors_batch)

                estimator_loss = self.GaussianNLLLoss(mean, target, cov)

                # estimator_loss = self.MSELoss(mean, target)

                # actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_input, actions_batch)
                # value_batch = self.critic.evaluate(actor_input)

                # Adjusting the learning rate using KL divergence
                # mu_batch = self.actor.action_mean
                # sigma_batch = self.actor.distribution.std

                # KL
                # if self.desired_kl != None and self.schedule == 'adaptive':
                #     with torch.inference_mode():
                #         kl = torch.sum(
                #             torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                #         kl_mean = torch.mean(kl)
                #
                #         if kl_mean > self.desired_kl * 2.0:
                #             self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                #         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                #             self.learning_rate = min(1e-2, self.learning_rate * 1.2)
                #
                #         for param_group in self.optimizer.param_groups:
                #             param_group['lr'] = self.learning_rate
                #
                # # Surrogate loss
                # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # surrogate = -torch.squeeze(advantages_batch) * ratio
                # surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                #                                                                    1.0 + self.clip_param)
                # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()




                # decoder_loss = self.criteria(decoder_output_true, decoder_predict)

                # Value function loss
                # if self.use_clipped_value_loss:
                #     value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                #                                                                                     self.clip_param)
                #     value_losses = (value_batch - returns_batch).pow(2)
                #     value_losses_clipped = (value_clipped - returns_batch).pow(2)
                #     value_loss = torch.max(value_losses, value_losses_clipped).mean()
                # else:
                #     value_loss = (returns_batch - value_batch).pow(2).mean()

                # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                #        + lambda_loss_ROA \
                #        + loss_ROA \
                #        + estimator_loss
                       # + obj_f_dynamics_loss \
                       # + latent_f_dyn_loss \
                loss = estimator_loss

                # dynamics_loss = obj_f_dynamics_loss

                # Add kl divergence term to normalize the latent vector

                # Gradient step
                self.optimizer.zero_grad()
                # self.dynamics_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_([
                    # *self.actor.parameters(),
                    #                       *self.critic.parameters(),
                                          *self.encoder.parameters(),
                                          *self.estimator.parameters(),
                                          *self.estimator_cov.parameters()
                                          # *self.encoder_ROA.parameters(),
                                          # *self.estimator.parameters()
                                          ], self.max_grad_norm)

                # nn.utils.clip_grad_norm_([*self.obj_f_dynamics.parameters(),
                #                           *self.latent_f_dynamics.parameters()]
                #                          ,self.max_grad_norm)

                # remove estimator
                # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters()], self.max_grad_norm)


                self.optimizer.step()
                # self.dynamics_optimizer.step()
                # self.plot_grad_flow(self.encoder[0].architecture.named_parameters())

                if log_this_iteration:
                    self.mean_loss += loss.item()
                    # mean_value_loss += value_loss.item()
                    # mean_surrogate_loss += 0
                    # self.loss_ROA += loss_ROA.item()
                    # self.lambda_loss_ROA += lambda_loss_ROA.item()
                    self.estimator_loss += estimator_loss.item()
                    # self.obj_f_dynamics_loss += obj_f_dynamics_loss.item()
                    # self.entropy_mean = (self.entropy_coef * entropy_batch.mean()).item()
                    # self.latent_f_dyn_loss += latent_f_dyn_loss.item()
                    # self.decoder_loss += decoder_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            self.mean_loss /= num_updates
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            self.loss_ROA /= num_updates
            self.lambda_loss_ROA /= num_updates
            self.estimator_loss /= num_updates
            # self.obj_f_dynamics_loss /= num_updates
            self.entropy_mean /= num_updates
            # self.obj_f_dynamics_loss /= num_updates
            # self.decoder_loss /= num_updates

        return self.mean_loss, locals()
