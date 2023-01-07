from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage_encoding_discrete import RolloutStorage
import matplotlib.pyplot as plt


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 encoder,
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

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.encoder_input_dim = self.encoder.architecture.input_shape[0]
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters(), *self.estimator.parameters()], lr=learning_rate)
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

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.inference_mode():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions

    def step(self, value_obs, obs, rews, dones):

        """
        remove actor_obs, value_obs -> into encoder_obs
        """
        self.storage.add_transitions(self.actor_obs, value_obs, obs, self.actions, self.actor.action_mean, rews, dones,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):

        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(log_this_iteration)
        self.mean_value_loss = infos['mean_value_loss']
        self.mean_surrogate_loss = infos['mean_surrogate_loss']
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs

        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])

    def encode(self, obs):

        output = self.encoder.evaluate_update(obs)

        return output

    def encode_ROA(self, obs):
        output = self.encoder_ROA.evaluate_update(obs)

        return output
    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grad = p.grad.clone().detach().to('cpu')
                ave_grads.append(ave_grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig('gradient.png')

    def get_obs_ROA(self, obs_batch):
        obs_ROA_batch = []


        for i in range(self.num_history_batch):
            # Get proprioceptive part of observation
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i:
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i
                                                + self.encoder.architecture.pro_dim])

            # Get Exteroceptive part of observation except inertial parameter
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i
                                 + self.encoder.architecture.pro_dim:
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i
                                 + self.encoder.architecture.pro_dim
                                 + self.encoder.architecture.ext_dim - self.inertial_dim])

            # Get action part of observation
            obs_ROA_batch.append(obs_batch[...,
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i
                                 + self.encoder.architecture.pro_dim+self.encoder.architecture.ext_dim:
                                 (self.encoder.architecture.pro_dim + self.encoder.architecture.ext_dim + self.encoder.architecture.act_dim)*i
                                 + self.encoder.architecture.pro_dim+self.encoder.architecture.ext_dim+self.encoder.architecture.act_dim])

        # Distillate the true inertial parameter (oracle)
        estimator_true_data = (obs_batch[...,
                                   (self.encoder.architecture.pro_dim +
                                    self.encoder.architecture.ext_dim +
                                    self.encoder.architecture.act_dim)*(self.num_history_batch-1)
                                   + self.encoder.architecture.pro_dim
                                   + self.encoder.architecture.ext_dim - self.inertial_dim:
                                   (self.encoder.architecture.pro_dim +
                                    self.encoder.architecture.ext_dim +
                                    self.encoder.architecture.act_dim)*(self.num_history_batch-1)
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

        self.lambda_ROA = pow(self.lambda_ROA, self.lambdaDecayFactor)
        print(self.lambda_ROA)
        for epoch in range(self.num_learning_epochs):
            for obs_batch, actions_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                self.encoder.architecture.reset()
                self.encoder_ROA.architecture.reset()

                obs_ROA_batch, estimator_true_data = self.get_obs_ROA(obs_batch)

                obs_concat = self.encode(obs_batch)
                obs_concat_d = obs_concat.clone().detach()
                obs_concat_ROA = self.encode_ROA(obs_ROA_batch)
                obs_concat_ROA_d = obs_concat_ROA.clone().detach()

                estimator_input = self.encoder_ROA.evaluate_update(obs_ROA_batch).clone().detach().reshape(-1, self.encoder_ROA.architecture.hidden_dim)

                estimator_loss = self.criteria(self.estimator.evaluate(estimator_input), estimator_true_data)

                lambda_loss_ROA = self.lambda_ROA * self.criteria(obs_concat, obs_concat_ROA_d)
                loss_ROA = self.criteria(obs_concat_d, obs_concat_ROA)



                actions_log_prob_batch, entropy_batch = self.actor.evaluate(obs_concat, actions_batch)
                value_batch = self.critic.evaluate(obs_concat)

                # Adjusting the learning rate using KL divergence
                mu_batch = self.actor.action_mean

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        # kl = torch.sum(
                        #     torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)

                        kl = torch.sum(torch.matmul(torch.transpose(old_mu_batch, 0, 1), torch.log(old_mu_batch/mu_batch)), axis = -1)

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
                       + estimator_loss


                # Add kl divergence term to normalize the latent vector

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters(), *self.estimator.parameters()], self.max_grad_norm)

                # remove estimator
                # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), *self.encoder.parameters(), *self.encoder_ROA.parameters()], self.max_grad_norm)



                self.optimizer.step()

                # self.plot_grad_flow(self.encoder[0].architecture.named_parameters())

                if log_this_iteration:
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    self.loss_ROA += loss_ROA.item()
                    self.lambda_loss_ROA += lambda_loss_ROA.item()
                    self.estimator_loss += estimator_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            self.loss_ROA /= num_updates
            self.lambda_loss_ROA /= num_updates
            self.estimator_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
