import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_obs = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.obs = np.zeros([num_transitions_per_env, num_envs, *obs_shape], dtype=np.float32)
        self.rewards = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones = np.zeros([num_transitions_per_env, num_envs, 1], dtype=bool)
        self.contacts = np.zeros([num_transitions_per_env, num_envs, 1], dtype=bool)
        self.privileged_infos = np.zeros([num_transitions_per_env, num_envs, 22], dtype=np.float32)
        self.anchors = np.zeros([num_transitions_per_env, num_envs, 24*20], dtype=np.float32)

        # For PPO
        self.actions_log_prob = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)

        # torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        self.contacts_tc = torch.from_numpy(self.contacts).to(self.device)
        self.privileged_info_tc = torch.from_numpy(self.privileged_infos).to(self.device)
        self.anchors_tc = torch.from_numpy(self.anchors).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, actor_obs, critic_obs, obs, actions, mu, sigma, rewards, dones, actions_log_prob, contact = False, privileged_info=None, anchors=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step] = critic_obs
        self.obs[self.step] = obs
        self.actor_obs[self.step] = actor_obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        # self.rewards[self.step] = rewards.reshape(-1, 1)
        # self.dones[self.step] = dones.reshape(-1, 1)
        # self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        # self.contacts[self.step] = contact.reshape(-1, 1)
        self.privileged_infos[self.step] = privileged_info
        self.anchors[self.step] = anchors
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self):
        # with torch.inference_mode():
        #     self.values = critic.predict(torch.from_numpy(self.critic_obs).to(self.device)).cpu().numpy()
        #
        # advantage = 0
        #
        # for step in reversed(range(self.num_transitions_per_env)):
        #     if step == self.num_transitions_per_env - 1:
        #         next_values = last_values.cpu().numpy()
        #         # next_is_not_terminal = 1.0 - self.dones[step].float()
        #     else:
        #         next_values = self.values[step + 1]
        #         # next_is_not_terminal = 1.0 - self.dones[step+1].float()
        #
        #     next_is_not_terminal = 1.0 - self.dones[step]
        #     delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        #     advantage = delta + next_is_not_terminal * gamma * lam * advantage
        #     self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Convert to torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.obs_tc = torch.from_numpy(self.obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.contacts_tc = torch.from_numpy(self.contacts).to(self.device)
        self.privileged_info_tc = torch.from_numpy(self.privileged_infos).to(self.device)
        self.anchors_tc = torch.from_numpy(self.anchors).to(self.device)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):

            actor_obs_batch = torch.reshape(self.actor_obs_tc, (-1, *self.actor_obs_tc.size()[2:]))[indices]
            critic_obs_batch = torch.reshape(self.critic_obs_tc, (-1, *self.critic_obs_tc.size()[2:]))[indices]
            obs_batch = torch.reshape(self.obs_tc, (-1, *self.obs.size()[2:]))[indices]
            actions_batch = torch.reshape(self.actions_tc, (-1, self.actions_tc.size(-1)))[indices]
            sigma_batch = torch.reshape(self.sigma_tc, (-1, self.sigma_tc.size(-1)))[indices]
            mu_batch = torch.reshape(self.mu_tc, (-1, self.mu_tc.size(-1)))[indices]
            values_batch = torch.reshape(self.values_tc, (-1, 1))[indices]
            returns_batch = torch.reshape(self.returns_tc, (-1, 1))[indices]
            old_actions_log_prob_batch = torch.reshape(self.actions_log_prob_tc, (-1, 1))[indices]
            advantages_batch = torch.reshape(self.advantages_tc, (-1, 1))[indices]

            # actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
            # critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
            # obs_batch = self.obs_tc.view(-1, *self.obs.size()[2:])[indices]
            # actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
            # sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
            # mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
            # values_batch = self.values_tc.view(-1, 1)[indices]
            # returns_batch = self.returns_tc.view(-1, 1)[indices]
            # old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
            # advantages_batch = self.advantages_tc.view(-1, 1)[indices]
            yield actor_obs_batch, critic_obs_batch, obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        for batch_id in range(num_mini_batches):

            obs_batch = self.obs_tc[:, batch_id*(self.num_envs // num_mini_batches):(batch_id+1)*(self.num_envs // num_mini_batches), :]
            action_batch = torch.reshape(self.actions_tc, (-1, self.actions_tc.size(-1)))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            sigma_batch = torch.reshape(self.sigma_tc, (-1, self.sigma_tc.size(-1)))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            mu_batch = torch.reshape(self.mu_tc, (-1, self.mu_tc.size(-1)))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            value_batch = torch.reshape(self.values_tc, (-1, 1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            advantage_batch = torch.reshape(self.advantages_tc, (-1, 1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            return_batch = torch.reshape(self.returns_tc, (-1, 1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            action_log_prob_batch = torch.reshape(self.actions_log_prob_tc, (-1, 1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            contact_batch = torch.reshape(self.contacts_tc, (-1,1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            privileged_batch = torch.reshape(self.privileged_info_tc, (-1, self.privileged_info_tc.size(-1)))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
            anchors_batch = torch.reshape(self.anchors_tc, (-1, self.anchors_tc.size(-1)))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]

            yield obs_batch, action_batch, sigma_batch, mu_batch, value_batch, advantage_batch, return_batch, action_log_prob_batch, contact_batch, privileged_batch, anchors_batch

            # yield self.obs_tc[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.actions_tc.view(-1, self.actions_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.sigma_tc.view(-1, self.sigma_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.mu_tc.view(-1, self.mu_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.values_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.advantages_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.returns_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
            #     self.actions_log_prob_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]


