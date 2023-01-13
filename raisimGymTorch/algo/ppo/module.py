import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions import Categorical
from ..helper import Transformer

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture.forward(obs, actor=True).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.forward(obs, actor=True)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.forward(torch.from_numpy(obs).to(self.device), actor=True)

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        return 0

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.forward(obs, actor=False).detach()

    def evaluate(self, obs):
        return self.architecture.forward(obs, actor=False)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class Encoder:
    def __init__(self, architecture, device='cpu'):
        super(Encoder, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def evaluate(self, obs):
        return self.architecture(obs)

    def evaluate_update(self, obs):
        return self.architecture.forward_update(obs)



    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Encoder:
    def __init__(self, architecture, device='cpu'):
        super(Encoder, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def evaluate_update(self, obs):
        return self.architecture.forward_update(obs)



    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Estimator:
    def __init__(self, architecture, device='cpu'):
        super(Estimator, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def evaluate_update(self, obs):
        return self.architecture.forward_update(obs)



    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class Transformer_Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Transformer_Encoder).__init__()
        self.layers = Transformer.clones(layer, N)
        self.norm = Transformer.LayerNrom(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# class ROA(nn.Module):
#     def __init__(self, architecture, device='cpu'):
#         super(ROA, self).__init__()
#         self.architecture = architecture
#         self.architecture.to(device)
#
#     def predict(self, obs):
#         return self.architecture(obs).detach()
#
#     def evaluate(self, obs):
#         return self.architecture(obs)
#
#     def evaluate_update(self, obs):
#         return self.architecture.forward_update(obs)
#
#     def parameters(self):
#         return [*self.architecture.parameters()]
#
#     @property
#     def obs_shape(self):
#         return self.architecture.input_shape

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, ext_dim, pro_dim, dyn_dim, act_dim, hist_num, batch_num, num_env, device):
        super(LSTM, self).__init__()
        self.ext_dim = ext_dim
        self.pro_dim = pro_dim
        self.act_dim = act_dim
        self.dyn_dim = dyn_dim
        self.hist_num = hist_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_env = num_env
        self.batch_num = batch_num

        self.block_dim = ext_dim + pro_dim + dyn_dim + act_dim

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=False)

        # self.init_weights(self.architecture, scale)
        self.input_shape = [input_dim*batch_num]
        self.output_shape = [hidden_dim]
        self.h_0 = None
        self.c_0 = None
    def forward(self, obs):

        inputs = torch.reshape(obs, (self.num_env, -1, self.input_dim))
        inputs = torch.permute(inputs, (1, 0, 2))

        if (self.h_0 == None):
            outputs, (h_n, c_n) = self.lstm(inputs)
        else:
            outputs, (h_n, c_n) = self.lstm(inputs, (self.h_0, self.c_0))

        self.h_0 = h_n
        self.c_0 = c_n

        # print(outputs.shape)
        output = outputs[-1, :, :]

        return output

    def forward_update(self, obs):
        # ordered = self.obs_inorder(obs, update=True)
        # inputs = ordered.view(-1, self.num_env, self.input_dim)
        # print(obs[0,0,52:104]) # 40 300 260 (52 * 5)
        # inputs = obs.view(-1, self.num_env, self.input_dim)

        inputs = torch.permute(obs, (1,0,2)) # 40 300 5*a -> 300 40 5*a
        inputs = torch.reshape(inputs, (self.num_env, -1, self.input_dim)) # 300 200 a
        inputs = torch.permute(inputs, (1,0,2)) # 200 300 a



        # inputs = torch.reshape(obs, (200, self.num_env, -1)) # 200 300 52

        if (self.h_0 == None):
            outputs, (h_n, c_n) = self.lstm(inputs)
        else:
            outputs, (h_n, c_n) = self.lstm(inputs, (self.h_0, self.c_0))

        self.h_0 = h_n
        self.c_0 = c_n

        output = outputs.reshape(-1, self.hidden_dim)

        return output

    def reset(self):
        self.h_0 = None
        self.c_0 = None

class MLP_Prob(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP_Prob, self).__init__()
        self.activation_fn = actionvation_fn
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size*2))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        output = self.architecture(x)

        z_mu, z_logvar = torch.chunk(output, 2, dim=-1)

        # reparametrization
        _std = (torch.randn_like(z_mu)
                * torch.exp(0.5*z_logvar))

        z = z_mu + _std

        return z, z_mu, z_logvar


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, actor=False, discrete=False):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.discrete = discrete
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))

        # For low level controller
        # if actor:
        #     modules.append(nn.Sigmoid())

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, x, actor=False):
        output = self.architecture(x)

        if (actor == True):
            if(self.discrete == True):
                output = torch.softmax(output, dim=1)
            elif(self.discrete == False):
                norm = (torch.norm(output, dim=-1)+1e-8).unsqueeze(-1)
                output = torch.div(output, norm)
                output = output * 3 * torch.sigmoid(norm)

        return output

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class DiscreteDistribution(nn.Module):
    def __init__(self, dim, size, fast_sampler, seed = 0):
        super(DiscreteDistribution, self).__init__()
        self.dim = dim
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size,dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
    # def sample(self, logits):
    #     distribution = Categorical(probs=logits)
    #     sample = distribution.sample()
    #     return sample, distribution.log_prob(sample)
    def sample(self, logits):
        self.fast_sampler.sample(logits, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()
    def evaluate(self, logits, outputs):
        '''
        :param logits: Probability value that from network's output
        :param outputs: sampled label w.r.t probability
        :return:
        '''
        distribution = Categorical(probs=logits)
        actions_log_prob = distribution.log_prob(torch.argmax(outputs, dim=1))
        entropy = distribution.entropy()
        return actions_log_prob, entropy
    def entropy(self):
        return self.distribution.entropy()

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        '''
        :param logits: action mean
        :param outputs: sampled action from distribution
        :return: log probability that sampled action from distribution also entropy
        '''
        distribution = Normal(logits, self.std.reshape(self.dim))
        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
