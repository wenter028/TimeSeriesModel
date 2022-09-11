# This code is ddpg model *can use the competition edition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
import numpy as np

__author__ = 'Wenter'

__all__ = ['hard_copy',
           'soft_copy',
           'Actor',
           'Critic',
           'ActorNet',
           'CriticNet',
           'DDPG']


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
:论文地址: https://arxiv.org/pdf/1811.07522.pdf
"""


def hard_copy(target,
              source):
    """
    :func:to initialize ddpg network
    :param target: target net
    :param source: source net(copied)
    """
    for weight1, weight2 in zip(target, source):
        weight1.data = weight2.data.clone()


def soft_copy(target, source, w=0.1):
    """
    :func:to update ddpg network
    :param target: target net
    :param source: source net(copied)
    :param w: learning rate
    """
    for weight1, weight2 in zip(target, source):
        weight1.data = torch.add(
            weight1.data, torch.add(
                weight2.data, weight1.data, alpha=-1
            ), alpha=w)


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds = 0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds += 1
        np.random.seed(self.seeds)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def std_noise(self, mu, sigma):
        self.seeds += 1
        np.random.seed(self.seeds)
        return np.random.normal(mu, sigma, len(self.state))


class ActorNet(nn.Module):
    """
    :class: Actor net to give the action
    """
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: int):
        """
        :func:init
              GRU to handle time series data
              fc to combine
        :param state_dim: observation dim
        :param hidden_dim: fc hidden dim
        :param output_dim: action dim
        :param dropout: dropout rate
        """
        super(ActorNet, self).__init__()
        self.gru = nn.GRU(state_dim[0], hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, state_dim[1])

        self.fc1 = nn.Linear(2 * state_dim[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        :func: actor net process
        :param x: features
        :param y: account
        * account need to be concatenated in the features
        """
        _, x = self.gru(x)
        x = self.fc(x[1])
        x = torch.cat((x, y), 1)
        x = self.fc1(x)
        x = self.fc2(F.tanh(x))
        x = self.dropout(x)
        x = self.fc3(F.tanh(x))

        x = F.softmax(x)

        return x


class CriticNet(nn.Module):
    """
    :class: Critic net for q value of state and action
    """
    def __init__(self, state_dim, up_dim, down_dim, hidden_dim, out_dim, dropout):
        """
        :func: init
               GRU to handle the timeseries data
               fc to combine state and action
        :param state_dim: observation dim( feature and account)
        :param up_dim: action dim
        :param down_dim: state dim
        :param hidden_dim: fc hidden dim
        :param out_dim: q function dim
        :param dropout: dropout rate
        """
        super(CriticNet, self).__init__()
        self.gru = nn.GRU(state_dim[0], hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, down_dim - state_dim[1])

        self.fc_up = nn.Linear(up_dim, hidden_dim)
        self.fc_down = nn.Linear(down_dim + hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, a):
        """
        :func: critic process
               handle features first and then combine fc(action) output
        :param s: state(features and account)
        :param a: action
        """
        _, s1 = self.gru(s[0])
        s1 = self.fc(s1[1])
        s = torch.cat((s1, s[1]), 1)

        a = self.fc_up(a)
        sa = torch.cat((a, s), 1)

        sa = F.tanh(self.fc_down(sa))
        sa = self.dropout(sa)

        out = self.fc1(sa)

        return out


class Actor:
    """
    :class: whole actor network(initialize, soft update etc.)
    """
    def __init__(self, state_dim, hidden_dim, action_dim, dropout):
        """
        :func: init
               1 actor and 1 target in essay
               same initialize for 2 network
               adam optimizer
        :param state_dim: observation dim( feature and account)
        :param hidden_dim: fc hidden dim
        :param action_dim: .
        :param dropout: dropout rate
        """
        self.actor = ActorNet(state_dim, hidden_dim, action_dim, dropout).to(device)
        self.target = ActorNet(state_dim, hidden_dim, action_dim, dropout).to(device)
        self.actor_weights = [params for params in self.actor.parameters()]
        self.target_weights = [params for params in self.target.parameters()]
        self.optimizer = optim.Adam(self.actor.parameters())
        hard_copy(self.target_weights, self.actor_weights)

        self.prob = None
        self.exploration_noise = OUNoise(action_dim, sigma=0.1 / action_dim)

    def take_action_source(self, states):
        """
        :func: take action by sample the softmax (source)
        :param states: .
        :return: action
        *add noise
        """
        self.prob = self.actor(states[0], states[1])
        noise = self.prob.detach()
        #+ abs(self.exploration_noise.noise())  cuda?
        noise = Categorical(noise)
        self.prob = Categorical(self.prob)

        action = noise.sample()

        return action

    def take_action_target(self, states):
        """
        :func: take action by sample the softmax (target)
        :param states: .
        :return: action
        """
        self.target.zero_grad()
        prob = self.target(states[0], states[1])
        prob = Categorical(prob)
        action = prob.sample()

        return action

    def train(self, action, td_delta):
        """
        :func: train the model
               use mean(td_delta*(-probability of action) as loss
               * may use -q function
        :param action:.
        :param td_delta:.
        * may use grad to backward instead
        """
        self.optimizer.zero_grad()
        loss = torch.mean(-self.prob.log_prob(action) * td_delta.detach())
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """
        :func: soft update network in essay
        """
        soft_copy(self.target_weights, self.actor_weights)


class Critic:
    """
    :class: whole Critic network(initialize soft update etc.)
    """
    def __init__(self, state_dim, up_dim, down_dim, hidden_dim, out_dim, dropout):
        """
        :func: init
               1 actor and 1 target in essay
               same initialize for 2 network
               adam optimizer
        :param state_dim: observation dim( feature and account)
        :param up_dim: action dim
        :param down_dim: state dim
        :param hidden_dim: fc hidden dim
        :param out_dim: q function dim
        :param dropout: dropout rate
        """
        self.critic = CriticNet(state_dim, up_dim, down_dim, hidden_dim, out_dim, dropout).to(device)
        self.target = CriticNet(state_dim, up_dim, down_dim, hidden_dim, out_dim, dropout).to(device)
        self.critic_weights = [params for params in self.critic.parameters()]
        self.target_weights = [params for params in self.target.parameters()]
        self.optimizer = torch.optim.Adam(self.critic.parameters())
        hard_copy(self.target_weights, self.critic_weights)

    def source_q(self, state, action):
        """
        :func: q function (source)
        :param state:.
        :param action:.
        """
        return self.critic(state, action)

    def target_q(self, state, action):
        """
        :func: q function (target)
        :param state:.
        :param action:.
        """
        self.target.zero_grad()

        return self.target(state, action)

    def train(self, state, state_next, action, reward):
        """
        :func: train the model
               use mse(q function - (next q function + reward))
        :param state:.
        :param state_next:.
        :param action:.
        :param reward:.
        :return: td_delta
        * may use grad to backward instead
        """
        criterion = torch.nn.MSELoss()
        y = self.target_q(state_next, action) + reward
        q = self.source_q(state, action)

        loss = criterion(y, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return y - q

    def update_target(self):
        """
        :func: soft update in essay
        """
        soft_copy(self.target_weights, self.critic_weights)


class DDPG:
    """
    :class: Whole ddpg network
    *may not use the GRU as it may get nan loss
    """
    def __init__(self, state_dim, up_dim, down_dim, hidden_dim, out_dim, action_dim, dropout):
        """
        :func: init
               actor and critic
               replay buffer for quick and high efficient train
        :param state_dim: .
        :param up_dim: action dim
        :param down_dim: state GRU output dim
        :param hidden_dim: fc hidden dim
        :param out_dim: q function dim
        :param action_dim:.
        :param dropout: dropout rate
        """

        # Net
        self.actor = Actor(state_dim, hidden_dim, action_dim, dropout)
        self.critic = Critic(state_dim, up_dim, down_dim, hidden_dim, out_dim, dropout)

        # Replay Buffer
        self.step = 0
        self.buffer_max = 1000
        self.rb_state = [torch.zeros(self.buffer_max, 5, 13, device=device), torch.zeros(self.buffer_max, 1,device=device)]
        self.rb_state_next = [torch.zeros(self.buffer_max, 5, 13, device=device), torch.zeros(self.buffer_max, 1, device=device)]
        self.rb_action = torch.zeros(self.buffer_max, device=device)
        self.rb_reward = torch.zeros(self.buffer_max ,device=device )

        self.batch_size = 100

    def take_action(self, states):
        """
        :func: actor take action
        """

        return self.actor.take_action_source(states)

    def perceive(self, state, state_next, action, reward):
        """
        :func: save the data into replay buffer
        :param state: .
        :param state_next: .
        :param action: .
        :param reward: .
        * should concatenate the features and account together
        """

        if self.step < self.buffer_max:
            self.rb_state[0][self.step] = state[0]
            self.rb_state[1][self.step] = state[1]
            self.rb_state_next[0][self.step] = state_next[0]
            self.rb_state_next[1][self.step] = state_next[1]
            self.rb_action[self.step] = action
            self.rb_reward[self.step] = reward

        else:
            self.rb_state[0] = torch.cat((self.rb_state[0][1:], state[0].unsqueeze(0)), 0)
            self.rb_state[1] = torch.cat((self.rb_state[1][1:], state[1].unsqueeze(0)), 0)
            self.rb_state_next[0] = torch.cat((self.rb_state_next[0][1:], state_next[0].unsqueeze(0)), 0)
            self.rb_state_next[1] = torch.cat((self.rb_state_next[1][1:], state_next[1].unsqueeze(0)), 0)
            self.rb_action = torch.cat((self.rb_action[1:], action.unsqueeze(0)), 0)
            self.rb_reward = torch.cat((self.rb_reward[1:], reward.unsqueeze(0)), 0)

        self.step += 1

    def train(self):
        """
        :func: train the model
               if replay buffer is not enough to random pick then get whole replay buffer
               else random choose batch to train
        # pay attention to the nan loss
        """
        sample = torch.randint(0, self.buffer_max, [self.batch_size],device=device)

        state_batch = [torch.index_select(self.rb_state[0], 0, sample), torch.index_select(self.rb_state[1], 0, sample)]
        state_next_batch = [torch.index_select(self.rb_state_next[0], 0, sample),
                            torch.index_select(self.rb_state_next[1], 0, sample)]
        action_batch = torch.index_select(self.rb_action, 0, sample).unsqueeze(1)
        reward_batch = torch.index_select(self.rb_reward, 0, sample).unsqueeze(1)

        # Critic update
        td_delta = self.critic.train(state_batch, state_next_batch, action_batch, reward_batch)
        self.critic.update_target()
        # Actor update
        self.actor.train(action_batch, td_delta)
        self.actor.update_target()






