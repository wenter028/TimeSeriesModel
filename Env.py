# This code is basic environment for reinforcement learning
import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
import torch

__author__ = 'Wenter'

__all__ = ['BasisEnv',]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BasisEnv(gym.Env):
    """
    :class: basic env for trading
    * can be replaced by competition edition
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data: torch.tensor,
                 time_before: int):
        """
        :func:init
              for each batch create a new env
              action get the profit
              time_before * 13features
              account can't bankrupt
              action (0,6)->(sell3,buy3)
        :param data: whole batch data
        :param time_before: similar time interval how much time to see before trade

        """
        super(BasisEnv, self).__init__()

        self.data = data[0].unfold(1, 5, 1).transpose(2, 3).contiguous()
        self.batch = data[0].shape[0]
        self.len = data[0].shape[1] - time_before

        self.profit = data[1]

        self.time_before = time_before

        # Time index
        self.current_step = None

        # account
        self.account = None
        self.done = None

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.intc)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, 13), dtype=np.float16)

    def _next_observation(self):
        """
        :func: next time index observation
        """
        data = self.data[:, self.current_step]

        return data

    def _take_action(self, action):
        """
        :func: take action to get reward
        :param action: take action
        """

        reward = action * self.profit[:, self.current_step].squeeze()

        self.account += reward.unsqueeze(1)

        return reward

    def step(self, action):
        """
        :func: take action get reward and next observation
               account can't bankrupt(can't <0)
        :param action: take action
        """

        reward = self._take_action(action - 3)

        self.current_step += 1

        if (self.current_step + 1) == self.len:
            self.done = True

        # remaining balance nonnegative
        if any(self.account < 0):
            self.done = True
            reward = -1000

        return [self._next_observation(), self.account / 100000], reward, self.done

    def reset(self):
        """
        :func: reset the env to the first time index
               current_step 0
               done False
               account 100000
        """
        self.current_step = 0  # random.randint(0,self.df.shape[0]-1)
        self.done = False

        # account
        self.account = torch.full((self.data.shape[0], 1), 100000, dtype=torch.float,device=device)

        return self._next_observation(), self.account / 100000

    def render(self):
        pass
