import gym
import numpy as np
from typing import Tuple


class EnvWrapper(gym.Env):
    
    def __init__(self, env_fct, attack_fct, freq=1):
        self.env = env_fct()
        self.attack_fct = attack_fct
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.freq = freq
        self.count = 0
    
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        if self.count % self.freq == 0:
            adv_obs = self.attack_fct(np.expand_dims(obs,axis=0)).detach().numpy().squeeze()
            self.env.attack_env(adv_obs)
        self.count += 1
        return obs, reward, terminal, info
    
    def reset(self):
        obs = self.env.reset()
        obs = np.expand_dims(obs,axis=0)
        adv_obs = self.attack_fct(obs).detach().numpy().squeeze()
        self.env.attack_env(adv_obs)
        self.count = 1
        return obs.squeeze()
    
    def render(self):
        return self.env.render()