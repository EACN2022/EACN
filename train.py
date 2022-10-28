import os
import json

import random
import numpy as np

import torch

import gym
import highway_env

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import env_wrapper
import attack
import attacked_highway






random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
sb3.common.utils.set_random_seed(0,using_cuda=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



with open("config.json", 'r') as fp:
    config = json.load(fp)



env_fct = lambda : gym.make("highway-v0", config=config)
train_env = SubprocVecEnv([env_fct for _ in range(4)])
train_env.seed(seed)
train_env.reset()

eval_env_fct = lambda : gym.make("highway-v0", config=config)
eval_env = eval_env_fct()
eval_env.seed(seed)
obs = eval_env.reset()




agent_name = "Agent_{}".format(seed)
if not os.path.isdir(agent_name):
    os.mkdir(agent_name)


    
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, dict(vf=[256], pi=[256])])
agent = PPO(MlpPolicy, train_env, batch_size=64, n_steps=1024, n_epochs=10, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1, target_kl=0.1, policy_kwargs=policy_kwargs,
            verbose=False, tensorboard_log="{}/tensorboard/".format(agent_name), device=device, seed=seed)

evalcallback = EvalCallback(eval_env=eval_env, n_eval_episodes=100, eval_freq=10000, deterministic=True, verbose=False)

agent.learn(total_timesteps=500000, reset_num_timesteps=False, callback=evalcallback)






def critic_network(obs):
    features = agent.policy.extract_features(obs)
    _, latent_vf = agent.policy.mlp_extractor(features)
    value = agent.policy.value_net(latent_vf)
    return value


search_space = torch.zeros(7,5)
for i in range(1,len(search_space)):
    search_space[i,1] = 1
    search_space[i,3] = 1
search_space = search_space.flatten()


attack_fct = lambda obs : attack.critic_attack(obs, critic_network, eps=eps, search_space=search_space)

train_adv_env_fct = lambda : env_wrapper.EnvWrapper(env_fct,attack_fct)

train_adv_env = SubprocVecEnv([train_adv_env_fct for _ in range(4)])
train_adv_env.seed(seed)
    
agent.set_env(train_adv_env)

evalcallback = EvalCallback(eval_env=eval_env, n_eval_episodes=100, eval_freq=10000, deterministic=True, verbose=False)


agent.learn(total_timesteps=500000, reset_num_timesteps=False, callback=evalcallback)



agent.save("{}/model".format(agent_name))