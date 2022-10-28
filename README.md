# Environment Attack based on the Critic Network

This repository contains a PyTorch implementation of the method EACN (Environment Attack based on the Critic network) based on the article [Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network](https://arxiv.org/abs/2104.03154) (2022).

The implementation use the [StableBaslines3](https://github.com/DLR-RM/stable-baselines3) framework for training the RL agents. And the methods are applied on the [Highway-env](https://github.com/eleurent/highway-env) autonomous driving simulator environment.




## Installation

```sh
git clone git@github.com:EACN2022/EACN.git
cd EACN

conda create --name EACN python=3.9
conda activate EACN

pip install -r requirements.txt
```



## Environment Modifications

The environment requires the function `attack_env()` to use the attack EACN
- `attack_env()` takes as input an adversarial observation and modify the environment to create an adversarial state that match this observation, and it outputs the observation of this adversarial state.


Check the file [attacked_highway.py](https://github.com/EACN2022/EACN/blob/master/attacked_highway.py) to see how it is implemented in this highway environment.



## Usage Example

### EACN Adversarial Attack

To apply the attack EACN at each timestep in an environment :
```python
critic_attack_fct  = lamda obs: attack.critic_attack(obs, model, ...)
eacn_adv_env = env_wrapper.EnvWrapper(env_fct,critic_attack_fct)
#run the agent in the environment eacn_adv_env
```

Check the python script `train.py` see how it is applied in Highway-env to train an agent.



### References


* [Highway-env: An Environment for Autonomous Driving Decision-Making](https://github.com/eleurent/highway-env) (2018)
* [Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network](https://arxiv.org/abs/2104.03154) (2022)