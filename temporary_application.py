import gym
import callback_factory
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# #It is included if you install stable-baselines with the extra packages: `pip install stable-baselines3[extra]`

# Parallel environments
env = make_vec_env("LunarLander-v2", n_envs=4, monitor_dir="logs")
# https://stable-baselines3.readthedocs.io/en/master/common/env_util.html


callbacklist = callback_factory.callbacklist(env, n_envs=4)


model = PPO("MlpPolicy", env)


model.learn(100_000, callback=callbacklist)
#model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("models/best_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()