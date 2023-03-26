from gym.envs.registration import register



register(
    id='my_first_env-v0',
    entry_point='loyal_wingmen.envs:my_first_env',
)
