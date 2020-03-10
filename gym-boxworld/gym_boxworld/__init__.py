from gym.envs.registration import register

register(
    id='boxworld-v0',
    entry_point='gym_boxworld.envs:BoxworldEnv',
)