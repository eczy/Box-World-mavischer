from gym.envs.registration import register

register(
    id='boxworld-v0',
    entry_point='gym_boxworld.envs:BoxworldEnv',
)

register(
    id='boxworldMini-v0', #
    entry_point='gym_boxworld.envs:BoxworldEnv',
    kwargs={
        "n": 5,  # size of board
        "goal_length": 3,  # length of correct path (e.g. 4 means goal can be unlocked with 3rd key)
        "num_distractor": 1,  # number of distractor branches, can be list
        "distractor_length": 1,  # length/"depth" of each distractor branch, can be list
        "num_colors": 8,
        "max_steps": 250
    }
)

register(
    id='boxworldNano-v0', #most simple task imaginable: only 1 color, no distractor, negative step cost as incentive
    entry_point='gym_boxworld.envs:BoxworldEnv',
    kwargs={
        "n": 5,  # size of board
        "goal_length": 2,  # meaning there is one intermediary key before the goal
        "num_distractor": 0,  # number of distractor branches, can be list
        "num_colors": 1,
        "max_steps": 50,
        "step_cost": -0.1
    }
)