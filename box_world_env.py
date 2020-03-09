import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box

import numpy as np
import matplotlib.pyplot as plt

from boxworld_gen import *

class Boxworld(gym.Env):
    """Boxworld representation
    Args:
      n: size of the board (n x n) excluding the edge
      goal_length: length of correct solution
      num_distractor: number of distractor branches
      distractor_length: length of each distractor path (currently all distractor paths are same length
      max_steps: maximum steps the environment allows before terminating
      world: an existing world data. If this is given, use this data.
             If None, generate a new data by calling world_gen() function
    """

    def __init__(self, n, goal_length, num_distractor, distractor_length, max_steps=5000, world=None):
        """
           Args:
             n: size of the field (n x n) without the outline
             goal_length
             num_distractor
             distractor_length
             world: an existing world data. If this is given, use this data.
                    If None, generate a new data by calling world_gen() function
           """
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor

        # Penalties and Rewards
        self.step_cost = 0.0# 0.1 #todo: remove or not?
        self.reward_gem = 10
        self.reward_dead = -1
        self.reward_correct_key = 1
        self.reward_key = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(low=0, high=255, shape=(n, n, 3), dtype=np.uint8)

        # Game initialization
        self.owned_key = [220, 220, 220]

        self.np_random_seed = None
        self.reset(world)

    def seed(self, seed=None):
        self.np_random_seed = seed
        return [seed]

    def save(self):
        np.save('box_world.npy', self.world)

    def step(self, action):

        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        self.num_env_steps += 1

        reward = -self.step_cost
        done = self.num_env_steps == self.max_steps

        # Move player if the field in the moving direction is either
        print(self.player_position)
        print(new_position)
        print(self.n)
        if np.any(new_position < 1) or np.any(new_position >= self.n+1): #at boundary
            possible_move = False

        # elif np.array_equal(new_position, [0, 0]):
        #     possible_move = False

        elif is_empty(self.world[new_position[0], new_position[1]]):
            # No key, no lock
            possible_move = True

        elif new_position[1] == 0 or is_empty(self.world[new_position[0], new_position[1]-1]):
            # It is a key
            if is_empty(self.world[new_position[0], new_position[1]+1]):
                # Key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                self.world[0, 0] = self.owned_key
                # print(self.owned_key)
                # print(goal_color)
                # print(self.goal_colors)
                # print(self.owned_key in self.dead_ends)
                # print(self.owned_key in self.goal_colors)
                if np.array_equal(self.owned_key, np.array(goal_color)):
                    # Goal reached
                    reward += self.reward_gem
                    done = True
                elif np.any([np.array_equal(self.owned_key,dead_end) for dead_end in self.dead_ends]): #reached a dead
                    # end,
                # terminate episode
                    reward += self.reward_dead
                    done = True
                elif np.any([np.array_equal(self.owned_key, key) for key in self.correct_keys]):
                    reward += self.reward_correct_key
                else:
                    reward += self.reward_key
            else:
                possible_move = False
        else:
            # It is a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # The lock matches the key
                possible_move = True
            else:
                possible_move = False
                print("lock color is {}, but owned key is {}".format(
                    self.world[new_position[0], new_position[1]], self.owned_key))

        if possible_move:
            self.player_position = new_position
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
        }

        return self.world, reward, done, info

    def reset(self, world=None):
        if world is None:
           self.world, self.player_position, self.dead_ends, self.correct_keys = world_gen(n=self.n,
                                                                                  goal_length=self.goal_length,
                                                         num_distractor=self.num_distractor,
                                                         distractor_length=self.distractor_length,
                                                        seed=self.np_random_seed)
        else:
            self.world, self.player_position, self.dead_ends, self.correct_keys = world

        self.num_env_steps = 0

        return self.world

    def render(self, mode="human", figAx=None):
        img = self.world.astype(np.uint8)
        if mode == "return":
            return img

        else:
            if (figAx) == None:
                fig,ax = plt.subplots()
            else:
                fig,ax = figAx
            # from gym.envs.classic_control import rendering
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
            # return self.viewer.isopen
            ax.imshow(img, vmin=0, vmax=255, interpolation='none')
            fig.show()
            return (fig,ax)

    def get_action_lookup(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


# if __name__ == "__main__":
#     # execute only if run as a script
#     env = Boxworld(12, 4, 2, 2)
#     # env.seed(1)
#     env.reset()
#     env.render()
