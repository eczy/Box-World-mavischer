#Adaptation of https://github.com/nathangrinsztajn/Box-World/blob/master/box_world_env.py that used the gym
# ecosystem's distribution mechanism. Install it by calling pip install -e gym-boxworld inside gym-boxworld and
# instantiate an environment by calling env = gym.make('gym-boxworld:boxworld-v0') or 'boxworldMini-v0' for a smaller
# version.

import gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

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

#this version does not use black as a key color because it's used for the board's outline
COLORS = {0: [0, 0, 117],  1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200], 4: [255, 216, 177],
          5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216], 9: [66, 212, 244],
          10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49], 14: [230, 25, 75],
          15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144], 19: [100, 70, 0]}

for key in COLORS.keys():
    COLORS[key] = np.array(COLORS[key], dtype=np.uint8)
AGENT_COLOR = np.array([128, 128, 128], dtype=np.uint8)
GOAL_COLOR = np.array([255, 255, 255], dtype=np.uint8)
BACKGD_COLOR = np.array([220, 220, 220], dtype=np.uint8)

class BoxworldEnv(gym.Env):
    """Boxworld as gym environment.

    So far, the rewards are hard-coded in the environment, not parameterized in the init.
    Args:
        n: size of the board (n x n) excluding the outline (width 1 around board, so in total (n+2 x n+2)
        goal_length: length of correct solution path, i.e. #keys that have to be picked up in the correct sequence
        num_distractor: number of distractor branches
        distractor_length: length of each distractor path (currently all distractor paths are same length)
        max_steps: maximum steps the environment allows before terminating
        world: an existing world data. If this is given, use this data.
             If None, generate a new data by calling world_gen() function

    """
    metadata = {'render.modes': ['human','return']}

    def __init__(self, n=12, goal_length=5, num_distractor=2, distractor_length=2, #solution configuration
                 step_cost=0, reward_gem=10, reward_dead=0, reward_correct_key=1, reward_wrong_key=-1, #reward structure
                 num_colors=20, max_steps=3000, world=None, verbose=False):
        """
        Args:
            STATE SPACE
            n: size of the board (n x n) excluding the outline (width 1 around board, so in total (n+2 x n+2)
            GOAL PATH
            goal_length: length of correct solution path, i.e. #keys that have to be picked up in the correct
            sequence, _including the gem_ at the end
            num_distractor: number of distractor branches
            distractor_length: length of each distractor path (currently all distractor paths are same length)
            REWARDS
            step_cost: base cost of performing a step, should be negative
            reward_gem: reward for collecting the gem (game terminates afterwards)
            reward_dead: reward for reading a dead end (game terminates afterwards), should be negative
            reward_correct_key: reward for collecting (not just unlocking) a key on the solution path (correct key)
            reward_wrong_key: reward for collecting (not just unlocking) a key not on the solution path (incorrect key)
            ENVIRONMENT
            max_steps: maximum steps the environment allows before terminating
            world: an existing world data. If this is given, use this data.
                If None, generate a new data by calling world_gen() function
            verbose: print messages when creating and interacting with the environment
        """
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor

        # Penalties and Rewards
        self.step_cost = step_cost #assumed to be negative, like e.g. -0.1
        self.reward_gem = reward_gem
        self.reward_dead = reward_dead
        self.reward_correct_key = reward_correct_key
        self.reward_wrong_key = reward_wrong_key

        # Create custom color set for smaller state spaces
        if num_colors < goal_length - 1 + distractor_length * num_distractor or num_colors > 20:
            raise ValueError(f"Unacceptable number of possible colors: {num_colors}")
        self.num_colors = num_colors
        self.colors = {}
        for i in range(self.num_colors):
            self.colors[i] = COLORS[i]

        # Other Settings
        self.viewer = None
        self.verbose = verbose
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(low=0, high=255, shape=(n+2, n+2, 3), dtype=np.uint8)

        # Game initialization
        self.owned_key = [0, 0, 0]

        self.np_random_seed = None
        self.reset(world)

    def seed(self, seed=None):
        self.np_random_seed = seed
        return [seed]

    def plot_solution_graph(self, goal_colors, distractor_colors, distractor_roots, colors):
        """Plots game problem as directed graph of colors that were used to render a given environment.
        Not very pretty yet.
        """
        vis = np.ones([len(distractor_roots) + 1,
                       max(len(goal_colors),
                           max([distractor_roots[path] + len(distractor_colors[path]) for path in range(len(
                               distractor_roots))])) + 1, 3], dtype=int) * BACKGD_COLOR[0]  # length of longest path
        for i, col in enumerate(goal_colors):
            vis[0, i, :] = colors[col]
        vis[0, len(goal_colors), :] = GOAL_COLOR
        for j, dist_branch in enumerate(distractor_colors):
            for i_raw, dist_col in enumerate(dist_branch):
                i = i_raw + distractor_roots[j] + 1
                vis[j + 1, i, :] = colors[dist_col]
        plt.title("problem graph")
        plt.imshow(vis)
        plt.yticks(ticks=list(range(len(distractor_roots) + 1)),
                   labels=["solution path"] + [f"distractor path {i + 1}" for i in range(len(distractor_roots))])
        plt.xticks(list(range(len(goal_colors) + 1)))
        plt.xlabel("key #")

    def sample_pair_locations(self, num_pair):
        """Generates random key,lock pairs locations in the environment.

        Makes sure the objects don't collide and everything is reachable by the agent.
        The locations can be filled later on with filled with correct or distractor colors.
        First key is returned separately because it comes without a neighbor.

        Args:
            num_pair: number of key-lock pairs to be generated, results from length of correct path + length of all
            distractor branches.
        Returns:
            x,y-coordinates of all keys, locks, first key and agent.
        """
        n = self.n #size of the board excluding boundary
        possibilities = set(range(1, n * (n - 1)))
        keys = []
        locks = []
        for k in range(num_pair):
            key = random.sample(possibilities, 1)[0]
            key_x, key_y = key // (n - 1), key % (n - 1)
            lock_x, lock_y = key_x, key_y + 1
            to_remove = [key_x * (n - 1) + key_y] + \
                        [key_x * (n - 1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] + \
                        [key_x * (n - 1) - i + key_y for i in range(1, min(2, key_y) + 1)]

            possibilities -= set(to_remove)
            keys.append([key_x, key_y])
            locks.append([lock_x, lock_y])
        agent_pos = random.sample(possibilities, 1)
        possibilities -= set(agent_pos)
        first_key = random.sample(possibilities, 1)

        agent_pos = np.array([agent_pos[0] // (n - 1), agent_pos[0] % (n - 1)])
        first_key = first_key[0] // (n - 1), first_key[0] % (n - 1)
        return keys, locks, first_key, agent_pos

    def world_gen(self, seed=None, plot_solution=False):
        """Wrapper to generate boxworld if it is not loaded from array.

        Originally written to be not a global function, so some handling of the variables is slightly suboptimal but
        after all it gets only called once to create the environment."""
        if seed is None:
            random.seed(seed)

        world_dic = {}
        # background
        world = np.ones((self.n, self.n, 3), dtype=np.uint8) * BACKGD_COLOR

        # pick colors for intermediate goals and distractors
        goal_colors = random.sample(range(self.num_colors), self.goal_length - 1)
        distractor_possible_colors = [color for color in range(len(self.colors)) if color not in goal_colors]
        distractor_colors = [random.sample(distractor_possible_colors, self.distractor_length) for k in
                             range(self.num_distractor)]

        # sample where to branch off distractor branches from goal path
        # this line mainly prevents arbitrary distractor path length
        distractor_roots = random.choices(range(self.goal_length - 1), k=self.num_distractor)

        # find legal positions for all pairs
        keys, locks, first_key, agent_pos = \
            self.sample_pair_locations(self.goal_length - 1 + self.distractor_length* self.num_distractor)

        # rudimentary plot of solution DAG
        if plot_solution:
            self.plot_solution_graph(goal_colors, distractor_colors, distractor_roots, self.colors)

        dead_ends = [] #this needs to be initialized outside of if clause

        if self.goal_length == 1:  # special case where there is only an orphaned goal
            world[first_key[0], first_key[1]] = GOAL_COLOR

        else:         # create the goal path
            for i in range(1, self.goal_length):
                if i == self.goal_length - 1:
                    color = GOAL_COLOR  # final key is white
                else:
                    color = self.colors[goal_colors[i]]
                if self.verbose:
                    print("place a key with color {} on position {}".format(color, keys[i - 1]))
                    print("place a lock with color {} on {})".format(self.colors[goal_colors[i - 1]], locks[i - 1]))
                world[keys[i - 1][0], keys[i - 1][1]] = color
                world[locks[i - 1][0], locks[i - 1][1]] = self.colors[goal_colors[i - 1]]

            # keys[0] is orphaned key, so this happens outside the loop
            world[first_key[0], first_key[1]] = self.colors[goal_colors[0]]
            if self.verbose:
                print("place the first key with color {} on position {}".format(goal_colors[0], first_key))
            # a dead end is the end of a distractor branch, saved as color so it's consistent with world representation
            # iterate over distractor branches to place all distractors
            for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
                # choose x,y locations for keys and locks from keys and locks (previously determined so nothing collides)
                key_distractor = keys[self.goal_length - 1 + i * self.distractor_length: \
                                      self.goal_length - 1 + (i + 1) * self.distractor_length]
                lock_distractor = locks[self.goal_length - 1 + i * self.distractor_length: \
                                        self.goal_length - 1 + (i + 1) * self.distractor_length]
                # determine colors and place key,lock-pairs
                for k, (key, lock) in enumerate(list(zip(key_distractor, lock_distractor))):
                    if k == 0:  # first lock has color of root of distractor branch
                        color_lock = self.colors[goal_colors[root]]
                    else:
                        color_lock = self.colors[distractor_color[k - 1]]  # old key color now becomes current lock color
                    color_key = self.colors[distractor_color[k]]
                    world[key[0], key[1], :] = color_key
                    world[lock[0], lock[1]] = color_lock
                dead_ends.append(color_key)  # after loop is run through the remaining color_key is the dead end

        # place an agent
        world[agent_pos[0], agent_pos[1]] = AGENT_COLOR
        # convert goal colors to rgb so they have the same format as returned world
        goal_colors_rgb = [self.colors[col] for col in goal_colors]
        # add outline to world by padding
        world = np.pad(world, [(1, 1), (1, 1), (0, 0)])
        agent_pos += [1, 1]  # account for padding
        return world, agent_pos, dead_ends, goal_colors_rgb

    def save(self):
        """Save environement as ndarray.

        Environment can be loaded again by instantiating a BoxworldEnv with saved array as world parameter."""
        np.save('box_world.npy', self.world)

    def step(self, action):
        """Canonical step function."""
        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        self.num_env_steps += 1

        reward = self.step_cost #assumed to be negative
        done = self.num_env_steps == self.max_steps

        # Move player if the field in the moving direction is either
        if np.any(new_position < 1) or np.any(new_position >= self.n+1): #at boundary
            possible_move = False

        # elif np.array_equal(new_position, [0, 0]):
        #     possible_move = False

        elif self.is_empty(self.world[new_position[0], new_position[1]]):
            # No key, no lock
            possible_move = True

        elif new_position[1] == 1 or self.is_empty(self.world[new_position[0], new_position[1]-1]): #first condition
            # is to catch keys at left boundary
            # It is a key
            if self.is_empty(self.world[new_position[0], new_position[1]+1]):
                # Key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                self.world[0, 0] = self.owned_key
                if np.array_equal(self.owned_key, GOAL_COLOR):
                    # Goal reached
                    reward += self.reward_gem
                    done = True
                elif np.any([np.array_equal(self.owned_key,dead_end) for dead_end in self.dead_ends]): #reached a dead
                    # end, terminate episode
                    reward += self.reward_dead
                    done = True
                elif np.any([np.array_equal(self.owned_key, key) for key in self.correct_keys]):
                    reward += self.reward_correct_key
                else:
                    reward += self.reward_wrong_key
            else:
                possible_move = False
        else:
            # It is a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # The lock matches the key
                possible_move = True
            else:
                possible_move = False
                if self.verbose:
                    print("lock color is {}, but owned key is {}".format(
                        self.world[new_position[0], new_position[1]], self.owned_key))

        if possible_move:
            self.player_position = new_position
            self.update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
        }

        return self.world, reward, done, info

    def reset(self, world=None):
        """Canonical reset function.

        Uses random seed to generate new environment."""
        if world is None:
           self.world, self.player_position, self.dead_ends, self.correct_keys =self.world_gen(seed=self.np_random_seed)
        else:
            self.world, self.player_position, self.dead_ends, self.correct_keys = world

        self.num_env_steps = 0

        return self.world

    def render(self, mode="human", figAx=None):
        """Renders environment. 'human' renders by plotting it with matplotlib, 'return' just returns world array as
        uint8 ndarray.
        Can reuse fig, ax object to refresh image or render it into a larger figure.
        """
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

    def is_empty(self, room):
        """Checks wither place in grid is empty of a lock, i.e. either background or agent sitting on it."""
        return np.array_equal(room, BACKGD_COLOR) or np.array_equal(room, AGENT_COLOR)

    def update_color(self, world, previous_agent_loc, new_agent_loc):
        """Updates the grid after the agent has moved by refreshing the correct colors."""
        world[previous_agent_loc[0], previous_agent_loc[1]] = BACKGD_COLOR
        world[new_agent_loc[0], new_agent_loc[1]] = AGENT_COLOR


class RandomBoxworldEnv(BoxworldEnv):
    """Boxworld environment but with the possibility to specify parameters as lists which will be uniformly sampled
    from upon reset().

    So far, the rewards are hard-coded in the environment, not parameterized in the init.
    Args:
        Just like BoxWorldEnv, with the difference that goal_length, num_distractor, distractor_length can be lists
        that are randomly sampled from upon instantiating environment or  calling reset().
    """
    metadata = {'render.modes': ['human','return']}
    def __init__(self, n=12, list_goal_lengths=[5], list_num_distractors=[2], list_distractor_lengths=[2],
                 step_cost=0, reward_gem=10, reward_dead=0, reward_correct_key=1, reward_wrong_key=-1, #reward sucture
                 num_colors=20, max_steps=3000, world=None, verbose=False):

        self.list_goal_lengths = list_goal_lengths
        self.list_num_distractors = list_num_distractors
        self.list_distractor_lengths = list_distractor_lengths

        #sample initial configuration
        goal_length = random.choice(self.list_goal_lengths)
        num_distractor = random.choice(self.list_num_distractors)
        distractor_length = random.choice(self.list_distractor_lengths)
        super(RandomBoxworldEnv, self).__init__(n=n, goal_length=goal_length, num_distractor=num_distractor,
                                          distractor_length=distractor_length,
                                          step_cost=step_cost, reward_gem=reward_gem, reward_dead=reward_dead,
                                          reward_correct_key=reward_correct_key, reward_wrong_key=reward_wrong_key,
                                          num_colors=num_colors, max_steps=max_steps, world=world, verbose=verbose)

    def sample_config(self):
        """Samples new values of  self.goal_length, self.num_distractors, self.distractor_length from respective
        lists so that random world can be generated using world_gen() with minimal modifications."""
        self.goal_length = random.choice(self.list_goal_lengths)
        self.num_distractor = random.choice(self.list_num_distractors)
        self.distractor_length = random.choice(self.list_distractor_lengths)


    def world_gen(self, seed=None, plot_solution=True):
        """Randomized version of world_gen that first samples new solution configuration values, modifies them
        directly in classes attributes and creates a world from those with the parent classes world_gen()."""
        self.sample_config()
        return super(RandomBoxworldEnv, self).world_gen(seed=seed, plot_solution=plot_solution)