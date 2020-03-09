import numpy as np
import random
import matplotlib.pyplot as plt

def sampling_pairs(num_pair, n=12):
    """Generates random key,lock pairs with corresponding x,y-coordinates that can be filled with correct or
    distractor colors later on, as well as a random agent x,y-position.
    First key is returned separately because it comes without a neighbor.
    """
    possibilities = set(range(1, n*(n-1)))
    keys = []
    locks = []
    for k in range(num_pair):
        key = random.sample(possibilities, 1)[0]
        key_x, key_y = key//(n-1), key%(n-1)
        lock_x, lock_y = key_x, key_y + 1
        to_remove = [key_x * (n-1) + key_y] +\
                    [key_x * (n-1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] +\
                    [key_x * (n-1) - i + key_y for i in range(1, min(2, key_y) + 1)]

        possibilities -= set(to_remove)
        keys.append([key_x, key_y])
        locks.append([lock_x, lock_y])
    agent_pos = random.sample(possibilities, 1)
    possibilities -= set(agent_pos)
    first_key = random.sample(possibilities, 1)

    agent_pos = np.array([agent_pos[0]//(n-1), agent_pos[0]%(n-1)])
    first_key = first_key[0]//(n-1), first_key[0]%(n-1)
    return keys, locks, first_key, agent_pos


colors = {0: [0, 0, 117], #no black because it's used for outline
          1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
          4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
          9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
          14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144]}

num_colors = len(colors)
agent_color = [128, 128, 128]
goal_color = [255, 255, 255]
grid_color = [220, 220, 220]

def plot_solution_graph(goal_colors, distractor_colors, distractor_roots, colors):
    """Plots game problem as directed graph of colors that were used to render a given environment.
    """
    vis = np.ones([len(distractor_roots)+1,
                   max(len(goal_colors),
                       max([distractor_roots[path]+len(distractor_colors[path]) for path in range(len(
                           distractor_roots))])) +1, 3], dtype=int)*grid_color[0] #length of longest path
    for i,col in enumerate(goal_colors):
        vis[0,i,:] = colors[col]
    vis[0,len(goal_colors),:] = goal_color
    for j,dist_branch in enumerate(distractor_colors):
        for i_raw,dist_col in enumerate(dist_branch):
            i = i_raw + distractor_roots[j]+1
            vis[j+1,i,:] = colors[dist_col]
    plt.title("problem graph")
    plt.imshow(vis)
    plt.yticks(ticks=list(range(len(distractor_roots)+1)),
               labels= ["solution path"] + [f"distractor path {i+1}" for i in range(len(distractor_roots))])
    plt.xticks(list(range(len(goal_colors)+1)))
    plt.xlabel("key #")

def world_gen(n=12, goal_length=5, num_distractor=2, distractor_length=2, seed=None):
    """generate boxworld
    """
    if seed is None:
        random.seed(seed)

    world_dic = {}
    world = np.ones((n, n, 3), dtype=np.int8) * 220
    goal_colors = random.sample(range(num_colors), goal_length - 1)
    distractor_possible_colors = [color for color in range(20) if color not in goal_colors]
    distractor_colors = [random.sample(distractor_possible_colors, distractor_length) for k in range(num_distractor)]
    distractor_roots = random.choices(range(goal_length - 1), k=num_distractor)
    #this line mainly prevents arbitrary distractor path length
    keys, locks, first_key, agent_pos = sampling_pairs(goal_length - 1 + distractor_length * num_distractor, n)
    plot_solution_graph(goal_colors, distractor_colors, distractor_roots, colors)

    # first, create the goal path
    for i in range(1, goal_length):
        if i == goal_length - 1:
            color = goal_color  # final key is white
        else:
            color = colors[goal_colors[i]]
        print("place a key with color {} on position {}".format(color, keys[i-1]))
        print("place a lock with color {} on {})".format(colors[goal_colors[i-1]], locks[i-1]))
        world[keys[i-1][0], keys[i-1][1]] = np.array(color)
        world[locks[i-1][0], locks[i-1][1]] = np.array(colors[goal_colors[i-1]])

    # keys[0] is an orphand key so skip it
    world[first_key[0], first_key[1]] = np.array(colors[goal_colors[0]])
    print("place the first key with color {} on position {}".format(goal_colors[0], first_key))

    # a dead end is the end of a distractor branch, saved as color so it's consistent with returned world representation
    dead_ends = []
    # place distractors
    #iterate over branches
    for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
        #choose x,y locations for keys and locks from keys and locks (previously determined so nothing collides)
        key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
        lock_distractor = locks[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
        #determine colors and place key,lock-pairs
        for k, (key,lock) in enumerate(list(zip(key_distractor, lock_distractor))):
            if k == 0: # first lock has color of root of distractor branch
                color_lock = colors[goal_colors[root]]
            else:
                color_lock = colors[distractor_color[k - 1]]  # old key color now becomes current lock color
            color_key = colors[distractor_color[k]]
            world[key[0],key[1],:] = np.array(color_key)
            world[lock[0],lock[1]] = np.array(color_lock)
        dead_ends.append(color_key) #after loop is run through the remaining color_key is the dead end

    # place an agent
    world[agent_pos[0], agent_pos[1]] = np.array(agent_color)
    # convert goal colors to rgb so they have the same format as returned world
    goal_colors_rgb = [colors[col] for col in goal_colors]
    #add outline to world by padding
    world = np.pad(world, [(1,1),(1,1),(0,0)])
    agent_pos += [1,1] #account for padding
    return world, agent_pos, dead_ends, goal_colors_rgb

def update_color(world, previous_agent_loc, new_agent_loc):
        world[previous_agent_loc[0], previous_agent_loc[1]] = grid_color
        world[new_agent_loc[0], new_agent_loc[1]] = agent_color

def is_empty(room):
    return np.array_equal(room, grid_color) or np.array_equal(room, agent_color)
