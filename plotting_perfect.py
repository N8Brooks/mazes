#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:57:02 2020

@author: nathan
"""


from perfect import *
from solve_perfect import *

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib import colors

from perfect import tree
from solve_perfect import dfs


algorithms = [(mouse, 'Mouse', create_maze(9, 13, modified_prim)),
              (wall, 'Wall Follower', create_maze(31, 55, modified_prim),),
              (bfs, 'Breadth First Search', create_maze(31, 55, wilson),),
              (dfs, 'Depth First Search', create_maze(55, 97, wilson),),
              (bibfs, 'Bidirectional Breadth First Search', create_maze(55, 97, wilson),),
              (tremaux, 'Tremaux', create_maze(55, 97, wilson),),
              (greedy_bfs, 'Best First Search', create_maze(55, 97, wilson),),
              (astar, 'A*', create_maze(55, 97, wilson),),
              (greedy_bibfs, 'Bidirectional Best First Search', create_maze(55, 97, wilson),), ]

mpl.rcParams['savefig.pad_inches'] = 0
cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
frames = list()

for algo, name, grid in algorithms:
    # 2 sec of title
    frames.extend([name] * 120)
    
    screen = grid.astype(int)
    goal_y, goal_x = map(lambda x: x-2, grid.shape)
    screen[1,1] = screen[goal_y, goal_x] = 2
    path = list()
    
    # x secs of algorithm
    for i, (y2, x2) in enumerate(algo(grid, path), start=1):
        screen[y2, x2] = 4
        frames.append(screen.copy())
        screen[y2,x2]=2 if (y2==1 and x2==1) or \
            (y2==goal_y and x2==goal_x) else 3
    
    # x secs of path
    path.pop(0)
    for y2, x2 in path:
        screen[y2,x2]=2 if (y2==1 and x2==1) or \
            (y2==goal_y and x2==goal_x) else 4
        frames.append(screen.copy())
    
    # 2 secs of complete
    frames.extend([frames[-1]] * 119)

fig = plt.figure(figsize=(32, 18,))
plt.axis('off')
pbar = tqdm(total=len(frames))

def animate(i):
    frame = frames[i]
    pbar.update()
    plt.clf()
    plt.axis('off')
    if isinstance(frame, str):
        plt.text(0.5, 0.5, frame, fontsize=48, 
             horizontalalignment='center', verticalalignment='center')
    else:
        sizes = np.shape(frame)     
        fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(frame, cmap=cmap, aspect='auto')
    

anim=animation.FuncAnimation(fig, animate, blit=False, \
                             frames=len(frames), interval=1)

anim.save(f'perfect_animation.mp4',\
          writer=animation.FFMpegWriter(fps=60))