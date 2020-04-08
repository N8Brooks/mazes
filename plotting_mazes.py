#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:07:53 2020

@author: nathan
"""

from perfect import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib import colors

algorithms = [wilson, backtracker, kruskal, prim, 
              modified_prim, tree, eller, sidewinder, division, ]

mpl.rcParams['savefig.pad_inches'] = 0

frames = list()
for algo in algorithms:
    # 1 sec of title
    frames.extend([algo.__name__.capitalize()] * 120)
    
    # x secs of algorithm
    if algo.__name__ == 'division':
        frames.extend([np.copy(x) for x in algo(107, 191)])
        frames.extend([frames[-1]] * 59)
    if algo.__name__ == 'wilson':
        frames.extend([np.copy(x) for x in algo(27, 49)])
        frames.extend([frames[-1]] * 59)
    else:
        frames.extend([np.copy(x) for x in algo(53, 95)])
        frames.extend([frames[-1]] * 59)

frames = dict(zip(range(len(frames)), frames))
fig = plt.figure(figsize=(16, 9,))
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
        plt.imshow(frame, cmap=colors.ListedColormap(['white', 'black']), aspect='auto')
    

anim=animation.FuncAnimation(fig, animate, blit=False, \
                             frames=len(frames), interval=1)

anim.save(f'mazes_animation.mp4',\
          writer=animation.FFMpegWriter(fps=60))