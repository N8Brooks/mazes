#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:15:59 2020

@author: nathan
"""

from generation import wilson
from generation import *

import matplotlib.pyplot as plt
from matplotlib import colors

def create_maze(size, algo=wilson):
    for grid in algo(size):
        pass
    grid.flags.writeable = False
    return grid

def plot_all(algo, grid):
    """
    plots each iteration of a maze solver
    """
    screen = grid.astype(int)
    goal = len(screen) - 2
    screen[1,1] = screen[goal, goal] = 2
    y1 = x1 = 1
    cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
    for y2, x2 in algo(grid):
        screen[(y1+y2)//2, (x1+x2)//2] = 3
        screen[y2, x2] = 4
        plt.tight_layout(pad=0)
        plt.figure(figsize=(8, 8,))
        plt.axis('off')
        plt.imshow(screen, cmap=cmap)
        plt.show()
        screen[y2, x2] = 2 if y2 == 1 and x2 == 1 else 3
        y1, x1 = y2, x2

def plot_last(algo, grid):
    """
    plots last iteration of a maze solver
    """
    screen = grid.astype(int)
    goal = len(screen) - 2
    screen[1,1] = screen[goal, goal] = 2
    y1 = x1 = 1
    cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
    for y2, x2 in algo(grid):
        screen[(y1+y2)//2, (x1+x2)//2] = 3
        screen[y2, x2] = 2 if y2 == 1 and x2 == 1 else 3
        y1, x1 = y2, x2
    screen[y2, x2] = 4
    plt.tight_layout(pad=0)
    plt.figure(figsize=(8, 8,))
    plt.axis('off')
    plt.imshow(screen, cmap=cmap)
    plt.show()