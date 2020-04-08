#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:13:16 2020

@author: nathan
"""

from perfect import wilson
from perfect import *

import matplotlib.pyplot as plt
from matplotlib import colors

from collections import deque, defaultdict

def create_maze(size, algo=wilson):
    """
    creates a maze of given size with given maze generation algorithm
    """
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
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    seen = set()
    cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
    
    for y2, x2 in algo(grid):
        for dy, dx in dirs:
            ym, xm = dy + y2, dx + x2
            key = 2 * dy + y2, 2 * dx + x2
            if not grid[ym, xm] and key in seen:
                screen[ym, xm] = 4
                plt.tight_layout(pad=0)
                plt.figure(figsize=(8, 8,))
                plt.axis('off')
                plt.imshow(screen, cmap=cmap)
                plt.show()
                screen[ym, xm] = 3
                break
        screen[y2, x2] = 4
        plt.tight_layout(pad=0)
        plt.figure(figsize=(8, 8,))
        plt.axis('off')
        plt.imshow(screen, cmap=cmap)
        plt.show()
        screen[y2, x2] = 2 if y2 == 1 and x2 == 1 else 3
        seen.add((y2, x2))

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

def dfs(grid):
    """
    solve maze using depth first search
    """
    goal = len(grid) - 2
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    
    stack = [(1,1,)]
    seen = set()
    y = x = 1
    
    while y != goal or x != goal:
        while stack[-1] in seen:
            stack.pop()
        y, x = stack.pop()
        seen.add((y, x,))
        yield y, x
        stack.extend((y+2*dy,x+2*dx) for dy,dx in dirs if not grid[y+dy,x+dx])

def bfs(grid):
    """
    solve maze using breadth first search
    """
    goal = len(grid) - 2
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    
    queue = deque([(1, 1,)])
    seen = set()
    y = x = 1
    
    while y != goal or x != goal:
        while queue[0] in seen:
            queue.popleft()
        y, x = queue.popleft()
        seen.add((y, x,))
        yield y, x
        queue.extend((y+2*dy,x+2*dx) for dy,dx in dirs if not grid[y+dy,x+dx])

def greedy(grid):
    """
    solve maze using greedy breadth first search
    """
    goal = len(grid) - 2
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    
    queue = [(1, 1,)]
    seen = set()
    y = x = 1
    
    while y != goal or x != goal:
        queue.sort(key = lambda x: -x[0] - x[1])
        while queue[0] in seen:
            queue.pop(0)
        y, x = queue.pop(0)
        seen.add((y, x,))
        queue.extend((y + 2 * dy, x + 2 * dx) for dy, dx in dirs if not \
            grid[y + dy, x + dx])
        yield y, x

def astar(grid):
    """
    solve maze using a* algorithm
    """
    goal = len(grid) - 2
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    
    gs = defaultdict(lambda: float('inf'))
    hs = lambda x: 2 * goal - sum(x)
    gs[(1, 1,)] = 0 - 2 * goal
    opened = {(1, 1,):0}
    closed = dict()
    y = x = 1
    
    while y != goal or x != goal:
        q = min(opened, key=lambda x: gs[x] + hs(x))
        y, x = q
        del opened[q]
        g = gs[q] + 2
        for dy, dx in dirs:
            if grid[y + dy, x + dx]: continue
            successor = y + 2*dy, x + 2*dx
            f = g + hs(successor)
            if opened.get(successor, float('inf')) <= f:
                continue
            if closed.get(successor, float('inf')) <= f:
                continue
            gs[successor] = g
            opened[successor] = f
        closed[q] = g - 2 + hs(q)
        yield q
    
def bibfs(grid):
    """
    solve maze using bidirectional breadth first search
    """
    dirs = [(0, -1,), (-1, 0,), (0, 1,), (1, 0,)]
    
    seen1 = set()
    seen2 = set()
    y1 = x1 = 1
    y2 = x2 = len(grid) - 2
    queue1 = deque([(y1, x1,)])
    queue2 = deque([(y2, x2,)])
    
    while (y2, x2,) not in seen1:
        while queue1[0] in seen1:
            queue1.popleft()
        y1, x1 = queue1.popleft()
        seen1.add((y1, x1,))
        yield y1, x1
        queue1.extend((y1+2*dy,x1+2*dx) for dy,dx in dirs if not grid[y1+dy,x1+dx])
        if (y1, x1,) in seen2:
            break

        while queue2[0] in seen2:
            queue2.popleft()
        y2, x2 = queue2.popleft()
        seen2.add((y2, x2,))
        yield y2, x2
        queue2.extend((y2+2*dy,x2+2*dx) for dy,dx in dirs if not grid[y2+dy,x2+dx])

















