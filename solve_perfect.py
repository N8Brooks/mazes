#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:09:12 2020

@author: nathan
"""

from perfect import wilson
from perfect import *
from imperfect import *

import matplotlib.pyplot as plt
from matplotlib import colors

from random import choice
from collections import deque, defaultdict
import numpy as np

def create_maze(y, x, algo=wilson):
    """
    function to create a maze of the given size
    """
    for grid in algo(y, x):
        pass
    grid.flags.writeable = False
    return grid

def solve_plot(algo, grid):
    """
    plots each iteration of a maze solver
    """
    screen = grid.astype(int)
    goal_y, goal_x = grid.shape
    goal_y, goal_x = goal_y - 2, goal_x - 2
    screen[1,1] = screen[goal_y, goal_x] = 2
    cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
    path = list()
    
    for i, (y2, x2) in enumerate(algo(grid, path), start=1):
        screen[y2, x2] = 4
        plt.title(f'steps: {i}')
        plt.tight_layout(pad=0)
        plt.figure(figsize=(6, 6,))
        plt.axis('off')
        plt.imshow(screen, cmap=cmap)
        screen[y2,x2]=2 if (y2==1 and x2==1) or \
            (y2==goal_y and x2==goal_x) else 3
    
    dist = sum(((path[i][0]-path[i-1][0])**2+(path[i][1]-path[i-1][1])**2)**0.5
        for i in range(1, len(path)))
    path.pop(0)
    for y2, x2 in path:
        screen[y2,x2]=2 if (y2==1 and x2==1) or\
            (y2==goal_y and x2==goal_x) else 4
        plt.title(f'steps: {i} - dist: {dist}')
        plt.tight_layout(pad=0)
        plt.figure(figsize=(6, 6,))
        plt.axis('off')
        plt.imshow(screen, cmap=cmap)

def solve_last(algo, grid):
    """
    plots last iteration of a maze solver
    """
    screen = grid.astype(int)
    goal = len(screen) - 2
    screen[1,1] = screen[goal, goal] = 2
    y1 = x1 = 1
    cmap = colors.ListedColormap(['white','black','green','mistyrose','red'])
    for i, (y2, x2) in enumerate(algo(grid), start=1):
        screen[(y1+y2)//2, (x1+x2)//2] = 3
        screen[y2, x2] = 2 if y2 == 1 and x2 == 1 else 3
        y1, x1 = y2, x2
    screen[y2, x2] = 4
    plt.title(f'steps: {i}')
    plt.tight_layout(pad=0)
    plt.figure(figsize=(6, 6,))
    plt.axis('off')
    plt.imshow(screen, cmap=cmap)
    plt.show()

dirs = [(1, 0,), (0, 1,), (-1, 0,), (0, -1,)]
def neighbors(y, x, grid):
    """
    function to find neighbors of a cell with horizontal/vertical moves
    """
    return [(y + dy, x + dx,) for dy, dx in dirs if not grid[y+dy, x+dx]]

def recover_path(path, parent, coords, goal=(1, 1,)):
    """
    function to recover path from a dictionary of moves
    """
    if path is None: return
    path.append(coords)
    while coords != goal:
        coords = parent.pop(coords)
        path.append(coords)

def mouse(grid, path=None):
    """
    solve maze using random mouse algorithm
    """
    coords = (1, 1,)
    goal = tuple(map(lambda x: x-2, grid.shape))
    parent = dict()
    while coords != goal:
        n = choice(neighbors(*coords, grid))
        parent.setdefault(n, coords)
        coords = n
        yield coords
    recover_path(path, parent, coords)

def wall(grid, path=None):
    """
    solve maze using the pledge algorithm
    """
    point = 0
    y = x = 1
    goal_y, goal_x = map(lambda x: x-2, grid.shape)
    parent = dict()
    yield y, x
    
    while y != goal_y or x != goal_x:
        neigh = [None if grid[y+dy, x+dx] else (dy+y, dx+x) for dy, dx in dirs]
        while neigh[point] is None:
            point = (point + 1) % 4
        parent.setdefault(neigh[point], (y, x,))
        y, x = neigh[point]
        point = (point - 1) % 4
        yield y, x
    
    recover_path(path, parent, (y, x,))

def tremaux(grid, path=None):
    """
    tremaux algorithms for solving a maze - uses greedy algorithm instead of 
    random path when it comes to a new junction
    """
    y1 = x1 = -1
    y2 = x2 = 1
    marks = {(y2, x2,):{(-1,-1,):1, (1,2,):0, (2,1,):0}}
    goal_y, goal_x = map(lambda x: x-2, grid.shape)
    
    def dist(arg):
        return abs(goal_y - arg[0]) + abs(goal_x - arg[1])
    
    while y2 != goal_y or x2 != goal_x:
        neigh = neighbors(y2, x2, grid)
        if (y2, x2,) in marks:
            # old junction
            old = marks[(y2, x2,)]
            if old[(y1, x1,)] == 0:
                old[(y1, x1,)] = 2
                y1, x1, y2, x2 = y2, x2, y1, x1
            else:
                old[(y1, x1,)] += 1
                y1, x1, (y2, x2) = y2, x2, min(neigh, key=old.get)
                old[(y2, x2,)] += 1
        elif len(neigh) > 2:
            # new junction
            new = {n:0 for n in neigh}
            new[(y1, x1,)] = 1
            y1,x1,(y2,x2)= y2,x2,min((a for a in neigh if a!=(y1,x1)),key=dist)
            new[(y2, x2,)] = 1
            marks[(y1, x1,)] = new
        elif len(neigh) == 2:
            # pathway
            y1,x1, (y2,x2)=y2,x2, neigh[0] if neigh[0]!=(y1,x1) else neigh[1]
        else:
            # deadend
            y1, x1, y2, x2 = y2, x2, y1, x1
        yield y2, x2
    
    # reconstruct path
    y1, x1, y2, x2 = y2, x2, y1, x1
    path.extend(((y1, x1,), (y2, x2,),))
    while y2 != 1 or x2 != 1:
        if (y2, x2,) in marks:
            for key, val in marks[(y2, x2,)].items():
                if val == 1 and key != (y1, x1,):
                    y1, x1, (y2, x2) = y2, x2, key
                    break
        else:
            neigh = [(y2+dy,x2+dx) for dy, dx in dirs if not grid[y2+dy,x2+dx]]
            y1, x1, (y2, x2) = y2, x2, next(n for n in neigh if n != (y1, x1,))
        path.append((y2, x2,))

def dfs(grid, path):
    """
    solve maze using depth first search
    """
    goal = tuple(map(lambda x: x-2, grid.shape))
    coords = (1, 1,)
    stack = [coords]
    parent = dict()
    
    while coords != goal and stack:
        coords = stack.pop()
        for n in neighbors(*coords, grid):
            if n not in parent:
                parent[n] = coords
                stack.append(n)
        yield coords
    
    recover_path(path, parent, coords)

def bfs(grid, path=None):
    """
    solve maze using breadth first search
    """
    goal = tuple(map(lambda x: x-2, grid.shape))
    coords = (1, 1,)
    queue = deque([coords])
    parent = dict()
    
    while coords != goal and queue:
        coords = queue.popleft()
        for n in neighbors(*coords, grid):
            if n not in parent:
                parent[n] = coords
                queue.append(n)
        yield coords
    
    recover_path(path, parent, coords)

def greedy_bfs(grid, path=None):
    """
    solve maze using weighted bfs with octile distance
    """
    def dist(coords):
        return abs(coords[0]-goal[0]) + abs(coords[1]-goal[1])
    
    goal = tuple(map(lambda x: x-2, grid.shape))
    coords = (1, 1,)
    queue = {coords}
    parent = dict()
    
    # pursue coordinates with least octile distance
    while coords != goal and queue:
        coords = min(queue, key=dist)
        queue.remove(coords)
        for n in neighbors(*coords, grid):
            if n not in parent:
                parent[n] = coords
                queue.add(n)
        yield coords
    
    recover_path(path, parent, coords)

def greedy_dfs(grid, path=None):
    """
    solve maze using best depth first search
    """
    def dist(coords):
        return abs(coords[0]-goal[0]) + abs(coords[1]-goal[1])
    
    coords = (1, 1,)
    goal = tuple(map(lambda x: x-2, grid.shape))
    stack = [coords]
    parent = dict()
    
    while coords != goal and stack:
        y, x = coords = stack.pop()
        for n in sorted(neighbors(y, x, grid), key=dist):
            if n not in parent:
                parent[n] = coords
                stack.append(n)
        yield coords
    
    recover_path(path, parent, (y, x,))

def astar(grid, path=None):
    """
    solve maze using weighted bfs with manhattan distance heuristic
    """
    def h(y, x):
        return abs(goal_y - y) + abs(goal_x - x)
    
    goal_y, goal_x = map(lambda x: x-2, grid.shape)
    opened = {(1, 1,)}
    closed = set()
    parent = dict()
    
    gScore = np.full_like(grid, 10e9, dtype=float)
    gScore[1, 1] = 0
    fScore = np.full_like(grid, 10e9, dtype=float)
    
    while opened:
        # find point on open with minimal f
        y, x = coords = min(opened, key=lambda arg: fScore[arg[0], arg[1]])
        if y == goal_y and x == goal_x:
            break
        
        closed.add(coords)
        opened.remove(coords)
        
        # only travel to successor if f is lower
        g = gScore[y, x]
        for coords2 in neighbors(y, x, grid):
            y2, x2 = coords2
            g2 = g + ((y - y2)**2 + (x - x2)**2)**0.5
            if g2 < gScore[y2, x2]:
                parent[coords2] = coords
                gScore[y2, x2] = g2
                fScore[y2, x2] = g2 + h(y2, x2)
                if coords2 not in closed:
                    opened.add(coords2)
        
        yield coords
    
    recover_path(path, parent, coords)
    
def bibfs(grid, path=None):
    """
    solve maze using bi-directional breadth first search
    """
    parents = [dict(), dict()]
    goal = tuple(map(lambda x: x-2, grid.shape))
    queues = [deque([(1, 1,)]), deque([goal])]
    i, j = 0, 1
    
    while True:
        # find next coords
        if queues[i]:
            coords = queues[i].popleft()
            if coords in parents[j]:
                yield coords
                recover_path(path, parents[1], coords, goal)
                path.reverse()
                recover_path(path, parents[0], coords)
                return
        else:
            break
        
        # add positions and extend queue
        for n in neighbors(*coords, grid):
            if n not in parents[i]:
                parents[i][n] = coords
                queues[i].append(n)
        yield coords
        i, j = j, i
    
def greedy_bibfs(grid, path=None):
    """
    solve maze using bi-directional breadth first search
    """
    def dist(coords):
        return abs(coords[0]-goal_y) + abs(coords[1]-goal_x)
    
    goal = tuple(map(lambda x: x-2, grid.shape))
    goal_y, goal_x = goal
    parents = [dict(), dict()]
    queues = [{(1, 1,)}, {goal}]
    i, j = 0, 1
    
    while True:
        # find next coords
        if queues[i]:
            coords = min(queues[i],key=dist) if j else max(queues[i],key=dist)
            queues[i].remove(coords)
            if coords in parents[j]:
                yield coords
                recover_path(path, parents[1], coords, goal)
                path.reverse()
                recover_path(path, parents[0], coords)
                return
        else:
            break
        
        # add positions and extend queue
        for n in neighbors(*coords, grid):
            if n not in parents[i]:
                parents[i][n] = coords
                queues[i].add(n)
        yield coords
        i, j = j, i
    
    
    
    
    
    
    
    
    
    