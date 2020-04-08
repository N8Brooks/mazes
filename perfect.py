#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:24:17 2020

@author: nathan
"""

import numpy as np
from random import randrange, choice, shuffle, sample
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors

def maze_plot(algo, y_max, x_max):
    """
    plots each iteration of maze generation for algorithm
    """
    for grid in algo(y_max, x_max):
        plt.tight_layout(pad=0)
        plt.figure(figsize=(8, 8,))
        plt.axis('off')
        plt.imshow(grid, cmap=colors.ListedColormap(['white', 'black']))
        plt.show()

def maze_last(algo, size):
    """
    plots maze generation for algorithm and returns time taken
    """
    t0 = time()
    for grid in algo(size):
        pass
    t1 = time()
    plt.tight_layout(pad=0)
    plt.figure(figsize=(8, 8,))
    plt.axis('off')
    plt.imshow(grid, cmap=colors.ListedColormap(['white', 'black']))
    plt.show()
    return t1 - t0

def valid(grid):
    """
    returns if the given grid is a valid perfect maze
    """
    size = len(grid)
    if grid[1:size:2, 1:size:2].any():
        return False
    if not grid[0:size:2, 0:size:2].all():
        return False
    if not grid[0].all() or not grid[-1].all():
        return False
    if not grid[:,0].all() or not grid[:,-1].all():
        return False
    
    dirs = [(-1, 0,), (1, 0,), (0, 1,), (0, -1,)]
    seen = set()
    def walk(pre, y, x):
        seen.add((y, x,))
        neigh = [(y+2*dy, x+2*dx,) for dy, dx in dirs if not grid[y+dy,x+dx]]
        if pre in neigh: neigh.remove(pre)
        if not neigh:
            return True
        elif any(key in seen for key in neigh):
            return False
        else:
            return all(walk((y, x,), *key) for key in neigh)
    
    return walk((-1, -1,), 1, 1) and len(seen) == (size // 2) * (size // 2)

def dfs(y_max, x_max):
    """
    depth first search method of generating a maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    dirs = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    def dfs(y, x):
        # set current cell to False and find neighbors
        grid[y, x] = False
        yield True
        neigh=[(y + dy, x + dx) for dy, dx in dirs]
        neigh=[(y,x) for y,x in neigh if 0<y<y_max and 0<x<x_max and grid[y,x]]
        
        # while there are neighbord call dfs for random neighbor
        while neigh:
            ny, nx = choice(neigh)
            grid[(y + ny)//2, (x + nx)//2] = False
            for _ in dfs(ny, nx):
                yield True
            neigh = [(y, x) for y, x in neigh if grid[y,x]]
    
    # call dfs for random cell
    for _ in dfs(2*randrange(y_max // 2) + 1, 2*randrange(x_max // 2) + 1):
        yield grid

def backtracker(y_max, x_max):
    """
    iterative version of depth fist search generation of maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    dirs = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    stack = [(2*randrange(y_max//2)+1, 2*randrange(x_max//2)+1,)]
    
    while stack:
        y, x = stack.pop()
        if grid[y, x]:
            grid[y, x] = False
            yield grid
        neigh=[(y + dy, x + dx) for dy, dx in dirs]
        neigh=[(y,x) for y,x in neigh if 0<y<y_max and 0<x<x_max and grid[y,x]]
        if len(neigh) > 1: stack.append((y, x,))
        if neigh:
            ny, nx = choice(neigh)
            grid[(y + ny)//2, (x + nx)//2] = False
            stack.append((ny, nx,))
            yield grid

def kruskal(y_max, x_max):
    """
    randomized kruskal's algorithm for generating a maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    
    # simple union find for each cell
    parent = {(y,x):(y,x) for y in range(1,y_max,2) for x in range(1,x_max,2)}
    def find(x):
        return x if parent[x] == x else find(parent[x])
    def unite(x, y):
        parent[find(x)] = find(y)
    
    # shuffled list of walls
    walls = [(1, x) for x in range(2, x_max - 1, 2)]
    for y in range(2, y_max - 2, 2):
        walls.extend((y, x) for x in range(1, x_max, 2))
        y += 1
        walls.extend((y, x) for x in range(2, x_max - 1, 2))
    shuffle(walls)
    
    # join cells adjacent to wall if they are in different sets
    for y, x in walls:
        if y % 2:
            coord1 = (y, x + 1,)
            coord2 = (y, x - 1,)
        else:
            coord1 = (y + 1, x,)
            coord2 = (y - 1, x,)
        if find(coord1) != find(coord2):
            grid[coord1[0], coord1[1]] = grid[coord2[0], coord2[1]] = False
            grid[y, x] = False
            unite(coord1, coord2)
            yield grid

def prim(y_max, x_max):
    """
    randomized prims algorithm for generating a maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    dirs = [(1, 0,), (-1, 0,), (0, 1,), (0, -1,)]
    
    # mark random cell and add adjacent walls to list
    y = 2 * randrange(y_max // 2) + 1
    x = 2 * randrange(x_max // 2) + 1
    grid[y, x] = False
    walls = {(y + dy, x + dx,) for dy, dx in dirs}
    yield grid
    
    while walls:
        # check wall coordinates and make sure they are valid
        y, x = sample(walls, 1)[0]
        walls.remove((y, x,))
        if y == 0 or y == y_max - 1 or x == 0 or x == x_max - 1:
            continue
        
        # find adjacent cells
        if y % 2:
            y1 = y2 = y
            x1, x2 = x + 1, x - 1
        else:
            y1, y2 = y + 1, y - 1
            x1 = x2 = x
        
        # if one is unvisited, add it, and add its walls to list
        if grid[y1, x1] != grid[y2, x2]:
            if grid[y1, x1]:
                grid[y1, x1] = grid[y, x] = False
                walls.update((y1 + dy, x1 + dx,) for dy, dx in dirs)
            else:
                grid[y2, x2] = grid[y, x] = False
                walls.update((y2 + dy, x2 + dx,) for dy, dx in dirs)
            yield grid
            
def modified_prim(y_max, x_max):
    """
    modified randomized prims algorithm for generating a maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    dirs = [(2, 0,), (-2, 0,), (0, 2,), (0, -2,)]
    
    # mark cell and add its neighbors to cells
    y = 2 * randrange(y_max // 2) + 1
    x = 2 * randrange(x_max // 2) + 1
    grid[y, x] = False
    cells = [(y + dy, x + dx,) for dy, dx in dirs]
    cells = {(y, x,) for y, x in cells if 0 < y < y_max and 0 < x < x_max}
    yield grid
    
    # continually remove cell and add neighbors that haven't been reached
    while cells:
        y, x = sample(cells, 1)[0]
        cells.remove((y, x,))
        neigh = [(y + dy, x + dx,) for dy, dx in dirs]
        neigh = [(y, x) for y, x in neigh if 0 < y < y_max and 0 < x < x_max]
        ny, nx = choice([(y, x,) for y, x in neigh if not grid[y, x]])
        grid[y, x] = grid[(ny + y) // 2, (nx + x) // 2] = False
        cells.update((y, x,) for y, x in neigh if grid[y, x])
        yield grid
    
def wilson(y_max, x_max):
    """
    wison's algorithm for generating unbiased mazes
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    dirs = [(2, 0,), (-2, 0,), (0, 2,), (0, -2,)]
    free = {(y, x,) for y in range(1, y_max, 2) for x in range(1, x_max, 2)}
    path = set()
    
    # initialize maze
    y = 2 * randrange(y_max // 2) + 1
    x = 2 * randrange(x_max // 2) + 1
    grid[y, x] = False
    free.remove((y, x,))
    yield grid
    
    while free:
        # start a random walk
        key = sample(free, 1)[0]
        y, x = key
        free.remove(key)
        path = [key]
        grid[y, x] = False
        yield grid
        
        # find its neighboring cells
        neigh = [(y + dy, x + dx,) for dy, dx in dirs]
        neigh = [(y, x) for y, x in neigh if 0 < y < y_max and 0 < x < x_max]
        key = choice(neigh)
        y, x = key
        
        # keep adding to random walk until it hits an open cell
        while grid[y, x]:
            grid[y, x] = False
            grid[(y + path[-1][0]) // 2, (x + path[-1][1]) // 2] = False
            neigh = [(y + dy, x + dx,) for dy, dx in dirs]
            neigh.remove(path[-1])
            neigh = [(y, x) for y, x in neigh if 0<y<y_max and 0<x<x_max]
            free.remove(key)
            path.append(key)
            key = choice(neigh)
            y, x = key
            yield grid
        
        # self erase if it hit its own path or add to maze
        if key in path:
            key1 = path.pop()
            free.add(key1)
            grid[key1[0], key1[1]] = True
            for key2 in reversed(path):
                free.add(key2)
                grid[key2[0], key2[1]] = True
                grid[(key1[0]+key2[0])//2, (key1[1]+key2[1])//2] = True
                key1 = key2
                yield grid
        else:
            grid[(y + path[-1][0]) // 2, (x + path[-1][1]) // 2] = False
            yield grid

def division(y_max, x_max):
    """
    division method of creating a maze
    """
    # empty walls to start
    assert y_max % 2 and x_max % 2
    grid = np.zeros((y_max, x_max,), dtype=bool)
    grid[0, 0:x_max] = grid[y_max - 1, 0:x_max] = True
    grid[0:y_max, 0] = grid[0:y_max, x_max - 1] = True
    
    def divide(lo_y, hi_y, lo_x, hi_x):
        if hi_x - lo_x < 3 or hi_y - lo_y < 3:
            return
        
        # choose to double doors on vertical or horizontal division
        if randrange(2):
            # divide with vertical line and add door
            x = randrange(lo_x + 2, hi_x, 2)
            grid[lo_y:hi_y, x] = True
            grid[randrange(lo_y + 1, hi_y, 2), x] = False
            yield None
            
            # divide with horizontal line and add two doors
            y = randrange(lo_y + 2, hi_y, 2)
            grid[y, lo_x:hi_x] = True
            grid[y, randrange(lo_x + 1, x, 2)] = False
            grid[y, randrange(x + 1, hi_x, 2)] = False
            yield None
        else:
            # divide with horizontal line and add two doors
            y = randrange(lo_y + 2, hi_y, 2)
            grid[y, lo_x:hi_x] = True
            grid[y, randrange(lo_x + 1, hi_x, 2)] = False
            yield None
            
            # divide with vertical line and add door
            x = randrange(lo_x + 2, hi_x, 2)
            grid[lo_y:hi_y, x] = True
            grid[randrange(lo_y + 1, y, 2), x] = False
            grid[randrange(y + 1, hi_y, 2), x] = False
            yield None
            
        
        # call for each chamber
        for _ in divide(lo_y, y, lo_x, x):
            yield None
        for _ in divide(y, hi_y, lo_x, x):
            yield None
        for _ in divide(lo_y, y, x, hi_x):
            yield None
        for _ in divide(y, hi_y, x, hi_x):
            yield None
    
    # call divide method for main chamber
    for _ in divide(0, y_max - 1, 0, x_max - 1):
        yield grid

def tree(y_max, x_max):
    """
    binary tree method of creating a maze
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    grid[1,1] = False
    yield grid
    
    # flip first row towards (1, 1)
    for x in range(3, x_max, 2):
        grid[1, x] = grid[1, x - 1] = False
        yield grid
    
    for y in range(3, y_max, 2):
        # flip first col towards (1,1)
        grid[y, 1] = grid[y - 1, 1] = False
        yield grid
        
        # randomly flip rest of row up or left
        for x in range(3, x_max, 2):
            if randrange(2):
                grid[y, x] = grid[y, x - 1] = False
            else:
                grid[y, x] = grid[y - 1, x] = False
            yield grid

def eller(y_max, x_max):
    """
    eller's method of generating mazes
    todo: use union find remove data structure
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    
    # cells are members of there own set
    parent = {x:{x} for x in range(1, x_max, 2)}
    
    for y in range(1, y_max - 2, 2):
        # create walls between cells
        grid[y, 1] = False
        yield grid
        for x in range(3, x_max, 2):
            if x not in parent[x - 2] and randrange(2):
                parent[x].update(parent[x - 2])
                for key in list(parent[x - 2]):
                    parent[key] = parent[x]
                grid[y, x-1:x+1] = False
            else:
                grid[y, x] = False
            yield grid
        
        # making bottom walls with at least one passage for each set
        for members in {frozenset(x) for x in parent.values()}:
            walls = [list(), list()]
            for x in members:
                walls[randrange(2)].append(x)
            if not walls[0]:
                walls.reverse()
            for x in walls[0]:
                grid[y+1, x] = False
                yield grid
            for x in walls[1]:
                for key in parent:
                    parent[key].discard(x)
                parent[x] = {x}
    
    # finish maze by making sure none are isolated
    y = y_max - 2
    grid[y, 1] = False
    yield grid
    for x in range(3, x_max, 2):
        if x not in parent[x - 2]:
            parent[x].update(parent[x - 2])
            for key in list(parent[x - 2]):
                parent[key] = parent[x]
            grid[y, x-1:x+1] = False
        else:
            grid[y, x] = False
        yield grid

def sidewinder(y_max, x_max):
    """
    sidewinder algorithm for maze generation
    """
    assert y_max % 2 and x_max % 2
    grid = np.ones((y_max, x_max,), dtype=bool)
    grid[1,1:x_max-1] = False
    yield grid
    
    for y in range(3, y_max, 2):
        # create horizontal hallways
        grid[y, 1] = False
        run = [1]
        yield grid
        for x in range(3, x_max, 2):
            if randrange(2):
                grid[y, x-1:x+1] = False
                run.append(x)
            else:
                # create door
                grid[y-1, choice(run)] = False
                yield grid
                grid[y, x] = False
                run = [x]
            yield grid
        if run:
            grid[y - 1, choice(run)] = False
            yield grid

