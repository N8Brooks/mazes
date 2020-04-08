# -*- coding: utf-8 -*-

import numpy as np
from random import randrange, random
from noise import pnoise3
from collections import deque

import solve

def solvable(grid):
    """
    returns how many steps it takes (vertical distance) to finish a maze
    returns 0 if it is impossible
    """
    y = x = 1
    stack = deque([(0, y, x,)])
    goal = len(grid) - 2
    found = np.ones_like(grid, dtype=bool)
    
    while stack:
        i, y, x = stack.popleft()
        i += 1
        for y2, x2 in solve.neighbors(y, x, grid):
            if found[y2, x2]:
                if y2 == goal and x2 == goal:
                    return i
                else:
                    found[y2, x2] = False
                    stack.append((i, y2, x2,))
    
    return 0

def tree(size):
    """
    binary tree method of creating an imperfect, yet completable maze
    """
    assert size % 2
    # creat grid and open first cell
    grid = np.array([[True] * size] * size)
    grid[1,1] = False
    yield grid
    
    # flip first row towards (1, 1)
    for x in range(3, size, 2):
        grid[1, x] = grid[1, x - 1] = False
        yield grid
    
    for y in range(3, size, 2):
        # flip first col towards (1,1)
        grid[y, 1] = grid[y - 1, 1] = False
        yield grid
        
        # randomly flip rest of row up or left
        for x in range(3, size, 2):
            if randrange(8) == 0:
                grid[y, x] = grid[y, x - 1] = grid[y - 1, x] = False
            elif randrange(2):
                grid[y, x] = grid[y, x - 1] = False
            else:
                grid[y, x] = grid[y - 1, x] = False
            
            yield grid

def splatter(size):
    """
    Create a solvable maze with approximately 1/4 cells being random walls
    """
    solvable = False
    while not solvable:
        grid = np.random.randint(0, 2, size=(size, size,), dtype=bool)
        grid |= np.random.randint(0, 2, size=(size, size,), dtype=bool)
        grid &= np.random.randint(0, 2, size=(size, size,), dtype=bool)
        grid[0, 0:size] = grid[size - 1, 0:size] = True
        grid[0:size, 0] = grid[0:size, size - 1] = True
        grid[1, 1] = grid[size-2, size-2] = False
        for y, x in solve.greedy_bfs(grid):
            pass
        solvable = y == size - 2 and x == size - 2
    yield grid

def maze(size):
    """
    maze cellular automaton algorithm for generating mazes
    this generally produces imperfect not completable mazes
    """
    assert size % 2
    grid = np.random.randint(0, 2, size=(size, size,), dtype=bool)
    grid[0, 0:size] = grid[size - 1, 0:size] = True
    grid[0:size, 0] = grid[0:size, size - 1] = True
    
    key = hash(str(grid))
    looped = set()
    yield grid
    
    def alive(i, j):
        n = np.sum(grid[max(0, i-1):i+2, max(0, j-1):j+2]) - grid[i, j]
        return 1 if grid[i, j] and 0 < n < 6 else int(n == 3)
    
    while key not in looped:
        looped.add(key)
        grid = np.array([[alive(i, j) for j in range(size)] \
            for i in range(size)], dtype=bool)
        grid[0, 0:size] = grid[size - 1, 0:size] = True
        grid[0:size, 0] = grid[0:size, size - 1] = True
        key = hash(str(grid))
        yield grid

def mazectric(size):
    """
    mazectric cellular automaton algorithm for generating mazes
    the only real change between this and maze is the cap on alive is 5
    this generally produces imperfect not completable mazes
    """
    assert size % 2
    grid = np.random.randint(0, 2, size=(size, size,), dtype=bool)
    grid[0, 0:size] = grid[size - 1, 0:size] = True
    grid[0:size, 0] = grid[0:size, size - 1] = True
    
    key = hash(str(grid))
    looped = set()
    yield grid
    
    def alive(i, j):
        n = np.sum(grid[max(0, i-1):i+2, max(0, j-1):j+2]) - grid[i, j]
        return 1 if grid[i, j] and 0 < n < 5 else int(n == 3)
    
    while key not in looped:
        looped.add(key)
        grid = np.array([[alive(i, j) for j in range(size)] \
            for i in range(size)], dtype=bool)
        grid[0, 0:size] = grid[size - 1, 0:size] = True
        grid[0:size, 0] = grid[0:size, size - 1] = True
        key = hash(str(grid))
        yield grid

def perlin(size, cut=0, div=4):
    grid = np.random.randint(0, 2, size=(size, size,), dtype=bool)
    grid[0, 0:size] = grid[size - 1, 0:size] = True
    grid[0:size, 0] = grid[0:size, size - 1] = True
    while solvable(grid) < 200:
        print(solvable(grid))
        z = random() * 1000000
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                grid[y, x] = pnoise3(x / div, y / div, z) > cut
        grid[1, 1] = grid[size-2, size-2] = False
        for y, x in solve.greedy_bfs(grid):
            pass
    yield grid