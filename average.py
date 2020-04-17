#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:06:54 2020

@author: nathan
"""

from perfect import *
from imperfect import solvable

algorithms = [backtracker, division, eller, kruskal, modified_prim,
              prim, sidewinder, tree, wilson]

import pandas as pd
from tqdm import trange
df = pd.DataFrame(columns=[a.__name__ for a in algorithms])

for i in trange(11, 1002, 10):
    rec = pd.Series(name=i)
    
    for algo in algorithms:
        rec[algo.__name__] = sum(solvable(create_maze(i, algo)) for _ in range(16)) // 16
    
    df.loc[i] = rec