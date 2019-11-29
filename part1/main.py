import numpy as np
import maze as mz
import maze_functions as mzf
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython import display



def main():
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
        ])

    weights = np.array([
        [-1, -1,   -100, -1,   -1,   -1,   -1,   -1],
        [-1, -1,   -100, -1,   -1,   -100, -1,   -1],
        [-1, -1,   -100, -1,   -1,   -100, -100, -100],
        [-1, -1,   -100, -1,   -1,   -100, -1,   -1],
        [-1, -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [-1, -100, -100, -100, -100, -100, -100, -1],
        [-1, -1,   -1,   -1,   -100,  1,  -1,   -1],
        ])

    env = mz.Maze(maze)
    #env.show()

 
    #dead = 0
    #goal = 0
    horizon = 22
    method = 'DynProg';
    start  = (0, 0);
    startM = (6, 6);
    #iterations = 20

    for _ in range(1):        
        policy, pathM = mzf.get_policy_and_pathM(env, horizon, start, startM, weights)
        path = env.simulate(start, policy, method)
        mzf.animate_solution(maze, path, pathM)        
            
    


"""
    for i in range(1, iterations+1): 

        policy, pathM = mzf.get_policy_and_pathM(env, horizon, start, startM, weights)

        path = env.simulate(start, policy, method)

        same_pos = np.array_equal(path, pathM)
        if same_pos:
            dead += 1   
        
        else: 
            if startM in path:
                goal += 1
            else:
                dead += 1

        #mzf.animate_solution(maze, path, pathM)
"""

if __name__ == "__main__":
    main()
