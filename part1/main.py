import numpy as np
import maze as mz
import maze_functions as mzf
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython import display
from plotting import *
import random



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

 
    dead = 0
    goal = 0
    #horizon = 22
    method = 'DynProg'
    start  = (0, 0)
    startM = (5, 6)
    iterations = 2

    #horizon = 15


    horizon_vector = []
    survive_vector = []
    for _ in range(100):
        #horizon = np.random.geometric(p=1-0.9666, size=1)
        horizon = random.randrange(0, 21, 1) 

        horizon_vector.append(horizon)

        if horizon < 16:
            survive_vector.append(0)
        else:
            survive_vector.append(1)

  
    plotResult("Probability surviving given time horizon", "Time Horizon", "Survive", horizon_vector, survive_vector)
    




    """
    for _ in range(1):
        horizon = 15       
        policy, pathM = mzf.get_policy_and_pathM(env, horizon, start, startM, weights)
        path = env.simulate(start, policy, method)
        mzf.animate_solution(maze, path, pathM)
    """
   



    """
    for i in range(1, iterations): 

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

        mzf.animate_solution(maze, path, pathM)
        print(dead)
        print(goal)
    """


if __name__ == "__main__":
    main()
