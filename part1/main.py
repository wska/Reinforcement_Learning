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
        [-1, -1,   -1,   -1,   -100,   60,   -1,   -1],
        ])
    #mz.draw_maze(maze)
    #plt.show()


    # Create an environment maze
    env = mz.Maze(maze)
    #env.show()


    horizon = 20
    method = 'DynProg';
    start  = (0,0);
    startM = (5, 6);



    policy, pathM = mzf.get_policy_and_pathM(env, horizon, start, startM, weights)

    path = env.simulate(start, policy, method)

    mzf.animate_solution(maze, path, pathM)



if __name__ == "__main__":
    main()
