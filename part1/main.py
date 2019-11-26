import numpy as np
import maze as mz
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
    # with the convention 
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze

    #mz.draw_maze(maze)
    #plt.show()


    # Create an environment maze
    env = mz.Maze(maze)
    #env.show()


    
    # Finite horizon
    horizon = 20
    # Solve the MDP problem with dynamic programming 
    V, policy= mz.dynamic_programming(env,horizon);

    #print(policy)


    # Simulate the shortest path starting from position A
    method = 'DynProg';
    start  = (0,0);
    path = env.simulate(start, policy, method);



    



    # Simulate the minotaur step
    #start  = (5,6); 
    pathM = [(3, 4) for i in range(len(path))]
    print(pathM)
    mz.animate_solution(maze, path, pathM)




if __name__ == "__main__":
    main()