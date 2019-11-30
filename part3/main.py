import numpy as np
import time, pickle, os
import maze as mz
from plotting import *




def evaluate_v(env, solution_policy=None):

    start_new  = (0,0)
    start_newM = (3,3)

    state = env.map[start_new]
    stateM = env.mapM[start_newM]

    t_max = 1000
    total_reward = 0

    r = env.rewards

    for _ in range(t_max):       

        action = solution_policy[state][stateM]

        state_new = env.move(state, action)
        state_newM = env.moveM(stateM)

    
        reward = get_reward(state, state_new, state_newM, action, r)
    

        total_reward += reward

        state = state_new
        stateM = state_newM

    return total_reward




def get_reward(state, state_new, state_newM, action, r):
    if state_new == state_newM:         
        return -10
           
    else:
        return r[state, action]





def main():
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        ])

    weights = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        ])

    env = mz.Maze(maze, weights=weights)



    iter_max = 10000000
    min_lr = 0.0001
    gamma = 0.8 
    SARSA = True
    Q = False
    epsilon = 0.05
    epsQ = 1
    

    r = env.rewards

    print ('----- using Q Learning -----')
    q_table = np.zeros((16, 16, 5))
    q_updates = np.ones(q_table.shape)

    start  = (0, 0)
    startM = (3, 3)

    state = env.map[start]
    stateM = env.mapM[startM]

    time_vector = []
    score_vector = []
    score_vector2 = []
    score_vector3 = []

    for i in range(1, iter_max+1):


        if Q:
            if np.random.uniform(0, 1) < epsQ:
                action = np.random.choice(len(env.actions.keys()))

            state_new = env.move(state, action)
            state_newM = env.moveM(stateM)

            reward = get_reward(state, state_new, state_newM, action, r)        
    
            # update q table
            n = q_updates[state][stateM][action]
            eta = max(min_lr, 1/(n**(2/3)))
            q_updates[state][stateM][action] += 1

            q_before = q_table[state][stateM][action]
            q_table[state][stateM][action] =  q_before + eta * (reward + gamma *  np.max(q_table[state_new][state_newM]) - q_before)



        if SARSA:
            if np.random.uniform(0, 1) < epsilon:
                action1 = np.random.choice(len(env.actions.keys()))
            else:
                action1 = np.argmax(q_table[state][stateM])

            if np.random.uniform(0, 1) < epsilon:
                action2 = np.random.choice(len(env.actions.keys()))
            else:
                action2 = np.argmax(q_table[state][stateM]) 


            state_new = env.move(state, action1)
            state_newM = env.moveM(stateM)

            reward = get_reward(state, state_new, state_newM, action1, r)  

            n = q_updates[state][stateM][action1]
            eta = max(min_lr, 1/(n**(2/3)))
            q_updates[state][stateM][action1] += 1

            q_before = q_table[state][stateM][action1]     

            q_table[state][stateM][action1] = q_before + eta * (reward + gamma * (q_table[state_new][state_newM][action2]) - q_before)

   
        state = state_new
        stateM = state_newM

        

        if i % 100 == 0: #i%(iter_max//100) == 0:            
            
            solution_policy = np.argmax(q_table, axis=2)

            #score = evaluate_v(env, solution_policy)
            #print(score)            

           
            start  = (0,0)    
            startM = (3,3)
            

            start2 = (1,1)
            start2M = (3,3)

            start3 = (1,1)
            start3M = (1,0)

            

            state2 = env.map[start2]
            state2M = env.mapM[start2M]

            state3 = env.map[start3]
            state3M = env.mapM[start3M]


            s2 = solution_policy[state2][state2M]
            score2 = q_table[state2][state2M][s2]

            s3 = solution_policy[state3][state3M]
            score3 = q_table[state3][state3M][s3]




            state = env.map[start]
            stateM = env.mapM[startM]

            initial_state_opt_action = solution_policy[state][stateM]
            score = q_table[state][stateM][initial_state_opt_action]
            
            time_vector.append(i)            
            score_vector.append(score)
            score_vector2.append(score2)
            score_vector3.append(score3)
        
            

  
    plotResult("Q-value of optimal action in state S with eps:" + str(epsilon), "Iteration", "Value", time_vector, score_vector)
    #plotResult("Q*(initial state)", "Iteration", "Value", time_vector, score_vector)
    #plotResult("Q*(initial state)", "Iteration", "Value", time_vector, score_vector)
    if Q:
        multiplot("Q-value of optimal action in state S", "Iteration", "Value", time_vector, (score_vector, score_vector2, score_vector3), ["Player"+str(start)+", "+"Police"+str(startM), "Player"+str(start2)+", "+"Police"+str(start2M), "Player"+str(start3)+", "+"Police"+str(start3M)])
    if SARSA:
        multiplot("Q-value of optimal action in state S  with eps:" + str(epsilon) , "Iteration", "Value", time_vector, (score_vector, score_vector2, score_vector3), ["Player"+str(start)+", "+"Police"+str(startM), "Player"+str(start2)+", "+"Police"+str(start2M), "Player"+str(start3)+", "+"Police"+str(start3M)])



if __name__ == '__main__':
    main()






