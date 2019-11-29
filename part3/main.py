import numpy as np
import time, pickle, os
import maze as mz


def evaluate_v(env, solution_policy=None):

    start_new  = (0,0)
    start_newM = (3,3)

    state = env.map[start_new]
    stateM = env.mapM[start_newM]

    t_max = 10000
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



    iter_max = 1000000
    min_lr = 0.003
    gamma = 0.8
    eps = 1
    

    r = env.rewards

    print ('----- using Q Learning -----')
    q_table = np.zeros((16, 16, 5))

    start  = (0, 0)
    startM = (3, 3)

    state = env.map[start]
    stateM = env.mapM[startM]

    for i in range(1, iter_max+1):

        eta = max(min_lr, 1/(i**(2/3)))

        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(len(env.actions.keys()))

        state_new = env.move(state, action)
        state_newM = env.moveM(stateM)

        reward = get_reward(state, state_new, state_newM, action, r)
     
   
        # update q table
        q_before = q_table[state][stateM][action]
        q_table[state][stateM][action] =  q_before + eta * (reward + gamma *  np.max(q_table[state_new][state_newM]) - q_before)

   
        state = state_new
        stateM = state_newM

        

        if i%(iter_max//1000) == 0:
            solution_policy = np.argmax(q_table, axis=2)
            score = evaluate_v(env, solution_policy)
            #print(solution_policy)
            print('Total reward of current best policy = %d.' %score)




if __name__ == '__main__':
    main()










"""
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            logits = q_table[a][b]
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            action = np.random.choice(env.action_space.n, p=probs)
"""






"""
env = gym.make('FrozenLake-v0')

epsilon = 0.9
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01

total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
rewards=0

for episode in range(total_episodes):
    t = 0
    state = env.reset()
    action = choose_action(state)

    while t < max_steps:
        env.render()

        state2, reward, done, info = env.step(action)

        action2 = choose_action(state2)

        learn(state, state2, reward, action, action2)

        state = state2
        action = action2

        t += 1
        rewards+=1

        if done:
            break
  # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
  # os.system('clear')
        time.sleep(0.1)


print ("Score over time: ", rewards/total_episodes)
print(Q)

with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
    pickle.dump(Q, f)
    """
