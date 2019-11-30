from maze import *
import numpy as np




def get_policy_and_pathM(env, horizon, start, startM, weights):

    T = horizon    
    pathM = []    

    #Position of player
    state_new = env.map[start]

    #Position of minotaur
    stateM_new = env.mapM[startM]


    for t in range(0, T):

        #Control if one step between
        row = env.states[state_new][0]
        col = env.states[state_new][1]

        rowM = env.statesM[stateM_new][0]
        colM = env.statesM[stateM_new][1]


        pathM.append(env.statesM[stateM_new])


        stateM_new, valid_movesM =  env.moveM(stateM_new)        
        

        if (abs(col - colM) + abs(row - rowM)) == 2:                         
            dangerous_player_pos = state_new       
            V, policy = dynamic_programming(env, horizon, weights, valid_movesM, dangerous_player_pos, t)            
        else:
            V, policy = dynamic_programming(env, horizon, weights)

        #print(env.rewards)


        if t == 0:
            policy_main = policy[:,t]
        else:
            policy_main = np.vstack((policy_main, policy[:,t]))

        state_new = env.move(state_new, policy[state_new,t])

    return policy_main.T, pathM










def dynamic_programming(env, horizon, weights=None, valid_movesM=None, dangerous_player_pos=None, player_t=None):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon    

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))

    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)    

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            if (dangerous_player_pos is not None) and (dangerous_player_pos == s) and player_t == t:                
                env.rewards = env.setRewards(weights, valid_movesM)                
            else:
                env.rewards = env.setRewards(weights)              
            for a in range(n_actions):
                r = env.rewards           
                # Update of the temporary Q val ues
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);        
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)    
    return V, policy




def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path, pathM):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):

        if i > 0:
            #if path[i] == path[i-1]:
            #    grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
            #    grid.get_celld()[(path[i])].get_text().set_text('Player is out')
            #else:
            grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
            grid.get_celld()[(path[i-1])].get_text().set_text('')

            grid.get_celld()[(pathM[i-1])].set_facecolor(col_map[maze[pathM[i-1]]])
            grid.get_celld()[(pathM[i-1])].get_text().set_text('')

            #grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
            #grid.get_celld()[(path[i-1])].get_text().set_text('')

        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')


        grid.get_celld()[(pathM[i])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(pathM[i])].get_text().set_text('Minotaur')


        plt.draw()
        plt.pause(0.5)
        display.clear_output(wait=True)

    plt.show();
