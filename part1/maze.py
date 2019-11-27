import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.actionsM                 = self.__actionsM();
        self.states, self.statesM, \
        self.map, self.mapM           = self.__states();
        self.n_actions                = len(self.actions);
        self.n_actionsM               = len(self.actions);
        self.n_states                 = len(self.states);
        self.n_statesM                = len(self.statesM);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.setRewards(weights=weights,

                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;



    def __actionsM(self):
        actions = dict();
        #actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;



    def __states(self):
        states = dict();
        statesM = dict();
        mapM = dict();
        map = dict();
        end = False;
        s = 0;
        sM = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    states[s] = (i,j);
                    map[(i,j)] = s;
                    s += 1;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                statesM[sM] = (i,j);
                mapM[(i,j)] = sM;
                sM += 1;
        return states, statesM, map, mapM



    def move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            return self.map[(row, col)];


    def moveM(self, state):

        valid_moves = []

        for a in self.actionsM.keys():
            row = self.statesM[state][0] + self.actionsM[a][0];
            col = self.statesM[state][1] + self.actionsM[a][1];
            # Is the future position an impossible one ?
            hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                                  (col == -1) or (col == self.maze.shape[1])
            if not hitting_maze_walls:
                valid_moves.append(self.mapM[(row, col)])

        new_stateM = np.random.choice(valid_moves)

        return new_stateM, valid_moves




    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.move(s,a);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;





    def setRewards(self, weights=None, random_rewards=None, valid_movesM=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.move(s,a);
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:

            if valid_movesM != None:
                minotaurMoves = []
                for valid_move in valid_movesM:
                    minotaurMoves.append(self.statesM[valid_move])

            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.

                     if valid_movesM != None:
                         if (i,j) in minotaurMoves:
                             rewards[s,a] = -100 / len(valid_movesM)
                         else:
                             rewards[s,a] = weights[i][j];
                     else:
                        rewards[s,a] = weights[i][j];
            '''
            if valid_movesM != None:
                for valid_move in range(valid_movesM):
                    i, j = self.statesM[valid_move]
                    weights[i][j] = -60 / len(valid_movesM)
            '''




        return rewards;





    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            print(policy)
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)
