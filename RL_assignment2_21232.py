import numpy as np

#RL environment
class gridworld:
    def __init__(self, dims, goal_states, error_states):
        # for actions
        self.actions = ["up", "down", "right", "left"]
        self.action_map = {"up": 0, "down":1, "right": 2, "left": 3}
        self.movement = [[-1,0], [1,0], [0,1], [0,-1]]	

        # for states
        self.rows = dims[0]
        self.columns = dims[1]
        self.current_state = None
        self.error_states = error_states 
        self.goal_states = goal_states
        self.empty_states =  [(i, j) for i in range(self.rows) for j in range(self.columns) if (i, j) not in (self.goal_states + self.error_states)]

        # visual representation of gridworld
        self.grid = np.array([["." for i in range(self.columns)] for j in range(self.rows)])
        for g in self.goal_states:
            self.grid[g] = "O"
        for e in self.error_states:
            self.grid[e] = "X"

    #print gridworld
    def print_grid(self):				
        print(self.grid)

    #set current state to start state
    def reset(self, start_state = None):
        if start_state != None:
             self.current_state = start_state
        else:
            self.current_state =  self.empty_states[np.random.choice(len(self.empty_states))]
        return self.current_state

    def step(self, action): 
        action_index = action
        action_taken = self.actions[action]  
        
        new_state, reward  = (-1, -1), 0 
        
        # out of the board , no movement but -reward for taking action
        if (action_taken == "up" and self.current_state[0] == 0) or (action_taken == "down" and self.current_state[0] == self.rows-1) :
            raise Exception("Invalid action taken")
        elif (action_taken == "left" and self.current_state[1] == 0) or (action_taken == "right" and self.current_state[1] == self.columns-1) :
            raise Exception("Invalid action taken")

        # inside the board
        else:
            new_state = (self.current_state[0] + self.movement[action_index][0] , self.current_state[1] + self.movement[action_index][1])
            if new_state in self.empty_states: # no reward in empty states 
                reward = -1
            elif new_state in self.goal_states: # good state
                reward = 10
            elif new_state in self.error_states: # bad state
                reward = -5 

        # current state changes because of taking action 
        self.current_state = new_state  

        #return current state, reward 
        return self.current_state, reward

class Solution:
    def __init__(self, env, gamma=0.9, epsilon=0.1, maxIter=5000, maxTimesteps = 100):
        self.q_table = {}
        self.env = env
        self.gamma, self.epsilon, self.maxIter, self.maxTimesteps = gamma, epsilon, maxIter, maxTimesteps
        ####### You may add extra functions/ variables in this class

    # Define the Q-learning algorithm
    def q_learning(self):
        q_table = np.zeros((self.env.rows, self.env.columns, len(self.env.actions)))
 
        for _ in range(self.maxIter):
            state = self.env.reset()
            # to calculate alpha for each iteration, alpha = 1 / visit_n(s, a)
            visit_n = {}

            ########## Complete this function #######
            q_table_old = np.copy(q_table)

            for _ in range(self.maxTimesteps):

                # we'll pick a random action for obtaining the next s'
                action = np.random.choice(self.env.actions)
                a = self.env.action_map[action]

                # out of the board, no movement but -reward for taking action
                if (action == "up" and state[0] == 0) or (action == "down" and state[0] == self.env.rows-1) :
                    next_state, reward = state, -5
                elif (action == "left" and state[1] == 0) or (action == "right" and state[1] == self.env.columns-1) :
                    next_state, reward = state, -5
                else:
                    # Take action a, observe r, s'
                    next_state, reward = self.env.step(a)
        
                # Q(s,a) <- (1-alpha)Q(s,a) + alpha[r + gamma * max_a' Q(s',a')]
                visit_n[(state, a)] = visit_n.get((next_state, a), 0) + 1
                alpha = 1 / visit_n[(state, a)]
                q_table[state][a] = (1-alpha) * q_table[state][a] + alpha * (reward + self.gamma * np.max(q_table[next_state]))

                state = next_state
        
                # print(state, action, reward, next_state)
                # print("---------")

            #############################

        return q_table 


if __name__ == "__main__":
    np.random.seed(100)

    size = 5
    goal_states = [(0, size-1), (1, size-1), (size-1, 0)] 
    error_states = [(o, o) for o in range(size)] 
    grid = gridworld((size, size), goal_states, error_states)

    grid.print_grid()
    solution = Solution(grid)
    q_table = solution.q_learning()

    print("Q-table:")
    print(q_table)


    if size == 3:
        assert round(q_table[0,0,1], 2)== 75.5 and round(q_table[0,1,2], 2)==100.0, 'wrong answer'
    if size == 5:
        assert round(q_table[2,0,1], 5)== 45.54251, 'wrong answer'