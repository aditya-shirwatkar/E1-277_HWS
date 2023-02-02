# -*- coding: utf-8 -*-
"""E1-277 Reinforcement Learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IK6dXFgGwDaeOJnzompYCjTr43XRVXmL
"""

import numpy as np
import math

class Maze:
    def __init__(self, gridHeight=6, gridWidth=6, terminalReward=10, lockPickProb=0.5):
        self.rewardsLeft = np.array([[-1, 0, 0, 0, 0, 0], 
                                    [-1, -1, 0, 0, 0,-10], 
                                    [-1, 0, 0, -1, -1, -1], 
                                    [0, 0, 0, -10, -1, -1],
                                    [-1, -1, 0, 0, -1, 0],
                                    [-1, 0, -1, 0, 0 ,-1]])

        self.rewardsRight =  np.array([[ 0, 0, 0, 0, 0, -1], 
                            [ -1, 0, 0 , 0, -10, -1],
                            [ 0, 0, -1, -1, -1, -1],
                            [ 0, 0, -10, -1, -1 ,-1],
                            [ -1, 0, 0, -1, 0, -1],
                            [ 0, -1, 0, 0, -1, -1]])

        self.rewardsUp  =  np.array([[ -1, -1, -1, -1, -1, -1], 
                            [ 0, -1, -1, -1, -1, 0],
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0,0, 0,0, 0],
                            [ 0, -10, -1, -1, -1, 0],
                            [ 0,  0, -1, -10, 0, 0]])


        self.rewardsDown =  np.array([[ 0, -1, -1, -1, -1, 0], 
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0, 0, 0, 0, 0],
                            [ 0, -10,-1,-1,-1, 0],
                            [  0,0,-1,-10,0, 0],
                            [ -1, -1, -1, 0, -1, -1]])

        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.lockPickProb = lockPickProb
        self.terminalReward = terminalReward


    def isStateTerminal(self, state):
        if state == (3, 0) :
            return True
        elif state == (5, 3):
            return True
        return False

    def takeAction(self, state, action):
        retVal = []
        if(self.isStateTerminal(state)):
            return [[state,1, self.terminalReward]] 

        if action=='left':
            reward = self.rewardsLeft[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]-1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]-1),1,-1])

        if action=='right':
            reward = self.rewardsRight[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]+1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]+1),1,-1])

        if action=='up':
            reward = self.rewardsUp[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]-1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]-1, state[1]),1,-1])

        if action=='down':
            reward = self.rewardsDown[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]+1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]+1, state[1]),1,-1])
        for i,[nextState, prob, reward] in enumerate(retVal):
            if(self.isStateTerminal(nextState)):
                retVal[i][2] = self.terminalReward   

        return retVal 

class GridworldSolution:
    def __init__(self, maze,horizonLength):
        self.env = maze
        self.actionSpace = ['left', 'right', 'up',  'down']
        self.horizonLength = horizonLength
        self.DP = np.ones((self.env.gridHeight, self.env.gridWidth, self.horizonLength)) * -np.inf
    
    def optimalReward(self, state, k):
        optReward = -np.inf
        rows = self.env.gridHeight
        cols = self.env.gridWidth

        #### Write your code here
        
        # Go backwards in time
        for t in range(self.horizonLength-1, -1, -1):
          
          # Iterate through all the states
          for i in range(rows):
            for j in range(cols):

              # Define expected reward at the end of the Horizon
              if t == (self.horizonLength-1):
                expected_rewards = []

                # Iterate through each action and get the corresponding expected reward
                for a in self.actionSpace:
                  retVal = self.env.takeAction((i,j), a)
                  expected_reward = 0
                  for _ , [nextState, prob, reward] in enumerate(retVal):
                    expected_reward += prob * reward
                  expected_rewards.append(expected_reward)
                max_reward = max(expected_rewards)

                self.DP[i,j, t] = max_reward

              else:
                expected_rewards = []
                for a in self.actionSpace:
                  retVal = self.env.takeAction((i,j), a)
                  expected_reward = 0
                  for _ , [nextState, prob, reward] in enumerate(retVal):
                    i_prime, j_prime = nextState
                    expected_reward += prob * (reward + self.DP[i_prime,j_prime, t+1]) # Dynamic Programming step
                  expected_rewards.append(expected_reward)
                max_reward = max(expected_rewards)

                self.DP[i,j, t] = max_reward

        optReward = self.DP[state[0], state[1], k]

        ########

        return optReward

if __name__ == "__main__":
    maze = Maze()
    solution = GridworldSolution(maze,horizonLength=5)
    print(" Horizon ",solution.horizonLength)
    optReward = solution.optimalReward((2,0),0)
    assert optReward==28.0, 'wrong answer'
    optReward = solution.optimalReward((4,3),0)
    assert optReward==39.34375, 'wrong answer'

