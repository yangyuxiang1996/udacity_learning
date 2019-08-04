import numpy as np
from collections import defaultdict
"""
Epsilon: 0.0001, Alpha = 0.2, Saramax
Episode 20000/20000 || Best average reward 9.313

Epsilon: 0.0001, Alpha = 0.2, Sarmax
Episode 20000/20000 || Best average reward 9.419
"""

class Agent:

    def __init__(self, nA=6, gamma=1.0, alpha=0.2, epsilon=0.0001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        print('Epsilon: {}, Alpha = {}'.format(self.epsilon ,self.alpha) )


    def epsilon_greedy_probs(self, Q_s, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s 

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        state_policy = self.epsilon_greedy_probs(self.Q[state], self.epsilon)
        action = np.random.choice(np.arange(self.nA), p=state_policy)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        old_Q = self.Q[state][action]

        """ Sarsamax """
        # self.Q[state][action] = old_Q + (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state]) - old_Q)))
        """ Sarsa """
        next_action = self.select_action(next_state)
        self.Q[state][action] = (1 - self.alpha)*old_Q + self.alpha*(reward + self.gamma * self.Q[next_state][next_action])
        """ 预期Sarsa """
        # probs = self.epsilon_greedy_probs(self.Q[state], self.epsilon)
        # self.Q[state][action] = (1 - self.alpha)*old_Q + self.alpha*(reward + self.gamma * np.sum(probs * self.Q[next_state]))
        

        # self.Q[state][action] = reward + self.Q[next_state][next_action]
