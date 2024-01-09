import os.path
import random

import numpy as np
from stateSpace import Action


class Rewards():
    """
    Class representing the rewards and Q-learning algorithm for the Ludo game.
    """

    rewards_table = np.zeros(len(Action))
    q_table = None
    epoch = 0
    iteration = 0

    def __init__(self, states, actions, epsilon=0.9, gamma=0.3, lr=0.2, learning=True):
        """
        Initializes the Rewards object.

        Args:
            states (int): The number of states in the game.
            actions (int): The number of actions in the game.
            epsilon (float, optional): The exploration rate for epsilon-greedy policy. Defaults to 0.9.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.3.
            lr (float, optional): The learning rate for updating Q-values. Defaults to 0.2.
            learning (bool, optional): Whether the agent is in learning mode or not. Defaults to True.
        """
        super().__init__()
        self.learning = learning
        
        self.q_table = np.zeros([states, actions])

        self.epsilon_greedy = epsilon
        self.gamma = gamma
        self.lr = lr

        self.max_expected_reward = 0
        
        # Define rewards for each action
        VERY_BAD = -0.8
        BAD = -0.4
        GOOD = 0.4
        VERY_GOOD = 1.2

        self.rewards_table[Action.SAFE_MoveOut.value] = 0.4
        self.rewards_table[Action.SAFE_MoveDice.value] = 0.01
        self.rewards_table[Action.SAFE_Goal.value] = 0.8
        self.rewards_table[Action.SAFE_Star.value] = 0.8
        self.rewards_table[Action.SAFE_Globe.value] = 0.4
        self.rewards_table[Action.SAFE_Protect.value] = 0.2
        self.rewards_table[Action.SAFE_Kill.value] = 1.5
        self.rewards_table[Action.SAFE_Die.value] = -0.5
        self.rewards_table[Action.SAFE_GoalZone.value] = 0.2

        self.rewards_table[Action.UNSAFE_MoveOut.value] = self.rewards_table[Action.SAFE_MoveOut.value] + BAD
        self.rewards_table[Action.UNSAFE_MoveDice.value] = self.rewards_table[Action.SAFE_MoveDice.value] + BAD
        self.rewards_table[Action.UNSAFE_Star.value] = self.rewards_table[Action.SAFE_Star.value] + BAD
        self.rewards_table[Action.UNSAFE_Globe.value] = self.rewards_table[Action.SAFE_Globe.value] + GOOD
        self.rewards_table[Action.UNSAFE_Protect.value] = self.rewards_table[Action.SAFE_Protect.value] + GOOD
        self.rewards_table[Action.UNSAFE_Kill.value] = self.rewards_table[Action.SAFE_Kill.value] + GOOD
        self.rewards_table[Action.UNSAFE_Die.value] = self.rewards_table[Action.SAFE_Die.value] + VERY_BAD
        self.rewards_table[Action.UNSAFE_GoalZone.value] = self.rewards_table[Action.SAFE_GoalZone.value] + GOOD
        self.rewards_table[Action.UNSAFE_Goal.value] = self.rewards_table[Action.SAFE_Goal.value] + GOOD

        self.rewards_table[Action.HOME_MoveOut.value] = self.rewards_table[Action.SAFE_MoveOut.value] + VERY_GOOD
        self.rewards_table[Action.HOME_MoveDice.value] = self.rewards_table[Action.SAFE_MoveDice.value] + VERY_BAD
        self.rewards_table[Action.HOME_Star.value] = self.rewards_table[Action.SAFE_Star.value] + VERY_BAD
        self.rewards_table[Action.HOME_Globe.value] = self.rewards_table[Action.SAFE_Globe.value] + VERY_BAD
        self.rewards_table[Action.HOME_Protect.value] = self.rewards_table[Action.SAFE_Protect.value] + VERY_BAD
        self.rewards_table[Action.HOME_Kill.value] = self.rewards_table[Action.SAFE_Kill.value] + VERY_BAD
        self.rewards_table[Action.HOME_Die.value] = self.rewards_table[Action.SAFE_Die.value] + VERY_BAD
        self.rewards_table[Action.HOME_GoalZone.value] = self.rewards_table[Action.SAFE_GoalZone.value] + VERY_BAD
        self.rewards_table[Action.HOME_Goal.value] = self.rewards_table[Action.SAFE_Goal.value] + VERY_BAD

    def update_epsilon(self, new_epsilon):
        """
        Updates the exploration rate (epsilon) for epsilon-greedy policy.

        Args:
            new_epsilon (float): The new exploration rate.
        """
        self.epsilon_greedy = new_epsilon

    def get_state_action_of_array(self, value, array):
        """
        Returns the state and action corresponding to a given value in the action table.

        Args:
            value (float): The value to search for in the action table.
            array (numpy.ndarray): The action table.

        Returns:
            tuple: The state and action corresponding to the value.
        """
        if np.isnan(value):
            return (-1, -1)
        idx = np.where(array == value)
        random_idx = random.randint(0, len(idx[0]) - 1)
        state = idx[0][random_idx]
        action = idx[1][random_idx]
        return (state, action)

    def choose_next_action(self, player, action_table):
        """
        Chooses the next action for the player based on the current action table.

        Args:
            player: The player for whom to choose the next action.
            action_table (numpy.ndarray): The current action table.

        Returns:
            tuple: The state and action of the chosen next action.
        """
        q_table_options = np.multiply(self.q_table, action_table)
    
        if random.uniform(0, 1) < self.epsilon_greedy:
            self.iteration = self.iteration + 1
            nz = action_table[np.logical_not(np.isnan(action_table))]
            randomValue = nz[random.randint(0, len(nz) - 1)]
            state, action = self.get_state_action_of_array(randomValue, action_table)
        else:
            maxVal = np.nanmax(q_table_options)
            if not np.isnan(maxVal):
                state, action = self.get_state_action_of_array(maxVal, q_table_options)
            else:
                nz = action_table[np.logical_not(np.isnan(action_table))]
                random_value = nz[random.randint(0, len(nz) - 1)]
                state, action = self.get_state_action_of_array(random_value, action_table)
        return (state, action)


    def reward(self, state, new_action_table, action):
        """
        Updates the Q-table based on the reward received for taking an action in a state.

        Args:
            state (int): The current state.
            new_action_table (numpy.ndarray): The updated action table after taking the action.
            action (int): The action taken in the current state.
        """
        state = int(state)
        action = int(action)

        # Q-learning equation
        reward = self.rewards_table[action]
        # Q-learning
        estimate_of_optimal_future_value = np.max(self.q_table * new_action_table)
        old_q_value = self.q_table[state, action]
        delta_q = self.lr * (reward + self.gamma * estimate_of_optimal_future_value - old_q_value)
        
        self.max_expected_reward += reward
        
        # Update the Q table from the new action taken in the current state
        self.q_table[state, action] = old_q_value + delta_q
        # print("update q table, state: {0}, action:{1}".format(state,action))
    