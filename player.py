import numpy as np
from qTable import Rewards
from stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace):
    """
    Q-learning agent for playing Ludo game.

    Attributes:
        ai_player_idx (int): Index of the AI player.
        debug (bool): Flag indicating whether to enable debug mode.
        q_learning (Rewards): Q-learning rewards object.
        state (object): Current state of the game.
        action (object): Current action taken by the agent.

    Methods:
        __init__(self, ai_player_idx, gamma=0.3, learning_rate=0.2): Initializes the QLearningAgent object.
        update(self, players, pieces_to_move, dice): Updates the agent's state and chooses the next action.
        reward(self, players, pieces_to_move): Calculates the reward for the agent based on the current state and action.
    """

    ai_player_idx = -1
    debug = False
    q_learning = None
    state = None
    action = None

    def __init__(self, ai_player_idx, gamma=0.3, learning_rate=0.2):
        """
        Initializes the QLearningAgent object.

        Args:
            ai_player_idx (int): Index of the AI player.
            gamma (float): Discount factor for future rewards (default: 0.3).
            learning_rate (float): Learning rate for updating Q-values (default: 0.2).
        """
        super().__init__()
        self.q_learning = Rewards(len(State), len(Action), gamma=gamma, lr=learning_rate)
        self.ai_player_idx = ai_player_idx

    def update(self, players, pieces_to_move, dice):
        """
        Updates the agent's state and chooses the next action.

        Args:
            players (list): List of players in the game.
            pieces_to_move (int): Number of pieces to move.
            dice (int): Value of the dice roll.

        Returns:
            int: Number of pieces to move.
        """
        super().update(players, self.ai_player_idx, pieces_to_move, dice)
        action_table = self.action_table_player.get_action_table()
        state, action = self.q_learning.choose_next_action(self.ai_player_idx, action_table)
        pieces_to_move = self.action_table_player.get_piece_to_move(state, action)
        self.state = state
        self.action = action

        return pieces_to_move

    def reward(self, players, pieces_to_move):
        """
        Calculates the reward for the agent based on the current state and action.

        Args:
            players (list): List of players in the game.
            pieces_to_move (int): Number of pieces to move.

        Returns:
            None
        """
        super().get_possible_actions(players, self.ai_player_idx, pieces_to_move)
        new_action_table = np.nan_to_num(self.action_table_player.get_action_table(), nan=0.0)
        self.q_learning.reward(self.state, new_action_table, self.action)
