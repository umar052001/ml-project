import numpy as np


class ActionTableEntry():
    """
    Represents an entry in the action table.

    Attributes:
        piece (int): The piece number.
        value (float): The value associated with the piece.
    """

    def __init__(self, piece, value):
        super().__init__()
        self.piece = piece
        self.value = value

    def add_entry(self, piece, value):
        """
        Adds an entry to the action table.

        Args:
            piece (int): The piece number.
            value (float): The value associated with the piece.
        """
        self.piece.append(piece)
        self.value.append(value)



class ActionTable():
    """
    Represents the action table for a Ludo game.

    Attributes:
        action_table (numpy.ndarray): The action table that maps states to actions.
        piece_to_move (numpy.ndarray): The table that maps states and actions to pieces.
        state (int): The current state.

    Methods:
        __init__(self, states, actions): Initializes the ActionTable instance.
        set_state(self, state): Sets the current state.
        get_action_table(self): Returns the action table.
        get_piece_to_move(self, state, action): Returns the piece to move for a given state and action.
        reset(self): Resets the action table and piece table.
        update_action_table(self, action, piece, value): Updates the action table with a new value for a given action and piece.
    """

    def __init__(self, states, actions):
        """
        Initializes the ActionTable instance.

        Args:
            states (int): The number of states.
            actions (int): The number of actions.
        """
        super().__init__()
        self.states = states
        self.actions = actions
        self.reset()

    def set_state(self, state):
        """
        Sets the current state.

        Args:
            state (int): The current state.
        """
        self.state = state.value

    def get_action_table(self):
        """
        Returns the action table.

        Returns:
            numpy.ndarray: The action table.
        """
        return self.action_table

    def get_piece_to_move(self, state, action):
        """
        Returns the piece to move for a given state and action.

        Args:
            state (int): The state.
            action (int): The action.

        Returns:
            int: The piece to move.
        """
        if state < 0 or action < 0:
            return -1
        return int(self.piece_to_move[state, action])

    def reset(self):
        """
        Resets the action table and piece table.
        """
        self.action_table = np.full((self.states, self.actions), np.nan)
        self.piece_to_move = np.full((self.states, self.actions), np.nan)

    def update_action_table(self, action, piece, value):
        """
        Updates the action table with a new value for a given action and piece.

        Args:
            action (int): The action.
            piece (int): The piece.
            value: The new value for the action and piece.
        """
        if np.isnan(self.action_table[self.state, action.value]):
            self.action_table[self.state, action.value] = 1
            self.piece_to_move[self.state, action.value] = piece