class GameInterface:
    def reset(self):
        """Reset the game to its initial state."""
        raise NotImplementedError

    def get_state(self, player_id):
        """Return the current state for the given player."""
        raise NotImplementedError

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        raise NotImplementedError

    def step(self, player_id, action):
        """
        Apply the player's action to the game.

        Returns:
            state (object): The new state after the action.
            reward (float): The reward received after the action.
            done (bool): Whether the game has ended.
            info (dict): Additional information.
        """
        raise NotImplementedError

    def is_over(self):
        """Check if the game has ended."""
        raise NotImplementedError

    def get_winner(self):
        """Determine the winner of the game."""
        raise NotImplementedError