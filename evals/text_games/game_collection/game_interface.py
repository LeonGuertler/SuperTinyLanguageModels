class GameInterface:
    def reset(self):
        """Reset the game to its initial state."""
        # TODO return prompt + types
        raise NotImplementedError

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        raise NotImplementedError

    def get_info(self):
        """Return additional info after the game"""
        raise NotImplementedError

    def step(self, player_id, action):
        """
        Apply the player's action to the game.

        Returns:
            state (str): The new state after the action.
            reward (None/dict): The reward received after the game ends.
            done (bool): Whether the game has ended.
            info (dict): Additional information.
        """
        # rename as observation
        # optinally return state (used to re-load game at checkpoint)
        raise NotImplementedError
