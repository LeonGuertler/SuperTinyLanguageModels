import chess
from evals.text_games.game_collection.game_interface import GameInterface

class ChessGame(GameInterface):
    def __init__(self, is_open=True, render=False, turn_limit=30, show_valid=True):
        """
        Initialize the Chess game.
        Args:
            is_open (bool): If True, both players can see the current board state.
                            If False, the players do not receive the current board state.
            render (bool): If True, the game will display the board in the terminal after each move.
        """
        self.name = "Chess"
        self.is_open = is_open
        self.show_valid = show_valid
        self.render = render
        self.turn_limit = turn_limit
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.board = chess.Board()
        self.game_over = False
        self.turn = 0  # 0 for White, 1 for Black

        # Return initial prompts for both players
        player_prompts = {
            0: "You are playing White in a game of chess. Make your move in UCI format (e.g., e2e4).",
            1: "You are playing Black in a game of chess. Make your move in UCI format (e.g., e7e5)."
        }
        return player_prompts

    def get_valid_actions(self, player_id):
        """Return a list of valid moves for the current player in UCI format."""
        if self.show_valid and (self.board.turn == chess.WHITE and player_id == 0) or (self.board.turn == chess.BLACK and player_id == 1):
            return [move.uci() for move in self.board.legal_moves]
        else:
            return []

    def get_info(self):
        """Return additional information, e.g., number of turns."""
        return {'num_turns': self.board.fullmove_number}

    def step(self, player_id, action):
        """
        Process the player's move.
        Args:
            player_id (int): The player's ID (0 for White, 1 for Black).
            action (str): The move in UCI notation.
        Returns:
            state (str): The new state after the move.
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        # Ensure it's the correct player's turn
        if (self.board.turn == chess.WHITE and player_id != 0) or (self.board.turn == chess.BLACK and player_id != 1):
            return None, None, False, {"info": "It's not your turn."}

        current_turn = "White" if self.board.turn == chess.WHITE else "Black"

        # Try to make the move
        try:
            move = chess.Move.from_uci(action.strip())
            if move in self.board.legal_moves:
                self.board.push(move)

                # Render the board if rendering is enabled
                if self.render:
                    print("\nCurrent Board State:")
                    print(self.board)

                # Check for game over conditions
                if self.board.is_game_over():
                    result = self.board.result()
                    if result == '1-0':
                        reward = {0: 1, 1: -1}
                    elif result == '0-1':
                        reward = {0: -1, 1: 1}
                    else:
                        reward = {0: 0, 1: 0}  # Draw
                    return None, reward, True, {"result": result, "reason": "Game Finished"}
                elif self.board.fullmove_number > self.turn_limit:
                    # draw by too long
                    return None, {0:0, 1:0}, True, {"reason": "Too many turns."}
                else:
                    # Prepare the next state
                    self.turn = 1 - player_id
                    if self.is_open:
                        # Return the board_state
                        state = f"Current board state: {self.board}"
                    else:
                        # Return the previous agent's move
                        state = f"{current_turn} moved: {action}"
                    return state, None, False, {"info": "Move accepted."}
            else:
                # Invalid move; Agent that did invalid move gets -1, other agent 0
                return None, {player_id: -1, 1 - player_id: 0}, True, {"reason": "Invalid move."}
        except ValueError:
            # Invalid move format; Agent that did invalid move gets -1, other agent 0
            return None, {player_id: -1, 1 - player_id: 0}, True, {"reason": "Invalid move format."}
