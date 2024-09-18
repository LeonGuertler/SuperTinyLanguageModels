from evals.text_games.game_collection.game_interface import GameInterface

class ConnectFourGame(GameInterface):
    def __init__(self, 
        is_open=True,
        render=False,
    ):
        """
        Initialize the Connect Four game.
        Args:
            is_open (bool): If True, the game state is visible to the players.
            render (bool): If True, print the board state at every step.
        """
        self.name = "Connect Four"
        self.render = render
        self.rows = 6
        self.cols = 7
        self.is_open = is_open

    def reset(self):
        """Reset the game to its initial state."""
        self.board = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        self.turn_counter = 0

        # Prepare initial prompts
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }
        
        if self.render:
            print("Game reset.")
            print(self._render_board())

        return player_prompts

    def _generate_player_prompt(self, player_id):
        prompt = (
            f"You are Player {player_id}.\n"
            f"Your disc symbol: {'X' if player_id == 0 else 'O'}.\n"
            "Take turns dropping your disc into one of the columns (0-6).\n"
            "First to connect four discs vertically, horizontally, or diagonally wins.\n"
            "Enter the column number to make your move."
        )
        return prompt

    def get_valid_actions(self, player_id):
        """Return valid column numbers where the player can drop a disc."""
        valid_actions = [str(c) for c in range(self.cols) if self.board[0][c] == '.']
        return valid_actions

    def get_info(self):
        """Return additional information."""
        return {'num_turns': self.turn_counter}

    def step(self, player_id, action):
        """
        Process the player's move.
        """
        col = action.strip()
        # check if number
        if not col.isdigit():
            return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} didn't give a column number: {action}"}
        col = int(col)
        # check if the action is valid
        if not (col>=0 and col<self.cols and self.board[0][col]=="."):
            return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried making an illegal move: {action}"}

        # place the disc 
        row = self._get_available_row(col)
        self.board[row][col] = 'X' if player_id == 0 else 'O'

        # render if required
        if self.render:
            print(f"Player {player_id} placed a disc in column {col}:")
            print(self._render_board())

        # Check for win
        if self._check_win(row, col):
            if self.render:
                print(f"Player {player_id} wins!")
            return None, {player_id:1, 1-player_id:-1}, True, {"reason": f"Player {player_id} wins!"}

        # Check for draw
        if all(self.board[0][c] != "." for c in range(self.cols)):
            if self.render:
                print("Game is a draw.")
            return None, {0:0, 1:0}, True, {"reason": "Game is a draw."}


        # Prepare state
        state = self._render_board() if self.is_open else ""
        return state, None, False, {}

    def _get_available_row(self, col):
        """Get the next available row in the specified column."""
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == '.':
                return r
        return -1  # Should not happen if move is valid

    def _check_win(self, row, col):
        """Check if placing a disc at (row, col) results in a win."""
        disc = self.board[row][col]
        directions = [
            [(0, 1), (0, -1)],    # Horizontal
            [(1, 0), (-1, 0)],    # Vertical
            [(-1, -1), (1, 1)],   # Diagonal /
            [(-1, 1), (1, -1)]    # Diagonal \
        ]
        for direction in directions:
            count = 1
            for dr, dc in direction:
                r, c = row, col
                while True:
                    r += dr
                    c += dc
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == disc:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False

    def _render_board(self):
        """Return a string representation of the board with column numbers."""
        column_numbers = ' '.join([str(c) for c in range(self.cols)])
        board_rows = '\n'.join([' '.join(row) for row in self.board])
        board_str = f"{column_numbers}\n{board_rows}"
        return board_str

