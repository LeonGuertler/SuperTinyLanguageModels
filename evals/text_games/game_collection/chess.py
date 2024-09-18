from evals.text_games.game_collection.game_interface import GameInterface 
import chess

class ChessGame(GameInterface):
    def __init__(self, provide_board_state=False):
        self.provide_board_state = provide_board_state
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.game_over = False
        self.turn = 0  # 0 for white, 1 for black

    def get_state(self, player_id):
        if self.provide_board_state:
            return str(self.board)
        else:
            return 'Moves so far: ' + ' '.join(self.board.move_stack)

    def get_valid_actions(self, player_id):
        return [move.uci() for move in self.board.legal_moves]

    def step(self, player_id, action):
        if self.game_over:
            return None, 0, True, {'message': 'Game is over.'}

        if player_id != self.turn:
            return None, 0, False, {'message': 'Not your turn.'}

        if action in self.get_valid_actions(player_id):
            move = chess.Move.from_uci(action)
            self.board.push(move)
            reward = 0
            if self.board.is_checkmate():
                self.game_over = True
                reward = 1  # Winning player gets 1
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                self.game_over = True
                reward = 0  # Draw
            self.turn = 1 - self.turn
            state = self.get_state(self.turn)
            done = self.game_over
            return state, reward, done, {}
        else:
            # Illegal move
            self.game_over = True
            reward = -1
            return None, reward, True, {'message': 'Illegal move.'}

    def is_over(self):
        return self.game_over

    def get_winner(self):
        if self.board.is_checkmate():
            return 1 - self.turn  # The player who made the last move
        else:
            return None  # Draw or game not over
