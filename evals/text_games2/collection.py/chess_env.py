import chess

from evals.text_games2.interfaces import Agent, Game

CHESS_PROMPT = """You are playing {Colour} in a game of chess. Make your move in UCI format (e.g. e2e4).

The current board state is:
{Board}

Valid moves are: {ValidMoves}"""


class ChessGame(Game):
    def __init__(self, agents: list[Agent], turn_limit=30):
        super().__init__(agents)
        self.board = chess.Board()
        self.turn = 0
        self.turn_limit = turn_limit

    def reset(self, agents: list[Agent]):
        self.agents = agents
        self.board = chess.Board()
        self.done = False
        self.turn = 0

    def next_round(self):
        if self.turn % 2 == 0:
            player_id = 0
            assert self.board.turn == chess.WHITE
        else:
            player_id = 1
            assert self.board.turn == chess.BLACK
        move = self.agents[player_id].generate_action(self.text_render(player_id))
        if move not in self.board.legal_moves:
            self.done = True
            loser = player_id
            return [-1 if i == loser else 1 for i in range(2)]

        self.board.push(chess.Move.from_uci(move))
        if self.board.is_checkmate():
            self.done = True
            winner = player_id
            return [1 if i == winner else -1 for i in range(2)]
        elif self.board.fullmove_number > self.turn_limit:
            self.done = True
        self.turn += 1
        return [0, 0]

    def text_render(self, player_id: int) -> str:
        """Render the game state as text."""
        return CHESS_PROMPT.format(
            Colour="White" if player_id == 0 else "Black",
            Board=self.board.unicode(),
            ValidMoves=self.board.legal_moves,
        )
