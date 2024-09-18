"""Contains the main classes used for our self-play games/ eval suite

We generally """

import abc
import enum


class Rendering(enum.Enum, str):
    """Enum for rendering options"""

    TEXT = "text"


class Agent:
    """Interface for game agents"""

    id = ...
    preferred_rendering = Rendering

    def __init__(self, agent_id: int):
        self.id = agent_id

    @abc.abstractmethod
    def generate_action(self, state_rendering) -> str:
        """Generate an action based on the current state."""

    @abc.abstractmethod
    def reset(self):
        """Reset the agent to its initial state."""


class Game:
    """Interface for games"""

    def __init__(self, agents):
        self.agents = agents
        self.state = None
        self.done = False

    @abc.abstractmethod
    def reset(self, agents):
        """Reset the state of the game, including by providing a new set of agents.
        Args:
            agents (list[Agent]): The agents playing the game."""

    @abc.abstractmethod
    def next_round(self) -> list[float]:
        """Proceed to the next round of the game
        Return the reward for the round to each player."""

    @abc.abstractmethod
    def get_winner(self) -> int:
        """Get the winner of the game."""

    @abc.abstractmethod
    def str_render(self) -> str:
        """Return a string representation of the game state."""

    def play(self) -> int:
        """Play a game and return a winner id."""
        while not self.done:
            reward = self.next_round()
        return self.get_winner()


class Competition:
    """Play a series of games between agents to determine the best agent."""

    def __init__(self, game: Game, num_rounds: int, agents: list[Agent]) -> None:
        self.game = game
        self.total_num_rounds = num_rounds
        self.current_round = 0
        self.agents = agents
        self.scores = {agent.id: 0 for agent in agents}

    @abc.abstractmethod
    def get_player_id_order(self) -> list[int]:
        """Get the order of players for the current round.
        Default implementation is to alternate between players."""
        current_round = self.current_round
        num_players = len(self.agents)
        return [current_round % num_players, (current_round + 1) % num_players]

    def play_competition(self) -> dict[int, int]:
        """Play a series of games between agents and return the scores."""
        while self.current_round < self.total_num_rounds:
            player_ids = self.get_player_id_order()
            for _id in player_ids:
                self.agents[_id].reset()
            self.game.reset([self.agents[_id] for _id in player_ids])
            winner_id = self.game.play()
            self.scores[winner_id] += 1
            self.current_round += 1
        return self.scores
