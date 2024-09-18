import os

from evals.text_games.game_collection.chess import ChessGame
from evals.text_games.game_collection.iterated_prisoners_dilemma import (
    IteratedPrisonersDilemma,
)
from evals.text_games.game_collection.negotiation import NegotiationGame
from evals.text_games.game_collection.taboo import TabooGame
from evals.text_games.human_agent import HumanAgent
from evals.text_games.llm_wrapper import GPT4Agent, LLMWrapper
from evals.text_games.two_player_game_wrapper import TwoPlayerGameWrapper


def main():
    # Initialize agents with unique agent IDs
    # Retrieve the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )

    gpt_4_agent = GPT4Agent(agent_id=0, api_key=api_key, model_name="gpt-3.5-turbo")
    gpt_4_agent2 = GPT4Agent(agent_id=1, api_key=api_key, model_name="gpt-4o-mini")
    # human_agent = HumanAgent(agent_id=1)

    # Assign agents to players
    agents = {0: gpt_4_agent, 1: gpt_4_agent2}

    """# Define game parameters
    game_kwargs = {
        "turn_limit": 6,
        "data_path": "evals/text_games/game_collection/data/taboo/",
    }

    # Initialize the TwoPlayerGameWrapper for Taboo
    game_wrapper = TwoPlayerGameWrapper(
        game_class=TabooGame, agents=agents, num_games=2, **game_kwargs
    )

    # Play the Taboo game
    agent_logs, agent_scores = game_wrapper.play_game()
    print("Taboo Game Results:")
    # print(agent_logs)
    print(agent_scores)
    # exit()

    # Initialize the TwoPlayerGameWrapper for Chess (Open)
    chess_open_game_kwargs = {
        "is_open": True,
        #'render': True,
        "turn_limit": 30,
        "show_valid": True,
    }

    chess_open_game_wrapper = TwoPlayerGameWrapper(
        game_class=ChessGame, agents=agents, num_games=1, **chess_open_game_kwargs
    )

    # Play the Chess Open game
    agent_logs, agent_scores = chess_open_game_wrapper.play_game()
    print("Chess Open Game Results:")
    print(agent_logs)
    print(agent_scores)

    # Initialize the TwoPlayerGameWrapper for Chess (Closed)
    chess_closed_game_kwargs = {"is_open": False, "turn_limit": 30, "show_valid": True}

    chess_closed_game_wrapper = TwoPlayerGameWrapper(
        game_class=ChessGame, agents=agents, num_games=1, **chess_closed_game_kwargs
    )

    # Play the Chess Closed game
    agent_logs, agent_scores = chess_closed_game_wrapper.play_game()
    print("Chess Closed Game Results:")
    print(agent_logs)
    print(agent_scores)
    input()"""

    # Initialize the TwoPlayerGameWrapper for Negotiation Game
    negotiation_game_kwargs = {"resource_types": 5, "max_turns": 10}

    negotiation_game_wrapper = TwoPlayerGameWrapper(
        game_class=NegotiationGame,
        agents=agents,
        num_games=10,
        **negotiation_game_kwargs
    )

    # Play the Negotiation game
    agent_logs, agent_scores = negotiation_game_wrapper.play_game()
    print("Negotiation Game Results:")
    print(agent_logs)
    print("\n\n")
    print(agent_scores)

    # Initialize the TwoPlayerGameWrapper for Iterated Prisoners Dilemma
    negotiation_game_wrapper = TwoPlayerGameWrapper(
        game_class=IteratedPrisonersDilemma,
        agents=agents,
        num_games=10,
        **negotiation_game_kwargs
    )

    # Play the Iterated Prisoners Dilemma game
    agent_logs, agent_scores = negotiation_game_wrapper.play_game()
    print("Iterated Prisoners Dilemma Game Results:")
    print(agent_logs)
    print(agent_scores)


if __name__ == "__main__":
    main()
