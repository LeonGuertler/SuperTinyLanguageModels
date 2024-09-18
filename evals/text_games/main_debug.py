import os 
from evals.text_games.llm_wrapper import LLMWrapper
from evals.text_games.human_agent import HumanAgent
from evals.text_games.game_collection import (
    TabooGame,
    ChessGame,
    NegotiationGame,
    TruthAndDeceptionGame,
    PokerGame,
    ConnectFourGame,
    DebateGame 
)
from evals.text_games.two_player_game_wrapper import TwoPlayerGameWrapper

from evals.text_games.llm_wrapper import GPT4Agent


def main():
    # Initialize agents with unique agent IDs
    # Retrieve the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    gpt_4_agent = GPT4Agent(agent_id=0, api_key=api_key, model_name="gpt-3.5-turbo")
    gpt_4_agent2 = GPT4Agent(agent_id=1, api_key=api_key, model_name="gpt-4o-mini")
    # human_agent = HumanAgent(agent_id=1)

    

    # Assign agents to players
    agents = {
        0: gpt_4_agent,
        1: gpt_4_agent2
    }

    """# Define game parameters
    game_kwargs = {
        'turn_limit': 6,
        'render': True,
    }

    # Initialize the TwoPlayerGameWrapper for Taboo
    game_wrapper = TwoPlayerGameWrapper(
        game_class=TabooGame,
        agents=agents,
        num_games=2,
        **game_kwargs
    )

    # Play the Taboo game
    agent_logs, agent_scores = game_wrapper.play_game()
    print("Taboo Game Results:")
    #print(agent_logs)
    print(agent_scores)
    #exit()


    # Initialize the TwoPlayerGameWrapper for Chess (Open)
    chess_open_game_kwargs = {
        'is_open': True,
        #'render': True,
        'turn_limit': 30,
        'show_valid': True
    }

    chess_open_game_wrapper = TwoPlayerGameWrapper(
        game_class=ChessGame,
        agents=agents,
        num_games=1,
        **chess_open_game_kwargs
    )

    # Play the Chess Open game
    agent_logs, agent_scores = chess_open_game_wrapper.play_game()
    print("Chess Open Game Results:")
    print(agent_logs)
    print(agent_scores)

    # Initialize the TwoPlayerGameWrapper for Chess (Closed)
    chess_closed_game_kwargs = {
        'is_open': False,
        'turn_limit': 30,
        'show_valid': True
    }

    chess_closed_game_wrapper = TwoPlayerGameWrapper(
        game_class=ChessGame,
        agents=agents,
        num_games=1,
        **chess_closed_game_kwargs
    )

    # Play the Chess Closed game
    agent_logs, agent_scores = chess_closed_game_wrapper.play_game()
    print("Chess Closed Game Results:")
    print(agent_logs)
    print(agent_scores)
    input()

    # Initialize the TwoPlayerGameWrapper for Negotiation Game
    negotiation_game_kwargs = {
        'max_turns': 10,
        'render': True
    }

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

    # initialize the game wrapper for truth and deception
    tad_game_kwargs = {
        'max_turns': 3, # uneven number is required
        'render': True
    }

    tad_game_wrapper = TwoPlayerGameWrapper(
        game_class = TruthAndDeceptionGame,
        agents=agents,
        num_games=1,
        **tad_game_kwargs
    )

    # Play the game
    agent_logs, agent_scores = tad_game_wrapper.play_game()
    print("Truth And Deception Game Results:")
    print(agent_logs)
    print("\n\n")
    print(agent_scores)
    #"""


    """# initialize the game wrapper for poker
    poker_game_kwargs = {
        'render': True
    }

    poker_game_wrapper = TwoPlayerGameWrapper(
        game_class = PokerGame,
        agents=agents,
        num_games=10,
        **poker_game_kwargs
    )

    # Play the game
    agent_logs, agent_scores = poker_game_wrapper.play_game()
    print("Truth And Deception Game Results:")
    print(agent_logs)
    print("\n\n")
    print(agent_scores)"""

    # initialize the game wrapper for poker
    connect_four_game_kwargs = {
        'render': True,
        'is_open': True
    }

    connect_four_game_wrapper = TwoPlayerGameWrapper(
        game_class = ConnectFourGame,
        agents=agents,
        num_games=10,
        **connect_four_game_kwargs
    )

    # Play the game
    agent_logs, agent_scores = connect_four_game_wrapper.play_game()
    print("Truth And Deception Game Results:")
    print(agent_logs)
    print("\n\n")
    print(agent_scores)
    

if __name__ == "__main__":
    main()

    input(agent_scores)
    exit()
    # Access agent logs
    agent_logs = detailed_log['agent_logs']

    # Print high-level summaries
    print("\n=== High-Level Game Summary ===")
    for game_summary in high_level_log:
        print(f"Round: {game_summary['round']}, Game: {game_summary['game']}")
        print(f"Number of Turns: {game_summary['turns']}")
        print(f"Game Reward: {game_summary['game_reward']}")
        print(f"Roles: {game_summary['roles']}")
        print(f"Reason for Game End: {game_summary['reason_for_game_end']}\n")

    # Print agent-specific logs
    print("\n=== Agent Logs ===")
    for agent_id, logs in agent_logs.items():
        print(f"\nAgent {agent_id} Logs:")
        for log in logs:
            input(log)
            #print(f"Round {log['round']}, Game {log['game']}, Role: {log.get('role', 'N/A')}, Action: {log['action']}, Reward: {log['reward']}")
            # Include additional log details as needed