
from evals.text_games.llm_wrapper import LLMWrapper
from evals.text_games.human_agent import HumanAgent
from evals.text_games.game_collection import TabooGame, WerewolfGame, NegotiationGame, ChessGame, DiplomacyGame

from evals.text_games.two_player_game_wrapper import TwoPlayerGameWrapper
from evals.text_games.game_collection.taboo import TabooGame

from evals.text_games.llm_wrapper import GPT4Agent


def main():
    # Initialize agents with unique agent IDs
    api_key = ""
    gpt_4_agent = GPT4Agent(agent_id=0, api_key=api_key, model_name="gpt-4o")
    gpt_4_agent2 = GPT4Agent(agent_id=1, api_key=api_key, model_name="gpt-4o-mini")
    #human_agent = HumanAgent(agent_id=0)

    # Assign agents to players
    agents = {
        0: gpt_4_agent,
        1: gpt_4_agent2
    }

# Define game parameters
    game_kwargs = {
        'turn_limit': 6,
        'data_path': "evals/text_games/game_collection/data/taboo/"
    }

    # Initialize the TwoPlayerGameWrapper
    game_wrapper = TwoPlayerGameWrapper(
        game_class=TabooGame,
        agents=agents,
        num_games=10,
        **game_kwargs
    )

    # Play the game
    agent_logs, agent_scores = game_wrapper.play_game()
    print(agent_logs)
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