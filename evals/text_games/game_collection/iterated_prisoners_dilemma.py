import random

from evals.text_games.game_collection import game_interface

MAX_ROUNDS = 100
NUM_CONVERSATION_ROUNDS_PER_GAME = 2

PROMPT = """You are playing iterated prisoners dilemma with conversation
We have {NUM_CONVERSATION_ROUNDS_PER_GAME} rounds of conversation inbetween each game
On a Conversation round, you can say anything you want. On a Game round, you can choose to Cooperate or Defect.
If both players cooperate, you both get 2 points. If one player defects, they get 3 points and the other gets 0.
If both players defect, you both get 1 point.
The player with the most points at the end of the game wins.
You can send your conversation messages {FIRST_SECOND}"""


class IteratedPrisonersDilemma(game_interface.GameInterface):
    """Iterated Prisoner's Dilemma game.

    For the sake of the interface our state consists of:
    - The list of previous actions taken by the agents
    - The conversation history between the agents

    The valid actions are "Cooperate" and "Defect" on turns where the agents are playing the game.
    "Reward" is given at the end of the game based on the winner rather than the accumulated score.
    """

    def __init__(self, agent1, agent2, max_steps=10):
        """
        Initialize the game.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            max_steps (int, optional): The maximum number of steps in the game.
        """
        super(IteratedPrisonersDilemma, self).__init__(agent1, agent2)
        self.max_steps = max_steps
        self.steps = 0
        self.total_rounds = None
        self.scores = {
            agent1.get_id(): 0,
            agent2.get_id(): 0,
        }
        self.games_record = []
        self.current_game_actions = {}
        self.conversation_record = []
        self.converation_turns_left = NUM_CONVERSATION_ROUNDS_PER_GAME
        self.agent1 = agent1
        self.agent2 = agent2

    def reset(self):
        """Reset the game to its initial state."""
        self.steps = 0
        self.total_rounds = random.randint(1, MAX_ROUNDS)
        self.agent1.reset()
        self.agent2.reset()
        self.scores = {
            self.agent1.get_id(): 0,
            self.agent2.get_id(): 0,
        }
        self.game_record = []
        self.conversation_record = []
        self.converation_turns_left = NUM_CONVERSATION_ROUNDS_PER_GAME
        self.current_game_actions = {}
        # now return a prompt showing how to play the game?? Specify this in the interface...
        return {
            self.agent1.get_id(): PROMPT.format(
                NUM_CONVERSATION_ROUNDS_PER_GAME=NUM_CONVERSATION_ROUNDS_PER_GAME,
                FIRST_SECOND="first" if self.agent1.get_id() == 0 else "second",
            ),
            self.agent2.get_id(): PROMPT.format(
                NUM_CONVERSATION_ROUNDS_PER_GAME=NUM_CONVERSATION_ROUNDS_PER_GAME,
                FIRST_SECOND="first" if self.agent2.get_id() == 0 else "second",
            ),
        }

    def is_over(self):
        """Check if the game has ended."""
        return self.steps >= self.total_rounds

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        if self.converation_turns_left == 0:
            return ["Cooperate", "Defect"]
        return None  # no restriction on form of conversation

    def get_winner(self):
        """Determine the winner of the game."""
        if sum(self.scores[self.agent1.get_id()]) > sum(
            self.scores[self.agent2.get_id()]
        ):
            return self.agent1.get_id()
        elif sum(self.scores[self.agent1.get_id()]) < sum(
            self.scores[self.agent2.get_id()]
        ):
            return self.agent2.get_id()
        else:
            return None

    def _assign_scores(self, action_1, action_2):
        if action_1 == "Cooperate" and action_2 == "Cooperate":
            return (2, 2)
        elif action_1 == "Cooperate" and action_2 == "Defect":
            return (0, 3)
        elif action_1 == "Defect" and action_2 == "Cooperate":
            return (3, 0)
        else:
            return (1, 1)

    def step(self, player_id, action):
        """
        Apply the player's action to the game.

        Returns:
            state (object): The new state after the action.
            reward (float): The reward received after the action.
            done (bool): Whether the game has ended.
            info (dict): Additional information.
        """
        state = {
            "game_record": self.game_record,
            "conversation_record": self.conversation_record,
            "turn_type": "Conversation" if self.converation_turns_left > 0 else "Game",
        }
        if self.converation_turns_left > 0:
            self.conversation_record.append((player_id, action))
            self.converation_turns_left -= 1
            return state, 0, False, {"reason": "Conversation"}
        else:
            self.steps += 1
            self.current_game_actions[player_id] = action
            if len(self.current_game_actions) == 2:
                # Determine the reward based on the actions
                score_1, score_2 = self._assign_scores(
                    self.current_game_actions[self.agent1.get_id()],
                    self.current_game_actions[self.agent2.get_id()],
                )
                self.scores[self.agent1.get_id()] += score_1
                self.scores[self.agent2.get_id()] += score_2
                self.game_record.append(
                    (
                        self.current_game_actions[self.agent1.get_id()],
                        self.current_game_actions[self.agent2.get_id()],
                    )
                )
                self.current_game_actions = {}
            if self.steps >= self.total_rounds:
                winner = self.get_winner()
                if player_id == winner:
                    reward = 1
                elif winner is None:
                    reward = 0
                else:
                    reward = -1
                return state, reward, True, {"reason": "Game Over"}
            return state, 0, False, {"reason": "Game"}


class TitForTatAgent:
    """Tit for Tat Agent"""

    def __init__(self, agent_id, prompt_prefix=""):
        """
        Initialize the HumanAgent.

        Args:
            player_id (int): The ID of the player.
            prompt_prefix (str): The prefix to display when prompting for input.
        """
        self.player_id = agent_id
        self.prompt_prefix = prompt_prefix
        self.history = []  # Stores tuples of (prompt, response)
        self.main_prompt = ""
        self.agent_id = agent_id

    def reset(self, main_prompt):
        """
        Reset the agent with a new main prompt.

        Args:
            main_prompt (str): The main prompt or instructions for the player.
        """
        self.main_prompt = main_prompt
        self.history = []  # Clear history for a new game

    def get_action(self, immediate_state, valid_actions=None):
        """
        Prompt the human user for an action.

        Args:
            immediate_state (str): The current state of the game specific to the player.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The action input by the user.
        """
        if immediate_state["turn_type"] == "Conversation":
            return "I got nothing to say to you buddy"
        elif len(immediate_state["game_record"]) == 0:
            return "Cooperate"
        prev_turn = immediate_state["game_record"][-1]
        opp_action = [prev_turn[id] for id in prev_turn if id is not self.agent_id][0]
        return opp_action

    def get_history(self):
        """
        Retrieve the conversation history.

        Returns:
            list: A list of tuples containing (prompt, response).
        """
        return self.history
