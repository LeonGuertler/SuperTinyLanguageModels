import os
import json
import random
from evals.text_games.game_collection.game_interface import GameInterface
import openai

class DebateGame(GameInterface):
    def __init__(self, 
                 render=False,
                 max_turns=4, 
                 api_key=None,
                 data_path="evals/text_games/game_collection/data/debate/topics.json",
             ):
        """
        Initialize the Debate game.

        Args:
            render (bool): Whether to print game information.
            max_turns (int): Number of turns per player.
            data_path (str): Path to the JSON file containing debate topics.
            api_key (str, optional): OpenAI API key. If not provided, it will be loaded from environment variables.
        """
        self.name = "Debate"
        self.render = render 
        self.max_turns = max_turns
        self.judge_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
        self.num_judges = 11

        # Load API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it as a parameter.")

        openai.api_key = api_key

        # Load debate topics
        self.data_path = data_path
        self._load_topics()

    def _load_topics(self):
        """Load debate topics from the JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Debate topics file not found at {self.data_path}")

        with open(self.data_path, 'r') as f:
            data = json.load(f)

        if "topics" not in data or not isinstance(data["topics"], list):
            raise ValueError("Invalid format for debate topics JSON. Expected a key 'topics' with a list of topics.")

        self.topics = data["topics"]
        if not self.topics:
            raise ValueError("Debate topics list is empty.")

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0
        self.game_over = False
        self.arguments = {0: [], 1: []}

        # Select a random topic
        self.topic = random.choice(self.topics)

        # Assign sides randomly
        self.sides = {
            0: "Affirmative",  # Supports the topic
            1: "Negative"      # Opposes the topic
        }
        # Optionally, randomize which player gets which side
        if random.choice([True, False]):
            self.sides = {
                0: "Negative",
                1: "Affirmative"
            }

        # Conduct pre-debate voting
        self.pre_votes = self._evaluate_debate(debate_transcript=None)

        if self.render:
            print(f"Debate Topic: {self.topic}")
            print(f"Player 0 is {self.sides[0]}")
            print(f"Player 1 is {self.sides[1]}")
            print(f"Number of Judges: {self.num_judges}")
            print(f"Pre-Debate Votes: {self.pre_votes}\n")

        # Prepare initial prompts
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }

        return player_prompts

    def _generate_player_prompt(self, player_id):
        """Generate the initial prompt for a player based on their side."""
        prompt = (
            f"You are participating in a debate.\n"
            f"Topic: {self.topic}\n"
            f"Your position: {self.sides[player_id]}\n"
            f"You will have {self.max_turns} turns to present your arguments.\n"
            f"Please make your first argument."
        )
        return prompt

    def _evaluate_debate(self, debate_transcript=None):
        """
        Conduct evaluation by judges.
        If debate_transcript is None, judges evaluate based on the topic only (pre-debate voting).
        Otherwise, they evaluate based on the topic and the debate transcript (post-debate voting).

        Returns:
            dict: A dictionary with sides as keys and their corresponding vote counts.
        """
        votes = { "Affirmative": 0, "Negative": 0 }

        for i in range(self.num_judges):
            model = self.judge_models[i % len(self.judge_models)]
            prompt = f"Debate Topic: {self.topic}\n"
            if debate_transcript:
                prompt += f"Debate Transcript:\n{debate_transcript}\n"
                prompt += "Please vote on which side you agree with based on the debate transcript and the topic."
            else:
                prompt += "No debate has occurred yet. Please vote on which side you agree with based solely on the topic."

            prompt += "\nVote for 'Affirmative' or 'Negative'. Provide only the side you vote for."

            try:
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
                judge_decision = response.choices[0].text.strip().lower()
                if "affirmative" in judge_decision:
                    votes["Affirmative"] += 1
                elif "negative" in judge_decision:
                    votes["Negative"] += 1
                else:
                    # In case of ambiguity, randomly assign
                    chosen = random.choice(["Affirmative", "Negative"])
                    votes[chosen] += 1
            except Exception as e:
                # In case of API failure, randomly assign
                chosen = random.choice(["Affirmative", "Negative"])
                votes[chosen] += 1

        if self.render and debate_transcript is None:
            print(f"Pre-Debate Votes: {votes}\n")
        elif self.render and debate_transcript:
            print(f"Post-Debate Votes: {votes}\n")

        return votes

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        return None  # Any argument is valid

    def get_info(self):
        """Return additional information."""
        return {
            'num_turns': self.turn,
            'pre_votes': self.pre_votes,
            'arguments': self.arguments
        }

    def step(self, player_id, action):
        """
        Process the player's argument.

        Args:
            player_id (int): The ID of the player (0 or 1).
            action (str): The argument provided by the player.

        Returns:
            tuple: (state, reward, done, info)
        """
        done = False
        reward = None
        info = {}

        # Store the argument
        self.arguments[player_id].append(action)

        if self.render:
            print(f"\n[Player {player_id} ({self.sides[player_id]})] {action}")

        # Prepare the state for the other player (last argument made)
        state = action

        self.turn += 1

        if self.turn >= self.max_turns: # * 2:
            # Debate is over
            done = True
            info, reward = self._evaluate_and_determine_winner()
            #info = {"info": f"Player {winner_id} wins the debate based on judge votes."}
            state = None

            if self.render:
                print(info)
                print(f"Final Rewards: {reward}\n")
                input()

        return state, reward, done, info

    def _evaluate_and_determine_winner(self):
        """
        Evaluate the debate and determine the winner based on judge votes.

        Returns:
            tuple: (winner_id, reward_dict)
        """
        # Compile the debate transcript
        debate_transcript = ""
        for pid in [0, 1]:
            for arg in self.arguments[pid]:
                debate_transcript += f"Player {pid} ({self.sides[pid]}): {arg}\n"

        # Conduct post-debate voting
        post_votes = self._evaluate_debate(debate_transcript=debate_transcript)

        # Calculate gains
        gain_affirmative = post_votes["Affirmative"] - self.pre_votes["Affirmative"]
        gain_negative = post_votes["Negative"] - self.pre_votes["Negative"]

        if self.render:
            print(f"Gain for Affirmative: {gain_affirmative}")
            print(f"Gain for Negative: {gain_negative}")

        # Determine which side has a larger gain
        if gain_affirmative > gain_negative:
            winner_id = list(self.sides.keys())[list(self.sides.values()).index("Affirmative")]
            info = {"reason": f"Player {winner_id} (Affirmative) wins the debate."}
            reward = {winner_id:1, 1-winner_id:0}

        elif gain_affirmative < gain_negative:
            winner_id = list(self.sides.keys())[list(self.sides.values()).index("Negative")]
            info = {"reason": f"Player {winner_id} (Negative) wins the debate."}
            reward = {winner_id:1, 1-winner_id:0}

        else:
            # it's a tie
            info = {"reason": "It's a Tie"}
            reward = {0:0, 1:0}


        return info, reward
