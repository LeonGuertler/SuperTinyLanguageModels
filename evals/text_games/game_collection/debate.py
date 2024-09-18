from evals.text_games.game_collection.game_interface import GameInterface
import openai

class DebateGame(GameInterface):
    def __init__(self, 
        topic="Should artificial intelligence be regulated?", 
        max_turns=4, 
        api_key="YOUR_API_KEY"
    ):
        """
        Initialize the Debate game.
        Args:
            topic (str): The debate topic.
            max_turns (int): Number of turns for the debate.
            api_key (str): OpenAI API key for the judge.
        """
        self.name = "Debate"
        self.topic = topic
        self.max_turns = max_turns
        self.api_key = api_key
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0
        self.game_over = False
        self.arguments = {0: [], 1: []}

        # Assign sides
        self.sides = {
            0: "Affirmative",  # Supports the topic
            1: "Negative"      # Opposes the topic
        }

        # Prepare initial prompts
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }
        return player_prompts

    def _generate_player_prompt(self, player_id):
        prompt = (
            f"You are participating in a debate.\n"
            f"Topic: {self.topic}\n"
            f"Your position: {self.sides[player_id]}\n"
            f"You will have {self.max_turns} turns to present your arguments.\n"
            f"Please make your first argument."
        )
        return prompt

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        return None  # Any argument

    def get_info(self):
        """Return additional information."""
        return {'turn': self.turn}

    def step(self, player_id, action):
        """
        Process the player's argument.
        """
        done = False
        reward = None
        info = {}

        # Store the argument
        self.arguments[player_id].append(action)

        # Prepare the state for the other player (last argument made)
        state = action

        self.turn += 1

        if self.turn >= self.max_turns * 2:
            # Debate is over
            done = True
            winner_id = self._judge_debate()
            reward = {winner_id: 1, 1 - winner_id: -1}
            info = {"info": f"Player {winner_id} wins the debate."}
            state = None

        return state, reward, done, info

    def _judge_debate(self):
        """
        Use an AI judge to determine the winner.
        """
        openai.api_key = self.api_key

        debate_text = (
            f"Topic: {self.topic}\n\n"
            f"Player 0 ({self.sides[0]}):\n"
            f"{' '.join(self.arguments[0])}\n\n"
            f"Player 1 ({self.sides[1]}):\n"
            f"{' '.join(self.arguments[1])}\n\n"
            f"Please determine which player presented a stronger argument and explain your reasoning."
        )

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=debate_text,
            max_tokens=150,
            temperature=0.7
        )

        judge_decision = response.choices[0].text.strip().lower()
        if "player 0" in judge_decision:
            return 0
        elif "player 1" in judge_decision:
            return 1
        else:
            # In case of ambiguity, choose randomly
            return random.choice([0, 1])

