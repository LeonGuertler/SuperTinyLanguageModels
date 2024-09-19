# TODO


"""
The first model is given an emotion from a predefined list and needs to create a story conveying that emotion.
The other model needs to guess the emotion. The story is fixed in length. Scoring is similar to taboo
(the main difference from taboo is that there is not back and forth, only a fixed length story). Each 
emotion should also have a list of words that can't be mentioned in the story.
"""

import random
from evals.text_games.game_collection.game_interface import GameInterface

class EmotionExpressionGame(GameInterface):
    def __init__(self, render=False):
        """
        Initialize the Emotion Expression Game.
        Args:
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Emotion Expression"
        self.render = render
        self.emotions = {
            "Joy": ["happy", "joyful", "elated"],
            "Sadness": ["sad", "depressed", "unhappy"],
            "Anger": ["angry", "furious", "irritated"],
            "Fear": ["afraid", "scared", "fearful"],
            "Surprise": ["surprised", "astonished", "amazed"],
            "Disgust": ["disgusted", "revolted", "nauseated"]
        }
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.game_over = False
        self.story = None
        self.target_emotion = random.choice(list(self.emotions.keys()))
        self.forbidden_words = self.emotions[self.target_emotion]
        self.player_prompts = {
            0: self._generate_story_prompt(),
            1: self._generate_guess_prompt()
        }
        return self.player_prompts

    def _generate_story_prompt(self):
        """
        Generate the prompt for the story-telling player.
        """
        prompt = (
            "You are given an emotion to convey through a short story.\n"
            f"The emotion you need to convey is: '{self.target_emotion}'.\n"
            "Write a story that clearly expresses this emotion without using the following words:\n"
            f"{', '.join(self.forbidden_words)}\n"
            "Your story should be approximately 200 words.\n"
        )
        return prompt

    def _generate_guess_prompt(self):
        """
        Generate the prompt for the guessing player.
        """
        prompt = (
            "You will be given a short story written by another player.\n"
            "Your task is to identify the primary emotion conveyed in the story.\n"
            "Possible emotions are: Joy, Sadness, Anger, Fear, Surprise, Disgust.\n"
            "Respond with the emotion you believe is being expressed.\n"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """No restrictions on actions."""
        return None

    def get_info(self):
        """No additional info."""
        return {}

    def step(self, player_id, action):
        """
        Process the player's action.
        """
        done = False
        reward = {0: 0, 1: 0}
        info = {}

        if player_id == 0:
            # Story-telling player
            self.story = action
            # Check for forbidden words
            action_lower = action.lower()
            for word in self.forbidden_words:
                if word.lower() in action_lower:
                    reward[0] = -1
                    reward[1] = 1
                    done = True
                    info['reason'] = f"Forbidden word '{word}' used in the story. Player 1 wins."
                    return action, reward, done, info
            # Proceed to the next player
            info['info'] = "Story received. Waiting for Player 1 to guess the emotion."
        else:
            # Guessing player
            guessed_emotion = action.strip().capitalize()
            if guessed_emotion == self.target_emotion:
                reward[0] = 1
                reward[1] = 1
                info['reason'] = "Correct emotion guessed. Both players win."
            else:
                reward[0] = -1
                reward[1] = -1
                info['reason'] = f"Incorrect emotion guessed. The correct emotion was '{self.target_emotion}'. Both players lose."
            done = True

        if self.render:
            print(f"[Player {player_id}] '{action}'")

        return action, reward, done, info
