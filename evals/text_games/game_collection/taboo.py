import os
import json
import random
from evals.text_games.game_collection.game_interface import GameInterface 


class TabooGame(GameInterface):
    def __init__(
            self, 
            rounds_per_player=1, 
            turn_limit=10, 
            data_path="evals/text_games/game_collection/data/taboo/"
        ):
        self.rounds_per_player = rounds_per_player
        self.turn_limit = turn_limit
        self.data_path = data_path

        # load the data
        self._load_data()

    def reset(self):
        """Reset the game to its initial state."""
        self.trun = 0 # Current player turn {0, 1}
        self.game_over = False 
        self.role = {0: "Clue Giver", 1: "Guesser"}
        self.turn_count = 0
        self.word_to_guess, self.taboo_words = random.choice(list(self.data.items()))

        # Return the main prompts for both players
        player_prompts = {
            0: (
                f"You are the clue giver. The word to guess is '{self.word_to_guess}'. "
                f"Taboo words: {', '.join(self.taboo_words)}. Provide a clue for the word."
            ),
            1: (
                f"You are the guesser. Try to guess the correct word based on the clues."
            )
        }      
        return player_prompts

    def _load_data(self):
        """Load a random word and its taboo words from the data files."""
        taboo_files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
        if not taboo_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_path}")

        selected_file = random.choice(taboo_files)
        file_path = os.path.join(self.data_path, selected_file)

        with open(file_path, 'r') as f:
            self.data = json.load(f)

        if not self.data:
            raise ValueError(f"The file {selected_file} is empty or malformed.")


    def get_valid_actions(self, player_id):
        """ Return valid actions for the given player. """
        return None  # No restriction; actions will be validated in `step`

    def get_info(self):
        return self.turn_count

    def step(self, player_id, action):
        # Increment the turn counter
        self.turn_count += 1

        if self.role[player_id] == "Clue Giver":
            # Clue giver provides a clue; check for taboo words & legal word
            if any(taboo_word.lower() in action.lower() for taboo_word in self.taboo_words):
                game_reward = {self.turn:-1, 1-self.turn:0}
                return None, game_reward, True, {
                    "reason": "Clue giver used taboo word.",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
            elif self.word_to_guess.lower() in action.lower():
                game_reward = {self.turn:-1, 1-self.turn:0}
                return None, game_reward, True, {
                    "reason": "Clue giver used guess word.",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
            else:
                self.turn = 1 - player_id
                return f"\n{self.role[player_id]}: {action}", None, False, {
                    "info": "Clue Giver gave a clue",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
        else:
            # Guesser attempts to guess the word
            if self.word_to_guess.lower() in action.lower():
                game_reward = {0:1, 1:1} # both players win
                return None, game_reward, True, {
                    "reason": "The word was guessed correctly.",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
            # check for turn limit
            elif self.turn_count >= self.turn_limit:
                return None, {0:1, 1:1}, True, {
                    "reason": "The turn limit was reached.",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
            else:
                self.turn = 1 - player_id
                return f"\n{self.role[player_id]}: {action}", None, False, {
                    "info": "Guesser made a guess.",
                    "player_id": player_id,
                    "role": self.role[player_id]
                }
