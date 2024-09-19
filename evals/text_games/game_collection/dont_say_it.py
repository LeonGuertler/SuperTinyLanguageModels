"""
Each player is given a word. They need to try to make the other player say that word, before they 
say the word the other player was given.
"""

import os, json, random
from evals.text_games.game_collection.game_interface import GameInterface

class DontSayItGame(GameInterface):
    def __init__(
        self, 
        max_turns=10, 
        render=False,
        data_path="evals/text_games/game_collection/data/dont_say_it"):
        """
        Initialize the Don't Say It Game.
        Args:
            max_turns (int): Maximum number of turns for the game.
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Don't Say It"
        self.max_turns = max_turns
        self.render = render
        self.data_path = data_path
        self._load_words()
        #self.word_list = ["Apple", "Banana", "Cat", "Dog", "Elephant", "Flower", "Guitar", "House", "Ice", "Jungle"]
        #self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0
        self.target_words = {
            0: random.choice(self.word_list),
            1: random.choice(self.word_list) 
        }
        while self.target_words[0] == self.target_words[1]:
            self.target_words[1] = random.choice(self.word_list)  # Ensure different words

        player_prompts = {
            0: self._generate_player_prompt(player_id=0),
            1: self._generate_player_prompt(player_id=1)
        }
        return player_prompts


    def _load_words(self):
        """ Load all available words from a JSON file """
        with open(os.path.join(self.data_path, "words.json"), "r") as file:
            data = json.load(file)
            self.word_list = data["words"]

    def _generate_player_prompt(self, player_id):
        """
        Generate the initial prompt for a player.
        """
        prompt = (
            f"You are playing 'Don't Say It'.\n"
            f"Your secret word is: '{self.target_words[player_id]}'.\n"
            "Your goal is to get the other player to say your secret word before you say theirs.\n"
            "You can converse freely, but try to be subtle to avoid making it obvious.\n"
            f"The game lasts for {self.max_turns} turns in total.\n"
            "On your turn, simply type your message.\n"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """No restrictions on actions."""
        return None

    def get_info(self):
        """Return additional information."""
        return {'num_turns': self.turn}

    def step(self, player_id, action):
        """
        Process the player's action.
        """
        self.turn += 1

        if self.render:
            print(f"[Player {player_id}, Secret Word: {self.target_words[player_id]}] {action}\n")

        # check if the secret word was mentioned
        if self.target_words[1-player_id].lower() in action.lower():
            # target word mentioned
            if self.render:
                print(f"Player {player_id} mentioned the hidden word.")
            return None, {player_id:-1, 1-player_id:1}, True, {"reason": f"Player {player_id} mentioned the hidden word."}

        elif self.turn > self.max_turns:
            # turn limit has been reached
            if self.render:
                print(f"The turn limit has been reached.")
            return None, {player_id:-1, 1-player_id:-1}, True, {"reason": "The turn limit has been reached."}

        else:
            # normal action
            return action, None, False, {"info": f"Player {player_id}: {action}"}
            

