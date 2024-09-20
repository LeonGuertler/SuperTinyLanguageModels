import re, random
import enchant # for word checking
from evals.text_games.game_collection.game_interface import GameInterface

import nltk
from nltk.corpus import words
nltk.download('words')
nltk_words = set(word.lower() for word in words.words())


class WordChainsGame(GameInterface):
    def __init__(self, max_turns=None, render=False):
        """
        Initialize the Word Chains Game.
        Args:
            max_turns (int): Maximum number of turns before the game ends in a draw.
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Word Chains Game"
        self.max_turns = max_turns
        self.render = render
        self.word_checker_us = enchant.Dict("en_US")
        self.word_checker_uk = enchant.Dict("en_GB")


    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0
        self.game_over = False
        self.used_words = set()

        # Select a random starting word
        self.starting_word = random.choice(list(nltk_words))
        self.used_words.add(self.starting_word.lower())

        # Set the required starting letter for the first player's word
        self.required_start_letter = self.starting_word[-1].lower()

        if self.render:
            print(f"\n\nGame start. The first word is: {self.starting_word}")

        # Generate initial prompts for both players, informing them of the starting word
        player_prompts = {
            0: self._generate_player_prompt(),
            1: self._generate_player_prompt()
        }
        return player_prompts

    def _generate_player_prompt(self):
        """
        Generate the initial prompt for a player.
        """
        prompt = (
            f"You are playing the Word Chains Game.\n"
            f"Taking turns your are competing to find valid english words that start with the last letter "
            f"of the previous word. Repetition of words is not allowed.\n"
            "If you provide an invalid word, repeat a word, or fail to follow the sequence, you lose.\n"
            "Please wrap your word in squared brackets. (i.e. '[fox]', etc.\n"
            f"The starting word is [{self.starting_word}]. Plase give the next word.\n"
        )
        if self.max_turns:
            prompt += f"The game will end after {self.max_turns} turns if no player fails.\n"
        prompt += "On your turn, please respond with your word.\n"
        return prompt

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        # In this game, any word is a potential action.
        # Validation is handled in the `step` method.
        return None

    def get_info(self):
        """Return additional information."""
        return {'num_turns': self.turn}

    def step(self, player_id, action):
        """
        Process the player's action.
        Args:
            player_id (int): The player's ID.
            action (str): The player's action (the word).
        Returns:
            state (str): The action taken.
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        observation, done, reward, info = None, False, None, {}


        self.turn += 1


        word = action.strip().lower()
        match = re.search(r'\[(\w+)\]', word)
        if match:
            word = match.group(1)
        else:
            done = True 
            reward = {player_id:-1, 1-player_id:1}
            info["reason"] = f"Player {player_id} didn't provide a word in the right format."
        

        # check if the word starts with the required letter
        if not word.startswith(self.required_start_letter):
            done = True 
            reward = {player_id:-1, 1-player_id:1}
            info["reason"] = f"Player {player_id} gave a word starting with the wrong letter."
        
        # check if the word is a valid English word
        elif not (
            self.word_checker_us.check(word) and 
            self.word_checker_uk.check(word) and 
            self.word_checker_us.check(word.title()) and 
            self.word_checker_uk.check(word.title()) 
        ):
            done = True 
            reward = {player_id:-1, 1-player_id:1}
            info["reason"] = f"Player {player_id}'s word is not a valid english word."

        # check if the word has been used before
        elif word in self.used_words:
            done = True 
            reward = {player_id:-1, 1-player_id:1}
            info["reason"] = f"Player {player_id} tried using a word twice."

        # check if max turns have been reached
        elif (self.max_turns is not None and self.turn > self.max_turns):
            done = True 
            reward = {player_id:0, 1-player_id:0}
            info["reason"] = f"The turn limit has been reached."

        else:
            # the word is valid
            self.used_words.add(word)
            self.required_start_letter = word[-1]
            observation = f"[Player {player_id}] - {word}"
            info["info"] =  f"Player {player_id} gave the word {word}"

        if self.render:
            print(f"\n[Player {player_id}] {word}")
            if done:
                print(f"Game concluded: {info['reason']}")

        return observation, reward, done, info 
