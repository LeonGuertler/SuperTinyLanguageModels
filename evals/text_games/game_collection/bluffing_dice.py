# TODO
"""
Bluffing Dice Game
- each player rolls three dice at the beginning of the game
- the first player calls what the threshold his (i.e. total count greater x)
- afterwards, taking turns, each player can do one of three things [call] <amount> (increase the number), [accept]
- if a player accepts, the dice are revealed. If the total count is higher than the current <amount> the player who 
accepted losses, if it is lower, the player who accepted wins
"""
import random
import re
from evals.text_games.game_collection.game_interface import GameInterface

class BluffingDiceGame(GameInterface):
    def __init__(self, render=False):
        """
        Initialize the Bluffing Dice Game.
        Args:
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Bluffing Dice Game"
        self.render = render

    def reset(self):
        """Reset the game to its initial state."""
        self.current_claim = 0
        self.turn = 0 
        self.dice_rolls = {
            0: [random.randint(1, 6) for _ in range(3)],
            1: [random.randint(1, 6) for _ in range(3)],
        }

        if self.render:
            print(f"Total Dice: {sum(self.dice_rolls[0])+sum(self.dice_rolls[1])}\n")

        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1),
        }
        return player_prompts

    def _generate_player_prompt(self, player_id):
        """
        Generate the initial prompt for a player, including their own die rolls.
        """
        dice = self.dice_rolls[player_id]
        prompt = (
            f"You are playing the Bluffing Dice Game.\n"
            f"You have rolled three dice and got: {dice[0]}, {dice[1]}, {dice[2]}.\n"
            "In each round, both players have three dice with secret totals.\n"
            "Players take turns making claims about the combined total of both dice (e.g., '[claim] <amount>').\n"
            "On your turn, you can either make a claim or respond to the opponent's claim.\n"
            "If you think the opponent is bluffing, you can '[reveal] their bluff'.\n"
            "- If the combined total is greater than the current <amount>, you lose.\n"
            "- If the combined total is less than or equal to the current <amount>, you win.\n"
            "If you think the amount is higher, you can '[claim] <amount>', the amount you claim has to be higher than the previous claim."
            "For example, you claim might look like this: '[claim] 5'\n"
            "If you make a claim, make sure to include the claimed number.\n"
            "The game starts with claim: 0"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        return ["[claim] <amount>", "[reveal]"]

    def get_info(self):
        """Return additional information."""
        return {'num_turns': self.turn}

    def step(self, player_id, action):
        """
        Process the player's action.
        Args:
            player_id (int): The player's ID.
            action (str): The player's action.
        Returns:
            state (str): The action taken.
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        action_lower = action.strip().lower()

        self.turn += 1 
        # no need for a turn limit as the claims have to be progressively higher anyway

        if self.render:
            print(f"\n[Player {player_id}, player_total: {sum(self.dice_rolls[player_id])}, current_claim: {self.current_claim}] {action}")


        # check if player is making a claim or accepting
        if "[reveal]" in action_lower:
            # check if total count is greater than current claim
            dice_sum = sum(self.dice_rolls[0]) + sum(self.dice_rolls[1])
            if dice_sum > self.current_claim:
                # the player making the claim wins
                if self.render:
                    print(f"Player {player_id} revealed, but Player {player_id} wasn't bluffing.")
                return None, {1-player_id:1, player_id:-1}, True, {"reason": f"Player {player_id} revealed, but Player {player_id} wasn't bluffing."}
            else:
                # the player revealing wins
                if self.render:
                    print(f"Player {player_id} revealed, and was right.")
                return None, {player_id:1, 1-player_id:-1}, True, {"reason": f"Player {player_id} revealed, and was right."}

        elif "[claim]" in action_lower:
            # parse to find the new claim number
            amount = re.search(r'\[claim\]\s*(\d+)', action_lower)
            if not amount:
                # a number must be provided
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} made a claim without number."}
            amount = int(amount.group(1))
            if amount <= self.current_claim:
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried to underbid the current claim."}
            elif amount > 36:
                # to prevent infitine loops
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} made an unrealistic claim [{amount}]"}
            else:
                # legit claim
                prev_claim = self.current_claim
                self.current_claim = amount 
                return f"[Player {player_id}, Current claim: {self.current_claim}] {action}", None, False, {"info": f"Player {player_id} claimed {amount}. (prev. claim{prev_claim})"}
        else:
            # illegal action
            return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} made an illegal action ({action})"}