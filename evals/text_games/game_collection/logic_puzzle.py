# TODO

"""
There is a fixed list of steps that can be used to solve the puzzle.
The Puzzle posing model provides the logic puzzle, and a viable solution
( as a list of the steps to be used )
The viable solution is verified to be legit.
Subsequently, the other model is allowed to try to solve it.
"""

import random
from evals.text_games.game_collection.game_interface import GameInterface

class LogicPuzzleGame(GameInterface):
    def __init__(self, render=False):
        """
        Initialize the Logic Puzzle Game.
        Args:
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Logic Puzzle"
        self.render = render
        self.puzzles = self._load_puzzles()
        self.reset()

    def _load_puzzles(self):
        """Load or define a set of logic puzzles with solutions."""
        # For simplicity, we'll define a small set of puzzles
        return [
            {
                "puzzle": "There are three boxes, one labeled 'Apples', one labeled 'Oranges', and one labeled 'Apples and Oranges'. All labels are incorrect. You may pick one fruit from one box. How can you label all boxes correctly?",
                "solution_steps": [
                    "Pick a fruit from the box labeled 'Apples and Oranges'.",
                    "Since all labels are incorrect, this box must contain only apples or only oranges.",
                    "If you pick an apple, this box is 'Apples'.",
                    "The box labeled 'Oranges' cannot be 'Oranges', so it must be 'Apples and Oranges'.",
                    "The remaining box is 'Oranges'."
                ]
            },
            # Add more puzzles as needed
        ]

    def reset(self):
        """Reset the game to its initial state."""
        self.game_over = False
        self.selected_puzzle = random.choice(self.puzzles)
        self.player_prompts = {
            0: self._generate_puzzle_pose_prompt(),
            1: self._generate_puzzle_solve_prompt()
        }
        return self.player_prompts

    def _generate_puzzle_pose_prompt(self):
        """
        Generate the prompt for the puzzle-posing player.
        """
        prompt = (
            "You are to present a logic puzzle to another player.\n"
            "Select a puzzle and provide a valid solution as a list of logical steps.\n"
            "Your solution will be verified before being presented to the other player.\n"
            "Puzzle:\n"
            f"{self.selected_puzzle['puzzle']}\n"
            "Provide your solution steps as a numbered list.\n"
        )
        return prompt

    def _generate_puzzle_solve_prompt(self):
        """
        Generate the prompt for the puzzle-solving player.
        """
        prompt = (
            "You have been given a logic puzzle to solve.\n"
            "Provide your solution as a list of logical steps.\n"
            "Puzzle:\n"
            f"{self.selected_puzzle['puzzle']}\n"
            "Provide your solution steps as a numbered list.\n"
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
            # Puzzle-posing player provides the solution
            self.pose_solution = action.strip().split('\n')
            # Verify the solution (simplified verification)
            if self._verify_solution(self.pose_solution, self.selected_puzzle['solution_steps']):
                info['info'] = "Valid solution provided. Waiting for Player 1 to solve the puzzle."
            else:
                reward[0] = -1
                reward[1] = 1
                done = True
                info['reason'] = "Invalid solution provided. Player 1 wins."
        else:
            # Puzzle-solving player attempts to solve
            self.solve_solution = action.strip().split('\n')
            # Check if the solution matches the verified one
            if self._verify_solution(self.solve_solution, self.selected_puzzle['solution_steps']):
                reward[0] = 1
                reward[1] = 1
                info['reason'] = "Correct solution provided. Both players win."
            else:
                reward[0] = -1
                reward[1] = -1
                info['reason'] = "Incorrect solution provided. Both players lose."
            done = True

        if self.render:
            print(f"[Player {player_id}] '{action}'")

        return action, reward, done, info

    def _verify_solution(self, provided_steps, correct_steps):
        """Simplified solution verification."""
        # For simplicity, we'll check if the provided steps contain the key points
        correct_keywords = set(' '.join(correct_steps).lower().split())
        provided_keywords = set(' '.join(provided_steps).lower().split())
        return correct_keywords.issubset(provided_keywords)
