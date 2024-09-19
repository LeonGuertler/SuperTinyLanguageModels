# TODO


"""
One model provides a mathematical statement and it's proof (using a formal proofing language so it can be verified easily).
Only the statement is shown to the other model and it needs to try to proof it.
"""

# TODO - Need to change this to use a formal proofing language

import random
from evals.text_games.game_collection.game_interface import GameInterface

class MathProofGame(GameInterface):
    def __init__(self, render=False):
        """
        Initialize the Math Proof Game.
        Args:
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Math Proof"
        self.render = render
        self.statements = self._load_statements()
        self.reset()

    def _load_statements(self):
        """Load or define a set of mathematical statements with proofs."""
        # For simplicity, we'll define a small set of statements
        return [
            {
                "statement": "Prove that the sum of two even numbers is even.",
                "proof": [
                    "Let n and m be even numbers.",
                    "By definition of even numbers, there exist integers k and l such that n = 2k and m = 2l.",
                    "Then n + m = 2k + 2l = 2(k + l).",
                    "Since k + l is an integer, n + m is divisible by 2.",
                    "Therefore, n + m is even."
                ]
            },
            # Add more statements as needed
        ]

    def reset(self):
        """Reset the game to its initial state."""
        self.game_over = False
        self.selected_statement = random.choice(self.statements)
        self.player_prompts = {
            0: self._generate_proof_pose_prompt(),
            1: self._generate_proof_solve_prompt()
        }
        return self.player_prompts

    def _generate_proof_pose_prompt(self):
        """
        Generate the prompt for the proof-posing player.
        """
        prompt = (
            "You are to present a mathematical statement along with its formal proof.\n"
            "Your proof should use formal mathematical language and logical steps.\n"
            "Statement:\n"
            f"{self.selected_statement['statement']}\n"
            "Provide your proof as a numbered list of steps.\n"
        )
        return prompt

    def _generate_proof_solve_prompt(self):
        """
        Generate the prompt for the proof-solving player.
        """
        prompt = (
            "You have been given a mathematical statement to prove.\n"
            "Provide your proof using formal mathematical language and logical steps.\n"
            "Statement:\n"
            f"{self.selected_statement['statement']}\n"
            "Provide your proof as a numbered list of steps.\n"
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
            # Proof-posing player provides the proof
            self.pose_proof = action.strip().split('\n')
            # Verify the proof (simplified verification)
            if self._verify_proof(self.pose_proof, self.selected_statement['proof']):
                info['info'] = "Valid proof provided. Waiting for Player 1 to provide their proof."
            else:
                reward[0] = -1
                reward[1] = 1
                done = True
                info['reason'] = "Invalid proof provided. Player 1 wins."
        else:
            # Proof-solving player attempts to prove
            self.solve_proof = action.strip().split('\n')
            # Check if the proof is valid
            if self._verify_proof(self.solve_proof, self.selected_statement['proof']):
                reward[0] = 1
                reward[1] = 1
                info['reason'] = "Correct proof provided. Both players win."
            else:
                reward[0] = -1
                reward[1] = -1
                info['reason'] = "Incorrect proof provided. Both players lose."
            done = True

        if self.render:
            print(f"[Player {player_id}] '{action}'")

        return action, reward, done, info

    def _verify_proof(self, provided_steps, correct_steps):
        """Simplified proof verification."""
        # For simplicity, we'll check if the provided proof contains the key logical steps
        correct_keywords = set(' '.join(correct_steps).lower().split())
        provided_keywords = set(' '.join(provided_steps).lower().split())
        return correct_keywords.issubset(provided_keywords)
