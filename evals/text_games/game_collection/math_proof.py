import os, subprocess, shutil, random, uuid 
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

        # confirm that lean is installed
        if not shutil.which("lean"):
            raise EnvironmentError("Lean is not installed or not found in PATH. Please install Lean to proceed.")
       


    def reset(self):
        """Reset the game to its initial state."""
        self.game_over = False
        self.phase = 1  # Phase 1: Player 0 provides statement and proof
        self.statement = ""
        self.player_prompts = {
            0: self._generate_proof_creator_prompt(),
            1: None  # Player 1 waits until Phase 2
        }
        return self.player_prompts

    def _generate_proof_creator_prompt(self):
        """
        Generate the prompt for the proof-creator player (Player 0).
        """
        prompt = (
            "You are the Proof-Creator in the 'Math Proof' game.\n"
            "Your task is to provide a mathematical statement and its formal proof in Lean.\n"
            "Ensure that your statement is clear and that your proof is correct.\n"
            "Format your submission as follows:\n"
            "Statement:\n"
            "<Lean-formatted statement>\n"
            "Proof:\n"
            "<Lean-formatted proof>\n"
            "Example:\n"
            "Statement:\n"
            "theorem sum_two_evens (n m : ℤ) (h1 : even n) (h2 : even m) : even (n + m) :=\n"
            "Proof:\n"
            "begin\n"
            "  use (n / 2) + (m / 2),\n"
            "  rw [← even_iff_two_dvd] at h1 h2,\n"
            "  rw [h1, h2],\n"
            "  ring,\n"
            "end\n"
        )
        return prompt

    def _generate_proof_recreator_prompt(self):
        """
        Generate the prompt for the proof-recreator player (Player 1).
        """
        prompt = (
            "You are the Proof-Recreator in the 'Math Proof' game.\n"
            "You have been given a mathematical statement to prove.\n"
            "Your task is to provide a formal proof in Lean that correctly proves the given statement.\n"
            "Ensure that your proof is correct and adheres to Lean's syntax.\n"
            "Format your submission as follows:\n"
            "Proof:\n"
            "<Lean-formatted proof>\n"
            "Example:\n"
            "Proof:\n"
            "begin\n"
            "  use (n / 2) + (m / 2),\n"
            "  rw [← even_iff_two_dvd] at h1 h2,\n"
            "  rw [h1, h2],\n"
            "  ring,\n"
            "end\n"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """No restrictions on actions."""
        return None

    def get_info(self):
        """Return additional information."""
        return {}

    def step(self, player_id, action):
        """
        Process the player's action.
        Args:
            player_id (int): ID of the player (0 or 1).
            action (str): The action/input provided by the player.
        Returns:
            Tuple: (response, reward, done, info)
        """
        if self.game_over:
            if self.render:
                print("Game is already over.")
            return None, {0: 0, 1: 0}, True, {"reason": "Game has already ended."}

        if self.phase == 1 and player_id == 0:
            # Phase 1: Player 0 provides statement and proof
            try:
                statement, proof = self._parse_proof_creator_submission(action)
                self.statement = statement
                is_valid = self._verify_proof(statement, proof, player_id=0)
                if is_valid:
                    if self.render:
                        print("Player 0 provided a valid proof.")
                    # Proceed to Phase 2
                    self.phase = 2
                    self.player_prompts[1] = self._generate_proof_recreator_prompt()
                    info = {"info": "Valid proof provided. Player 1, please provide your proof."}
                else:
                    if self.render:
                        print("Player 0 provided an invalid proof.")
                    # Player 0 loses
                    self.game_over = True
                    reward = {0: -1, 1: 1}
                    info = {"reason": "Invalid proof provided by Player 0. Player 1 wins."}
                    return None, reward, True, info
            except Exception as e:
                if self.render:
                    print(f"Error parsing Player 0's submission: {e}")
                # Player 0 loses due to error
                self.game_over = True
                reward = {0: -1, 1: 1}
                info = {"reason": f"Error parsing submission: {e}. Player 1 wins."}
                return None, reward, True, info

            # Successful Phase 1
            return None, {0: 1, 1: 0}, False, info

        elif self.phase == 2 and player_id == 1:
            # Phase 2: Player 1 provides proof based on the statement
            try:
                proof = self._parse_proof_recreator_submission(action)
                is_valid = self._verify_proof(self.statement, proof, player_id=1)
                if is_valid:
                    if self.render:
                        print("Player 1 provided a valid proof.")
                    # Player 1 succeeds, Player 0 loses
                    self.game_over = True
                    reward = {0: -1, 1: 1}
                    info = {"reason": "Player 1 provided a valid proof. Player 0 loses."}
                else:
                    if self.render:
                        print("Player 1 provided an invalid proof.")
                    # Player 1 loses
                    self.game_over = True
                    reward = {0: 1, 1: -1}
                    info = {"reason": "Player 1 provided an invalid proof. Player 0 wins."}
            except Exception as e:
                if self.render:
                    print(f"Error parsing Player 1's submission: {e}")
                # Player 1 loses due to error
                self.game_over = True
                reward = {0: 1, 1: -1}
                info = {"reason": f"Error parsing submission: {e}. Player 0 wins."}
                return None, reward, True, info

            return None, reward, True, info

        else:
            if self.render:
                print(f"Unexpected action from Player {player_id} in phase {self.phase}.")
            return None, {0: 0, 1: 0}, False, {"info": "Unexpected action."}

    def _parse_proof_creator_submission(self, submission):
        """
        Parse the submission from the Proof-Creator (Player 0).
        Expects a specific format with 'Statement:' and 'Proof:' sections.
        Args:
            submission (str): The raw submission text.
        Returns:
            Tuple[str, str]: (statement, proof)
        """
        try:
            parts = submission.strip().split("Proof:")
            statement_part = parts[0].replace("Statement:", "").strip()
            proof_part = parts[1].strip()
            if not statement_part.startswith("theorem") and not statement_part.startswith("lemma"):
                raise ValueError("Statement must start with 'theorem' or 'lemma'.")
            return statement_part, proof_part
        except IndexError:
            raise ValueError("Submission must contain both 'Statement:' and 'Proof:' sections.")

    def _parse_proof_recreator_submission(self, submission):
        """
        Parse the submission from the Proof-Recreator (Player 1).
        Expects a 'Proof:' section.
        Args:
            submission (str): The raw submission text.
        Returns:
            str: The proof in Lean syntax.
        """
        try:
            proof_part = submission.strip().replace("Proof:", "").strip()
            return proof_part
        except Exception:
            raise ValueError("Submission must contain a 'Proof:' section.")

    def _verify_proof(self, statement, proof, player_id):
        """
        Verify the provided proof using Lean.
        Args:
            statement (str): The Lean-formatted statement.
            proof (str): The Lean-formatted proof.
            player_id (int): ID of the player providing the proof.
        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        # Generate a unique filename to prevent conflicts
        unique_id = uuid.uuid4().hex
        lean_filename = f"temp_proof_{unique_id}.lean"

        # Construct the Lean theorem with the provided proof
        lean_content = f"{statement} :=\n{proof}\n"

        try:
            # Write the Lean file
            with open(lean_filename, 'w') as f:
                f.write(lean_content)

            # Invoke Lean to check the proof
            result = subprocess.run(['lean', lean_filename], capture_output=True, text=True, timeout=10)

            if self.render:
                if result.returncode == 0:
                    print("Lean verification succeeded.")
                else:
                    print(f"Lean verification failed with errors:\n{result.stderr}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            if self.render:
                print("Lean verification timed out.")
            return False
        except Exception as e:
            if self.render:
                print(f"Error during Lean verification: {e}")
            return False
        finally:
            # Clean up the Lean file
            if os.path.exists(lean_filename):
                os.remove(lean_filename)

    def get_player_prompts(self):
        """
        Get the current prompts for both players.
        Returns:
            dict: Mapping of player IDs to their prompts.
        """
        return self.player_prompts

