import os, re, random, openai 
from evals.text_games.game_collection.game_interface import GameInterface
from tqdm import tqdm

# Ensure the OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

class TrafficStopGame(GameInterface):
    def __init__(self, max_turns=20, render=False):
        """
        Initialize the Traffic Stop Game.
        Args:
            max_turns (int): Maximum number of turns before the game ends in a draw.
            initial_temperature (float): Starting temperature for GPT models.
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Traffic Stop Game"
        self.max_turns = max_turns
        self.render = render

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0

        # Initialize GPT models' temperature and drunk status
        drunk_person = random.choice(["alice", "bob"])
        sober_person = "alice" if drunk_person=="bob" else "bob"
        self.models = {
            drunk_person: {
                "temperature": 1.4,
                "frequency_penalty": 0.0,
                "presence_penalty": -1.0,
                "is_drunk": True
            },
            sober_person: {
                "temperature": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "is_drunk": False
            }
        }

        # Generate initial prompts for both GPT models
        self.person_prompts = {
            "alice": self._generate_person_prompt(
                person_name="alice",
                is_drunk=self.models["alice"]["is_drunk"]
            ),
            "bob": self._generate_person_prompt(
                person_name="bob",
                is_drunk=self.models["bob"]["is_drunk"]
            )
        }

        # Initialize player prompts
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }

        self.internal_chat_history = ""

        if self.render:
            print(f"Drunk Model: [alice]" if drunk_person == "alice" else f"Drunk Model: [bob]")

        return player_prompts

    def _generate_person_prompt(self, person_name, is_drunk):
        """
        Generate the initial prompt for a GPT model.
        """
        prompt = (
            f"You are roleplaying as {person_name} being pulled over by the police. They suspect you "
            f"of drunk driving, but you did not drink anything at all. Answer the questions accordingly."
        )
        if is_drunk:
            prompt += f"\n You should be very subtly in-consistent in your answers."
        return prompt

    def _generate_player_prompt(self, player_id):
        """
        Generate the prompt for a player.
        """
        prompt = (
            f"You are Player {player_id}. You can interact with [Alice] and [Bob].\n"
            "On your turn, you can either ask a question or decide to arrest one of them.\n"
            "To ask a question, prefix it with [Alice] or [Bob]. For example:\n"
            "- [ask - alice] How are you today?\n"
            "- [ask - bob] What is your favorite color?\n"
            "To arrest a person, use the following format:\n"
            "- [arrest - alice]\n"
            "- [arrest - bob]\n"
            "If you decide to arrest, it ends your turn.\n"
            "Remember, arresting the 'drunk' person wins you the game, while arresting the sober person causes you to lose.\n"
            f"If the other police officer (Player {player_id}) arrests the drunk person before you, you lose.\n"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """
        Return valid actions for the given player.
        Players can always choose to ask questions or arrest.
        """
        return None  # Any action is potentially valid; validation occurs in `step`

    def get_info(self):
        """Return additional information."""
        return {'num_turns': self.turn}

    def step(self, player_id, action):
        """
        Process the player's action.
        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's action.
        Returns:
            state (str): The action taken.
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        response = None 
        observation, reward, done, info = None, None, False, {}
        action_stripped = action.strip()
        self.turn += 1

        # Check if the player wants to arrest someone
        arrest_match = re.match(r"\[arrest\s*-\s*(alice|bob)\]", action_stripped, re.IGNORECASE)
        if arrest_match:
            arrested_person = arrest_match.group(1).lower()
            if arrested_person not in self.models:
                # Invalid person to arrest
                reward = {player_id:-1, 1-player_id:0}
                info["reason"] = f"Invalid person to arrest: {arrest_match.group(1)}."
            else:
                # Determine if the arrested person is drunk
                if self.models[arrested_person]["is_drunk"]:
                    # Correct arrest
                    reward = {player_id:1, 1-player_id:-1}
                    info["reason"] = f"Player {player_id} correctly arrested {arrest_match.group(1)}."
                else:
                    # Incorrect arrest
                    reward = {player_id:-1, 1-player_id:1}
                    info["reason"] = f"Player {player_id} incorrectly arrested {arrest_match.group(1)}."
            done = True

        else:
            # Assume the player is asking a question
            # Extract which person they're addressing
            question_match = re.match(r"\[ask\s*-\s*(alice|bob)\]\s*(.+)", action_stripped, re.IGNORECASE)
            if question_match:
                person = question_match.group(1).lower()
                question = question_match.group(2).strip()

                if person not in self.models:
                    # Invalid person addressed
                    done = True 
                    reward = {player_id:-1, 1-player_id:0}
                    info["reason"] = f"Invalid person addressed: {question_match.group(1)}."
                else:
                    self.internal_chat_history += f"\n\n[Officer {player_id} to {person}]: {question}"
                    # Send the question to the respective GPT model
                    response = self._ask_person(person)
                    self.internal_chat_history += f"\n\n[{person}]: {response}"
                    #observation = f"\n[Office {player_id} to {person}]: {question}\n[{person}]: {response}"
                    observation = f"{action}\n[{person}]: {response}"

                    # Check if max turns reached
                    if self.max_turns and self.turn >= self.max_turns:
                        done = True
                        reward = {player_id:-1, 1-player_id:-1}
                        info["reason"] = f"Maximum number of turns ({self.max_turns}) reached. The game is a draw."
            else:
                # Invalid action format
                done = True 
                reward = {player_id:-1, 1-player_id:0}
                info["reason"] = f"Player {player_id} - Invalid action format."


        if self.render:
            print("---------------------------------")
            print(self.internal_chat_history)
            if done:
                print("Game is done", info["reason"])
            #print(f"[Player {player_id}, Turn: {self.turn}] Action: '{action}'\n")
            #if response:
            #    print(response)

        return observation, reward, done, info

    def _ask_person(self, person):
        """
        Send a question to the specified person (GPT model) and return the response.
        Args:
            person (str): 'alice' or 'bob'.
            question (str): The question to ask.
        Returns:
            response (str): The GPT model's response.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.person_prompts[person]},
                    {"role": "user", "content": self.internal_chat_history}
                ],
                temperature=self.models[person]["temperature"],
                presence_penalty=self.models[person]["presence_penalty"],
                frequency_penalty=self.models[person]["frequency_penalty"],
                max_tokens=150,
                n=1,
                stop=None,
            )

            # Extract the assistant's reply
            reply = response.choices[0].message['content'].strip()

            # Remove person's name and square brackets if they occur at the start of the reply
            cleaned_reply = re.sub(r'^\s*\[[^\]]+\]:\s*', '', reply)

            return cleaned_reply

        except Exception as e:
            # Handle API errors
            if self.render:
                print(f"Error communicating with {person.capitalize()}: {e}")
            return "Error: Unable to process the request."
