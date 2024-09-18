# human_agent.py

class HumanAgent:
    def __init__(self, agent_id, prompt_prefix="> "):
        """
        Initialize the HumanAgent.

        Args:
            player_id (int): The ID of the player.
            prompt_prefix (str): The prefix to display when prompting for input.
        """
        self.player_id = player_id
        self.prompt_prefix = prompt_prefix
        self.history = []  # Stores tuples of (prompt, response)
        self.main_prompt = ""
        self.agent_id = agent_id

    def reset(self, main_prompt):
        """
        Reset the agent with a new main prompt.

        Args:
            main_prompt (str): The main prompt or instructions for the player.
        """
        self.main_prompt = main_prompt
        self.history = []  # Clear history for a new game

    def get_action(self, immediate_state, valid_actions=None):
        """
        Prompt the human user for an action.

        Args:
            immediate_state (str): The current state of the game specific to the player.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The action input by the user.
        """
        print(f"######################## current input ########################")
        # use the main prompt and state history to construct the agent input
        prompt = self.main_prompt + "\n"
        for h_state, h_action in self.history:
            prompt += f"{h_state}\n\n{h_action}\n"

        # append the current state to all of this
        prompt += f"{immediate_state}\n"
        # Combine main prompt and immediate state for context
        #prompt = f"{self.main_prompt}\n{immediate_state}"
        print(f"\n[Player {self.player_id} - Human Agent]")
        print(f"{prompt}")
        if valid_actions:
            print(f"Valid actions: {valid_actions}")
        while True:
            action = input(f"{self.prompt_prefix}Enter your action: ").strip()
            if valid_actions:
                if action in valid_actions:
                    self.history.append((
                        immediate_state, 
                        f"Previous Action: {action}"
                    ))
                    return action, prompt
                else:
                    print("Invalid action. Please choose from the valid actions listed.")
            else:
                if action:
                    self.history.append((
                        immediate_state,
                        f"Previous Action: {action}"
                    ))
                    return action, prompt
                else:
                    print("Action cannot be empty. Please enter a valid action.")

    def get_history(self):
        """
        Retrieve the conversation history.

        Returns:
            list: A list of tuples containing (prompt, response).
        """
        return self.history
