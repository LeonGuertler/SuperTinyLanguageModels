"""
basic wrappers for standard LLMs to make them perform better
"""
import torch, openai

class LLMWrapperInterface:
    def __init__(self, model, tokenizer, **kwargs):
        """Initialize with the provided model and tokenizer."""
        raise NotImplementedError

    def get_action(self, state, valid_actions=None):
        """
        Generate an action based on the current state.

        Args:
            state (str): The textual representation of the game state.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The chosen action.
        """
        raise NotImplementedError

class LLMWrapper:
    def __init__(self, agent_id, model, device='cpu', max_tokens=100):
        """
        Initialize the LLMWrapper agent.

        Args:
            agent_id (int): A unique identifier for the agent.
            model: The pre-loaded language model.
            device (str): 'cpu' or 'cuda', for model inference.
            max_tokens (int): Maximum number of tokens to generate.
        """
        self.agent_id = agent_id
        self.model = model
        self.device = device
        self.max_tokens = max_tokens
        self.history = []  # Stores tuples of (prompt, response)
        self.main_prompt = ""

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
        Use the language model to generate an action.

        Args:
            immediate_state (str): The current state of the game specific to the player.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The generated action from the model.
            prompt (str): The full prompt sent to the model.
        """
        print(f"######################## current input ########################")

        # Use the main prompt and state history to construct the agent input
        prompt = self.main_prompt + "\n"
        for h_state, h_action in self.history:
            prompt += f"{h_state}\n\n{h_action}\n"

        # Append the current state to all of this
        prompt += f"{immediate_state}\n"
        
        # If valid actions are provided, include them in the prompt for context
        if valid_actions:
            prompt += f"\nValid actions: {', '.join(valid_actions)}\n"

        print(f"\n[Player - LLM Agent]")
        print(f"{prompt}")

        # Generate a response from the model
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(input_ids, max_length=self.max_tokens + len(input_ids[0]))

        # Decode the output and extract the generated action
        generated_text = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
        action = generated_text[len(prompt):].strip().split("\n")[0]  # Extract only the first action

        # Check if the action is valid (if valid actions are provided)
        if valid_actions and action not in valid_actions:
            print("Generated action is not in the valid actions list. Picking a random valid action instead.")
            action = valid_actions[0]  # Or any strategy for picking valid actions

        # Add to history
        self.history.append((
            immediate_state,
            f"Previous Action: {action}"
        ))

        return action, prompt

    def get_history(self):
        """
        Retrieve the conversation history.

        Returns:
            list: A list of tuples containing (prompt, response).
        """
        return self.history



class GPT4Agent:
    def __init__(self, agent_id, api_key, max_tokens=100, model_name="gpt-4"):
        """
        Initialize the GPT-4 agent.

        Args:
            agent_id (int): A unique identifier for the agent.
            api_key (str): Your OpenAI API key.
            max_tokens (int): Maximum number of tokens to generate.
            model_name (str): The model to use, e.g., 'gpt-4'.
        """
        self.agent_id = agent_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.history = []  # Stores tuples of (prompt, response)
        self.main_prompt = ""

        openai.api_key = self.api_key

    def reset(self, main_prompt):
        """
        Reset the agent with a new main prompt.

        Args:
            main_prompt (str): The main prompt or instructions for the player.
        """
        self.main_prompt = main_prompt
        self.history = []  # Clear history for a new game

    def get_action(self, state, valid_actions=None):
        """
        Use the OpenAI GPT-4 API to generate an action.

        Args:
            state (str): The current state of the game specific to the player.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The generated action from the model.
            prompt (str): The full prompt sent to the model.
        """
        #print(f"######################## current input ########################")

        # Use the main prompt and state history to construct the agent input
        prompt = self.main_prompt + "\n"
        for h_state, h_action in self.history:
            prompt += f"{h_state}\n\n{h_action}\n"

        # Append the current state to all of this
        prompt += f"{state}\n"

        # If valid actions are provided, include them in the prompt for context
        if valid_actions:
            prompt += f"\nValid actions: {', '.join(valid_actions)}\n"

        #print(f"\n[Player - GPT-4 API Agent]")
        #print(f"{prompt}")

        # Construct the messages for the chat completion
        messages = [{"role": "system", "content": self.main_prompt}]
        for h_state, h_action in self.history:
            messages.append({"role": "user", "content": h_state})
            messages.append({"role": "assistant", "content": h_action})
        messages.append({"role": "user", "content": state})


        # Call the GPT-4 API to generate a response
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Extract the generated action from the API response
        generated_text = response['choices'][0]['message']['content'].strip()
        action = generated_text.split("\n")[0]  # Extract only the first action


        # Check if the action is valid (if valid actions are provided)
        if valid_actions and action not in valid_actions:
            #print("Generated action is not in the valid actions list. Picking a random valid action instead.")
            action = valid_actions[0]  # Or any strategy for picking valid actions

        # Add to history
        self.history.append((
            state,
            f"Previous Action: {action}"
        ))

        return action, prompt

    def get_history(self):
        """
        Retrieve the conversation history.

        Returns:
            list: A list of tuples containing (prompt, response).
        """
        return self.history