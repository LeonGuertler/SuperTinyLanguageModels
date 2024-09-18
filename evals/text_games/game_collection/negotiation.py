import random, re
from evals.text_games.game_collection.game_interface import GameInterface


class NegotiationGame(GameInterface):
    def __init__(self, max_turns=10, render=False):
        """
        Initialize the Negotiation Game.
        Args:
            max_turns (int): Maximum number of turns for the game.
        """
        self.name = "Negotiation"
        self.max_turns = max_turns
        self.render = render
        self.resource_names = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
        self.base_values = {
            "Wheat": 5,
            "Wood": 10,
            "Sheep": 15,
            "Brick": 25,
            "Ore": 40
        }
        self.reset()  # Ensure reset is called during initialization

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0  # Current turn number
        self.current_player = 0  # Player 0 starts
        self.game_over = False
        self.current_offer = None  # Initialize current_offer
        self.message_buffers = {0: [], 1: []}  # Buffers to store messages for each player

        # Generate random quantities of resources for each player (1-5 for each resource)
        self.player_resources = {
            0: {resource: random.randint(5, 25) for resource in self.resource_names},
            1: {resource: random.randint(5, 25) for resource in self.resource_names}
        }

        # Generate player-specific values for each resource type (±20% of base value, capped at 40)
        self.player_values = {}
        for player_id in [0, 1]:
            self.player_values[player_id] = {}
            for resource in self.resource_names:
                base_value = self.base_values[resource]
                variation = int(0.2 * base_value)
                min_value = max(base_value - variation, 5)
                max_value = min(base_value + variation, 40)
                value = random.randint(min_value, max_value)
                self.player_values[player_id][resource] = value

        # Keep track of initial resources for value calculation
        self.initial_resources = {
            0: self.player_resources[0].copy(),
            1: self.player_resources[1].copy()
        }

        # Keep track of trades made
        self.trades_made = []

        # Initialize the previous action (None at the start)
        self.previous_action = {0: None, 1: None}

        # Return the main prompts for both players
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }
        return player_prompts

    def _generate_player_prompt(self, player_id):
        """
        Generate the initial prompt for a player.
        """
        resources = self.player_resources[player_id]
        resource_values = self.player_values[player_id]
        resource_value_list = "; ".join([f"{resources[res]} {res} (Value: {resource_values[res]})" for res in resources.keys()])
        prompt = (
            "You have some resources and your task is to trade such that the total value of your resources increased"
            f"The resources and associated values you currently have are: {resource_value_list}"
            "At each turn you can talk to your opponent or make an exlicit trade offer.\n"
            "Format: 'Offer: I give [your resources]; You give [their resources]'.\n"
            "Example: 'Offer: I give 2 Wheat, 1 Ore; You give 3 Sheep'.\n"
            "If your opponent made a trade offer, you need to 'Accept' or 'Deny' it."
            "In the same reply, after accepting or denying it, you can talk or make an offer as well."
            "Just make sure your reply starts either with 'Accept' or 'Deny'"
        )
        return prompt

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        if self.current_offer and self.current_offer['to_player'] == player_id:
            # After an offer is made to the player, they can only 'Accept' or 'Deny'
            return ['Accept', 'Deny']
        else:
            # Player can make an offer or send a message
            return None  # No restrictions; actions will be validated in `step`

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
            state (str): The new state after the action (the previous action).
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        # Other player ID
        other_player_id = 1 - player_id

        # Process the action
        action_stripped = action.strip()
        done = False
        reward = None

        if action_stripped.startswith("Offer:"):
            # Player is making an offer
            parsed_offer = self._parse_offer(action_stripped)
            if parsed_offer:
                self.current_offer = {
                    'from_player': player_id,
                    'to_player': other_player_id,
                    'offer': parsed_offer
                }
                info = {"info": "Offer made."}
            else:
                info = {"reason": "Invalid offer format."}
                # Invalid offer; agent receives -1 reward
                reward = {player_id: -1, other_player_id: 0}
                done = True
        elif action_stripped.lower().replace(".", "").startswith("accept"):
            if self.current_offer and self.current_offer['to_player'] == player_id:
                # Trade is accepted
                success, trade_result = self._execute_trade(self.current_offer['offer'], player_id)
                if success:
                    self.trades_made.append(self.current_offer)
                    info = {"info": "Trade accepted."}
                else:
                    # Determine who lacked resources
                    if trade_result == 'proposer':
                        # Assign rewards: proposer loses
                        reward = {self.current_offer['from_player']: -1, player_id: 0}
                        done = True
                        info = {"reason": "Proposer lacks resources. Game over."}
                    elif trade_result == 'acceptor':
                        # Assign rewards: acceptor loses
                        reward = {player_id: -1, other_player_id: 0}
                        done = True
                        info = {"reason": "Acceptor lacks resources. Game over."}
                self.current_offer = None
            else:
                info = {"info": "General message"}
                reward = None 
                done = False

        elif action_stripped.lower().replace(".", "").startswith("deny"):
            if self.current_offer and self.current_offer['to_player'] == player_id:
                # Trade is denied
                self.current_offer = None
                info = {"info": "Trade denied."}
            else:
                info = {"info": "General message"}
                reward = None 
                done = False
        else:
            info = {"info": "General message"}
            reward = None 
            done = False 
            

        # Increment turn and check for game over
        if not done:
            self.turn += 1
            if self.turn >= self.max_turns:
                done = True
                reward, reason = self._calculate_rewards()
                info = {"reason": reason}

        if self.render:
            # print information
            current_val, current_gain = self._get_inventory_value(player_id=player_id)
            print(f"[Player {player_id}, Inventory Value: {current_val}, Gain: {current_gain}] '{action}'")
        return action, reward, done, info

    def _parse_offer(self, action):
        """
        Parse the offer action to extract the resources being traded.
        Expected format: "Offer: I give [resources]; You give [resources]"
        Returns a dictionary with 'my_offer' and 'their_offer' as dictionaries of resources and quantities.
        """
        try:
            parts = action.split("Offer:")[1].strip()
            offer_parts = parts.split("; You give ")
            if len(offer_parts) != 2:
                return None
            my_offer_str = offer_parts[0].split("I give ")[1].strip()
            their_offer_str = offer_parts[1].strip().replace(".", "")

            my_offer = self._parse_resource_list(my_offer_str)
            their_offer = self._parse_resource_list(their_offer_str)

            if not my_offer or not their_offer:
                return None

            return {'my_offer': my_offer, 'their_offer': their_offer}
        except Exception as exp:
            return None

    def _parse_resource_list(self, resource_str):
        """
        Parse a string of resources and quantities into a dictionary.
        Example input: "2 Wheat, 1 Ore"
        Returns a dictionary: {'Wheat': 2, 'Ore': 1}
        """
        #resource_list = resource_str.split(',')
        resource_list = re.split(';|,|and', resource_str)
        resources = {}
        for item in resource_list:
            item = item.strip()
            if not item:
                continue
            try:
                qty_str, resource_name = item.split(' ', 1)
                qty = int(qty_str)
                resource_name = resource_name.strip()
                if resource_name not in self.resource_names or qty <= 0:
                    return None
                resources[resource_name] = qty
            except Exception:
                return None
        return resources

    def _execute_trade(self, trade, acceptor_id):
        """
        Execute the trade between players.
        Returns a tuple (success: bool, trade_result: str).
        trade_result can be 'proposer' or 'acceptor' indicating who lacks resources.
        """
        proposer_id = 1 - acceptor_id
        my_offer = trade['my_offer']  # Resources proposer gives
        their_offer = trade['their_offer']  # Resources acceptor gives

        # Check if proposer has enough resources
        for resource, qty in my_offer.items():
            if self.player_resources[proposer_id].get(resource, 0) < qty:
                return False, 'proposer'  # Proposer lacks resources

        # Check if acceptor has enough resources
        for resource, qty in their_offer.items():
            if self.player_resources[acceptor_id].get(resource, 0) < qty:
                return False, 'acceptor'  # Acceptor lacks resources

        # Execute the trade
        for resource, qty in my_offer.items():
            self.player_resources[proposer_id][resource] -= qty
            self.player_resources[acceptor_id][resource] += qty
        for resource, qty in their_offer.items():
            self.player_resources[acceptor_id][resource] -= qty
            self.player_resources[proposer_id][resource] += qty

        return True, 'success'

    def _calculate_rewards(self):
        """
        Calculate the rewards for both players based on their resource values.
        """
        gains = {}
        for player_id in [0, 1]:
            resources = self.player_resources[player_id]
            values = self.player_values[player_id]
            total_value = sum([qty * values[res] for res, qty in resources.items()])
            initial_resources = self.initial_resources[player_id]
            initial_value = sum([qty * values[res] for res, qty in initial_resources.items()])
            gains[player_id] = total_value - initial_value

        # Determine rewards
        #if self.trades_made is not None:
        if len(self.trades_made) == 0:
            # No trades made, both players lose
            rewards = {0: -1, 1: -1}
            reason = "No trades where made."
        else:
            reason = "higher trade gain."
            if gains[0] > gains[1]:
                rewards = {0: 1, 1: -1 if gains[1] < 0 else 0}
            elif gains[1] > gains[0]:
                rewards = {1: 1, 0: -1 if gains[0] < 0 else 0}
            else:
                rewards = {0: 0, 1: 0}
        return rewards, reason

    def _get_inventory_value(self, player_id):
        """
        For a given player ID, get the current inventory value and gain
        """
        resources = self.player_resources[player_id]
        values = self.player_values[player_id]
        total_value = sum([qty * values[res] for res, qty in resources.items()])
        initial_resources = self.initial_resources[player_id]
        initial_value = sum([qty * values[res] for res, qty in initial_resources.items()])
        gain = total_value - initial_value
        return total_value, gain


