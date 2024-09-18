import random
from evals.text_games.game_collection.game_interface import GameInterface

class NegotiationGame(GameInterface):
    def __init__(self, resource_types=5, max_turns=10):
        """
        Initialize the Negotiation Game.
        Args:
            resource_types (int): Number of different resource types.
            max_turns (int): Maximum number of turns for the game.
        """
        self.name = "Negotiation"
        self.resource_types = resource_types
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Reset the game to its initial state."""
        self.turn = 0  # Current turn number
        self.current_player = 0  # Player 0 starts
        self.game_over = False

        # Generate random allocations of resources for each player
        self.player_resources = {
            0: [random.randint(1, 5) for _ in range(self.resource_types)],
            1: [random.randint(1, 5) for _ in range(self.resource_types)]
        }

        # Generate player-specific values for each resource type (hidden from other player)
        self.player_values = {
            0: [random.randint(5, 15) for _ in range(self.resource_types)],
            1: [random.randint(5, 15) for _ in range(self.resource_types)]
        }

        # Keep track of initial resources for value calculation
        self.initial_resources = {
            0: self.player_resources[0][:],
            1: self.player_resources[1][:]
        }

        # Keep track of trades made
        self.trades_made = []

        # Return the main prompts for both players
        player_prompts = {
            0: (
                f"You have resources: {self.player_resources[0]}. "
                "Each resource has a specific value to you (hidden from the other player). "
                "Negotiate trades with the other player using structured offers. "
                "Format: 'Offer: I give [your resources], You give [their resources]'. "
                "You can also respond with 'Accept' or 'Deny' to an offer."
            ),
            1: (
                f"You have resources: {self.player_resources[1]}. "
                "Each resource has a specific value to you (hidden from the other player). "
                "Negotiate trades with the other player using structured offers. "
                "Format: 'Offer: I give [your resources], You give [their resources]'. "
                "You can also respond with 'Accept' or 'Deny' to an offer."
            )
        }
        return player_prompts

    def get_valid_actions(self, player_id):
        """Return valid actions for the given player."""
        return None  # No restrictions; actions will be validated in `step`

    def get_info(self):
        """Return additional information."""
        return {'turn': self.turn}

    def step(self, player_id, action):
        """
        Process the player's action.
        Args:
            player_id (int): The player's ID.
            action (str): The player's action.
        Returns:
            state (dict): The new state after the action.
            reward (dict): The rewards for each player.
            done (bool): Whether the game is over.
            info (dict): Additional information.
        """
        # Other player ID
        other_player_id = 1 - player_id

        # Process the action
        # Expected action formats:
        # - "Offer: I give [list], You give [list]"
        # - "Accept"
        # - "Deny"

        action_lower = action.lower().strip()
        response = {}
        done = False
        reward = None

        if action_lower.startswith("offer:"):
            # Player is making an offer
            parsed_offer = self._parse_offer(action)
            if parsed_offer:
                self.current_offer = {
                    'from_player': player_id,
                    'to_player': other_player_id,
                    'offer': parsed_offer
                }
                response[other_player_id] = f"Player {player_id} offers a trade: {action}"
                info = {"info": "Offer made."}
            else:
                response[player_id] = "Invalid offer format. Please follow the structure."
                info = {"info": "Invalid offer format."}
        elif action_lower == "accept":
            if hasattr(self, 'current_offer') and self.current_offer['to_player'] == player_id:
                # Trade is accepted
                self._execute_trade(self.current_offer['offer'], player_id)
                self.trades_made.append(self.current_offer)
                del self.current_offer
                response = {
                    player_id: "You accepted the trade.",
                    other_player_id: f"Player {player_id} accepted your trade."
                }
                info = {"info": "Trade accepted."}
            else:
                response[player_id] = "No valid offer to accept."
                info = {"info": "No valid offer to accept."}
        elif action_lower == "deny":
            if hasattr(self, 'current_offer') and self.current_offer['to_player'] == player_id:
                # Trade is denied
                del self.current_offer
                response = {
                    player_id: "You denied the trade.",
                    other_player_id: f"Player {player_id} denied your trade."
                }
                info = {"info": "Trade denied."}
            else:
                response[player_id] = "No valid offer to deny."
                info = {"info": "No valid offer to deny."}
        else:
            # General conversation
            response[other_player_id] = f"Player {player_id}: {action}"
            info = {"info": "Message sent."}

        # Increment turn and check for game over
        self.turn += 1
        if self.turn >= self.max_turns:
            done = True
            reward = self._calculate_rewards()
            info = {"info": "Maximum turns reached. Game over."}

        return response, reward, done, info

    def _parse_offer(self, action):
        """
        Parse the offer action to extract the resources being traded.
        Expected format: "Offer: I give [list], You give [list]"
        Returns a dictionary with 'offer' and 'request' resource lists.
        """
        try:
            parts = action.split("Offer:")[1].strip()
            offer_parts = parts.split(", You give ")
            my_offer_str = offer_parts[0].split("I give ")[1].strip()
            their_offer_str = offer_parts[1].strip()

            my_offer = [int(x) for x in my_offer_str.strip('[]').split(',')]
            their_offer = [int(x) for x in their_offer_str.strip('[]').split(',')]

            if len(my_offer) != self.resource_types or len(their_offer) != self.resource_types:
                return None

            return {'my_offer': my_offer, 'their_offer': their_offer}
        except Exception:
            return None

    def _execute_trade(self, trade, acceptor_id):
        """
        Execute the trade between players.
        """
        proposer_id = 1 - acceptor_id
        my_offer = trade['my_offer']
        their_offer = trade['their_offer']

        # Update the resources
        for i in range(self.resource_types):
            # Check if players have enough resources to trade
            if self.player_resources[proposer_id][i] < my_offer[i] or self.player_resources[acceptor_id][i] < their_offer[i]:
                continue  # Skip if they don't have enough resources

            # Execute the trade
            self.player_resources[proposer_id][i] -= my_offer[i]
            self.player_resources[acceptor_id][i] -= their_offer[i]
            self.player_resources[proposer_id][i] += their_offer[i]
            self.player_resources[acceptor_id][i] += my_offer[i]

    def _calculate_rewards(self):
        """
        Calculate the rewards for both players based on their resource values.
        """
        total_values = {}
        gains = {}
        for player_id in [0, 1]:
            resources = self.player_resources[player_id]
            values = self.player_values[player_id]
            total_value = sum([r * v for r, v in zip(resources, values)])
            initial_resources = self.initial_resources[player_id]
            initial_value = sum([r * v for r, v in zip(initial_resources, values)])
            total_values[player_id] = total_value
            gains[player_id] = total_value - initial_value

        # Determine rewards
        if not self.trades_made:
            # No trades made, both players lose
            rewards = {0: -1, 1: -1}
        else:
            if gains[0] > gains[1]:
                rewards = {0: 1, 1: -1 if gains[1] < 0 else 0}
            elif gains[1] > gains[0]:
                rewards = {1: 1, 0: -1 if gains[0] < 0 else 0}
            else:
                rewards = {0: 0, 1: 0}
        return rewards
