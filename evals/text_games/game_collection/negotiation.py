from evals.text_games.game_collection.game_interface import GameInterface 
import random

class NegotiationGame(GameInterface):
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.reset()

    def reset(self):
        self.players = list(range(self.num_players))
        self.resources = {player_id: {'gold': 10, 'silver': 10, 'bronze': 10} for player_id in self.players}
        self.values = {'gold': 3, 'silver': 2, 'bronze': 1}
        self.private_info = {player_id: {'demand': random.choice(['gold', 'silver', 'bronze'])} for player_id in self.players}
        self.game_over = False
        self.turn = 0
        self.trade_proposals = []

    def get_state(self, player_id):
        state = {
            'your_resources': self.resources[player_id],
            'your_private_info': self.private_info[player_id],
            'public_trade_proposals': self.trade_proposals
        }
        return state

    def get_valid_actions(self, player_id):
        return ['[PUBLIC] message', '[PRIVATE] player_id message', 'propose_trade', 'accept_trade', 'decline_trade', 'end_turn']

    def step(self, player_id, action):
        if self.game_over:
            return None, 0, True, {'message': 'Game is over.'}

        reward = 0
        info = {}
        action_type, *args = action.split(' ', 1)
        if action_type == 'propose_trade':
            # Format: 'propose_trade to_player_id offer_resources for_resources'
            to_player_id, offer_for = args[0].split(' ', 1)
            to_player_id = int(to_player_id)
            offer, for_ = offer_for.split(' for ')
            self.trade_proposals.append({
                'from': player_id,
                'to': to_player_id,
                'offer': offer,
                'for': for_
            })
            info['message'] = 'Trade proposed.'
        elif action_type == 'accept_trade':
            # Accept the last trade proposal directed to the player
            proposal = next((p for p in reversed(self.trade_proposals) if p['to'] == player_id and not p.get('accepted')), None)
            if proposal:
                # Execute trade
                self.execute_trade(proposal)
                proposal['accepted'] = True
                info['message'] = 'Trade accepted.'
            else:
                info['error'] = 'No trade to accept.'
        elif action_type == 'decline_trade':
            # Decline the last trade proposal directed to the player
            proposal = next((p for p in reversed(self.trade_proposals) if p['to'] == player_id and not p.get('accepted')), None)
            if proposal:
                proposal['declined'] = True
                info['message'] = 'Trade declined.'
            else:
                info['error'] = 'No trade to decline.'
        elif action_type == 'end_turn':
            self.turn = (self.turn + 1) % self.num_players
            if self.turn == 0:
                self.game_over = True
        else:
            info['error'] = 'Invalid action.'
        state = self.get_state(player_id)
        done = self.game_over
        return state, reward, done, info

    def execute_trade(self, proposal):
        from_player = proposal['from']
        to_player = proposal['to']
        offer = proposal['offer'].split(',')
        for_ = proposal['for'].split(',')

        # Update resources (assuming valid resources)
        for item in offer:
            resource, amount = item.split(':')
            amount = int(amount)
            self.resources[from_player][resource] -= amount
            self.resources[to_player][resource] += amount

        for item in for_:
            resource, amount = item.split(':')
            amount = int(amount)
            self.resources[to_player][resource] -= amount
            self.resources[from_player][resource] += amount

    def is_over(self):
        return self.game_over

    def get_winner(self):
        # Calculate total value of resources for each player
        total_values = {}
        for player_id, resources in self.resources.items():
            total_values[player_id] = sum(self.values[res] * qty for res, qty in resources.items())
        # Player with highest total value wins
        winner = max(total_values, key=total_values.get)
        return winner
