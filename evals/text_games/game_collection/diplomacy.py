from evals.text_games.game_collection.game_interface import GameInterface 
import random

class DiplomacyGame(GameInterface):
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.reset()

    def reset(self):
        self.players = list(range(self.num_players))
        self.game_over = False
        self.turn = 0
        self.map_state = {}  # Simplified map state
        self.orders = {player_id: None for player_id in self.players}
        self.phase = 'diplomacy'  # Phases: 'diplomacy', 'orders', 'resolution'
        self.messages = []

    def get_state(self, player_id):
        state = {
            'map_state': self.map_state,
            'phase': self.phase,
            'your_orders': self.orders[player_id],
            'messages': self.messages
        }
        return state

    def get_valid_actions(self, player_id):
        if self.phase == 'diplomacy':
            return ['[PUBLIC] message', '[PRIVATE] player_id message', 'submit_orders']
        elif self.phase == 'orders':
            return ['order unit_id action']
        else:
            return []

    def step(self, player_id, action):
        if self.game_over:
            return None, 0, True, {'message': 'Game is over.'}

        reward = 0
        info = {}

        if self.phase == 'diplomacy':
            if action.startswith('[PUBLIC]'):
                message = action[len('[PUBLIC]'):].strip()
                self.messages.append({'from': player_id, 'to': 'all', 'message': message})
            elif action.startswith('[PRIVATE]'):
                rest = action[len('[PRIVATE]'):].strip()
                to_player_id_str, message = rest.split(' ', 1)
                to_player_id = int(to_player_id_str)
                self.messages.append({'from': player_id, 'to': to_player_id, 'message': message})
            elif action == 'submit_orders':
                self.phase = 'orders'
        elif self.phase == 'orders':
            # Collect orders
            self.orders[player_id] = action
            if all(self.orders.values()):
                self.phase = 'resolution'
                self.resolve_orders()
        state = self.get_state(player_id)
        done = self.game_over
        return state, reward, done, info

    def resolve_orders(self):
        # Simplified resolution logic
        # For demonstration purposes, we'll randomly determine outcomes
        self.map_state = {'territory_control': {player_id: random.randint(1, 5) for player_id in self.players}}
        self.game_over = True

    def is_over(self):
        return self.game_over

    def get_winner(self):
        if self.game_over:
            # Player with most territories wins
            territories = self.map_state['territory_control']
            winner = max(territories, key=territories.get)
            return winner
        else:
            return None
