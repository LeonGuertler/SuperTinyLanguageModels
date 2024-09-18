# game_collection/game2.py

from evals.text_games.game_collection.game_interface import GameInterface
import random

class WerewolfGame(GameInterface):
    def __init__(self, num_players=4, num_rounds=1):
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.reset()

    def reset(self):
        self.players = list(range(self.num_players))
        self.roles = {}
        self.alive = {player_id: True for player_id in self.players}
        self.game_over = False
        self.round = 0
        self.assign_roles()
        self.turn_order = self.players.copy()
        self.current_speaker = 0
        self.votes = {}
        self.discussion_rounds = 2  # Number of discussion rounds per game

    def assign_roles(self):
        # Randomly assign one werewolf
        werewolf = random.choice(self.players)
        for player_id in self.players:
            self.roles[player_id] = 'werewolf' if player_id == werewolf else 'villager'

    def get_state(self, player_id):
        role = self.roles[player_id]
        if self.alive[player_id]:
            return f"You are a {role}. Alive players: {', '.join(str(p) for p in self.alive_players())}."
        else:
            return "You are dead."

    def get_valid_actions(self, player_id):
        if not self.alive[player_id]:
            return []
        if self.phase == 'discussion':
            return None  # Free-form discussion
        elif self.phase == 'voting':
            return [str(p) for p in self.alive_players() if p != player_id]
        else:
            return []

    def step(self, player_id, action):
        if self.game_over:
            return None, 0, True, {'message': 'Game is over.'}

        reward = 0
        info = {}

        if self.phase == 'discussion':
            # Players can say anything during discussion
            self.current_speaker = (self.current_speaker + 1) % len(self.alive_players())
            if self.current_speaker == 0:
                self.discussion_rounds -= 1
                if self.discussion_rounds == 0:
                    self.phase = 'voting'
        elif self.phase == 'voting':
            # Record votes
            try:
                vote = int(action)
                if vote in self.alive_players() and vote != player_id:
                    self.votes[player_id] = vote
                else:
                    info['error'] = 'Invalid vote.'
            except ValueError:
                info['error'] = 'Invalid vote format.'
            if len(self.votes) == len(self.alive_players()):
                # All votes are in
                self.execute_votes()
                self.round += 1
                if self.check_game_over():
                    self.game_over = True
                else:
                    self.phase = 'discussion'
                    self.discussion_rounds = 2
                    self.votes = {}
                    self.current_speaker = 0
        state = self.get_state(player_id)
        done = self.game_over
        return state, reward, done, info

    def alive_players(self):
        return [player_id for player_id, alive in self.alive.items() if alive]

    def execute_votes(self):
        vote_counts = {}
        for vote in self.votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        # Eliminate the player with the most votes
        if vote_counts:
            eliminated_player = max(vote_counts, key=vote_counts.get)
            self.alive[eliminated_player] = False

    def check_game_over(self):
        werewolves = [p for p in self.alive_players() if self.roles[p] == 'werewolf']
        villagers = [p for p in self.alive_players() if self.roles[p] == 'villager']
        if not werewolves:
            self.winner = 'villagers'
            return True
        if len(werewolves) >= len(villagers):
            self.winner = 'werewolves'
            return True
        return False

    def is_over(self):
        return self.game_over

    def get_winner(self):
        return self.winner if self.game_over else None

    @property
    def phase(self):
        return 'voting' if self.discussion_rounds == 0 else 'discussion'
