import random, re
from itertools import combinations
from evals.text_games.game_collection.game_interface import GameInterface

class PokerGame(GameInterface):
    RANKS = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
        '7': 7, '8': 8, '9': 9, '10': 10,
        'jack': 11, 'queen': 12, 'king': 13, 'ace': 14
    }
    def __init__(self, render=False):
        """
        Initialize the Simplified Poker game.
        Args:
            render (bool): If True, the game will display relevant info in the terminal.
        """
        self.name = "Poker"
        self.render = render 

    def reset(self):
        """Reset the game to its initial state."""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.player_hands = {0: [], 1: []}
        self.bets = {0: 0, 1: 0}
        self.pot = 0
        self.turn = 0
        self.max_turns = 20  # Increased max_turns to accommodate fewer community reveals
        self.game_over = False
        self.folded = None  # Player who folded, if any

        # Track the last player who made a bet to manage turn-based betting
        self.last_bet_player = None

        # Deal two cards to each player
        for player_id in [0, 1]:
            self.player_hands[player_id] = [self.deck.pop(), self.deck.pop()]

        self.player_chips = {0:100, 1:100}

        # Prepare initial prompts
        player_prompts = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1)
        }
        return player_prompts

    def _create_deck(self):
        """Create a standard 52-card deck."""
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = [str(n) for n in range(2, 11)] + ['Jack', 'Queen', 'King', 'Ace']
        deck = [f"{rank} of {suit}" for suit in suits for rank in ranks]
        return deck

    def _generate_player_prompt(self, player_id):
        hand = self.player_hands[player_id]
        prompt = (
            f"You are Player {player_id}. You have been dealt the following hand:\n"
            f"{hand[0]}, {hand[1]}.\n"
            "In each round, you can 'Bet', 'Call', 'Raise', or 'Fold'.\n"
            "You can also chat with the other player to bluff or communicate.\n"
            "Example actions:\n"
            "- '[bet] 10'\n"
            "- '[call]'\n"
            "- '[raise] 20'\n"
            "- '[fold]'\n"
            #"You can also just talk to your opponent. "
            "You start with 100 chips."
        )
        return prompt

    def get_valid_actions(self, player_id):
        """
        Return valid actions for the given player based on the current game state.
        - If the player needs to respond to a bet, restrict actions to 'call', 'raise', or 'fold'.
        - Otherwise, allow all actions: 'bet', 'call', 'raise', 'fold'.
        """
        other_player_id = 1 - player_id
        if self.bets[other_player_id] > self.bets[player_id]:
            # Player needs to respond to a bet
            return ['[call]', '[raise]', '[fold]']
        else:
            []

    def get_info(self):
        """Return additional information."""
        return {
            'num_turns': self.turn,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'bets': self.bets
        }

    def step(self, player_id, action):
        """
        Process the player's action.
        """
        if self.render:
            self._render(player_id, action)  # Render current action and state


        # Reveal a community card every 4 turns
        if self.turn > 0 and self.turn % 4 == 0 and len(self.community_cards) < 5:
            community_card = self.deck.pop()
            self.community_cards.append(community_card)
        state = "Community Cards: " + \
            ", ".join(self.community_cards) + \
            f"\n[Player {player_id}]: {action}"


        self.turn += 1  # Increment turn after processing action

        # Check for game over conditions
        if self.turn >= self.max_turns:
            # Evaluate hands and determine winner
            winner_id = self._determine_winner()
            if self.pot == 0:
                # If nothing was bet, the winning player gets -1 and the losing player 0
                reward = {
                    player_id: -1 if player_id==winner_id else 0,
                    1-player_id: -1 if player_id!=winner_id else 0 
                }
                return None, reward, True, {"reason": "No bets were made. Player with winning hand gets -1, other 0."}
            else:
                reward = {winner_id: 1, 1-winner_id: -1}
                return None, {winner_id: 1, 1-winner_id: -1}, True, {"reason": f"Player {winner_id} wins the pot."}

        # for ease of processing of the action
        action_lower = action.strip().lower()
        if "[bet]" in action_lower:
            amount = re.search(r'\[bet\]\s*(\d+)', action_lower)
            if not amount:
                # a number must be provided
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} made a bet without number."}
            amount = int(amount.group(1))
            if amount <= 0 or amount > self.player_chips[player_id]:
                 return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried making a negative/too high bet."}
            self.bets[player_id] += amount 
            self.pot += amount 
            self.player_chips[player_id] -= amount
            self.last_bet_player = player_id 
            return state, None, False, {"info": f"Player {player_id} bets {amount}"}

        elif "[call]" in action_lower:
            amount = self.bets[1-player_id] - self.bets[player_id]
            if amount <= 0 or amount > self.player_chips[player_id]:
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried calling a negative number or more than they have."}
            self.bets[player_id] += amount 
            self.pot += amount 
            self.player_chips[player_id] -= amount
            return state, None, False, {"info": f"Player {player_id} called."}

        elif "[raise]" in action_lower:
            amount = re.search(r'\[raise\]\s*(\d+)', action_lower)
            if not amount:
                # a number must be provided
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried raising without number."}
            raise_total = (self.bets[1-player_id]-self.bets[player_id]) + int(amount.group(1)) 
            if raise_total <= 0 or amount > self.player_chips[player_id]:
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} tried raise a negative/too-high amount."}
            self.bets[player_id] += raise_total 
            self.pot += raise_total 
            self.last_bet_player = player_id 
            self.player_chips[player_id] -= raise_total
            return state, None, False, {"info": f"Player {player_id} raises {amount}"}

        elif "[fold]" in action_lower:
            # determine hypothetical winner
            winner_id = self._determine_winner()
            reward = {
                player_id: -1 if winner_id==player_id else 0,
                1-player_id: 0
            }
            # if player would have won, -1, else 0
            return None, reward, True, {"reason": f"Player {player_id} folds. Player {winner_id} would have won the game."}

        else:
            # chat message
            return state, None, False, {"info": f"Chat message by Player {player_id}: {action}"}





    def _render(self, player_id, action):
        """
        Print the current state of the game and the action taken.
        """
        print(f"[Player {player_id}] Action: {action}")
        print(f"Pot: {self.pot} | Bets: {self.bets}")
        print(f"Community Cards: {', '.join(self.community_cards) if self.community_cards else 'None'}")
        print(f"Player {player_id} Hand: {self.player_hands[player_id]}")
        print(f"Player {1 - player_id} Hand: {self.player_hands[1 - player_id]} (Hidden)\n")

    def _determine_winner(self):
        """
        Determine the winner based on hand strength.
        Returns:
            winner_id (int): ID of the winning player (0 or 1). Returns None in case of a tie.
        """
        # if not all community cards have been revealed, do so
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())

        player_best_hands = {}
        for player_id in [0, 1]:
            all_cards = self.player_hands[player_id] + self.community_cards
            best_hand = self._get_best_hand(all_cards)
            player_best_hands[player_id] = best_hand
        # Compare the best hands
        if player_best_hands[0] > player_best_hands[1]:
            return 0
        elif player_best_hands[1] > player_best_hands[0]:
            return 1
        else:
            return None  # It's a tie

    def _get_best_hand(self, cards):
        """
        From the given cards, determine the best possible 5-card hand.
        Args:
            cards (list): List of card strings.
        Returns:
            best_hand (tuple): A tuple representing the best hand's rank and high cards.
        """
        best_hand = None
        for combo in combinations(cards, 5):
            current_hand = self._evaluate_hand(combo)
            if not best_hand or current_hand > best_hand:
                best_hand = current_hand
        return best_hand

    def _evaluate_hand(self, hand):
        """
        Evaluate a 5-card poker hand and return its rank.
        Args:
            hand (list): List of 5 card strings.
        Returns:
            hand_rank (tuple): A tuple representing the hand's rank and relevant high cards.
        """
        ranks = []
        suits = []
        for card in hand:
            rank_str, suit = card.lower().split(' of ')
            ranks.append(self.RANKS[rank_str])
            suits.append(suit)

        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), reverse=True)

        is_flush = len(set(suits)) == 1
        is_straight = False
        high_card = max(ranks)

        sorted_ranks = sorted(ranks, reverse=True)
        # Check for straight (including Ace-low straight)
        if sorted_ranks == list(range(high_card, high_card - 5, -1)):
            is_straight = True
        elif sorted_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            high_card = 5  # In Ace-low straight, the high card is 5

        if is_straight and is_flush:
            if high_card == 14:
                return (10,)  # Royal Flush
            return (9, high_card)  # Straight Flush
        elif counts[0] == 4:
            four_kind = max(rank for rank, count in rank_counts.items() if count == 4)
            kicker = max(rank for rank in ranks if rank != four_kind)
            return (8, four_kind, kicker)  # Four of a Kind
        elif counts[0] == 3 and counts[1] == 2:
            three_kind = max(rank for rank, count in rank_counts.items() if count == 3)
            pair = max(rank for rank, count in rank_counts.items() if count == 2)
            return (7, three_kind, pair)  # Full House
        elif is_flush:
            return (6, sorted_ranks)  # Flush
        elif is_straight:
            return (5, high_card)  # Straight
        elif counts[0] == 3:
            three_kind = max(rank for rank, count in rank_counts.items() if count == 3)
            kickers = sorted([rank for rank in ranks if rank != three_kind], reverse=True)
            return (4, three_kind, kickers)  # Three of a Kind
        elif counts[0] == 2 and counts[1] == 2:
            high_pair = max(rank for rank, count in rank_counts.items() if count == 2)
            low_pair = min(rank for rank, count in rank_counts.items() if count == 2)
            kicker = max(rank for rank in ranks if rank != high_pair and rank != low_pair)
            return (3, high_pair, low_pair, kicker)  # Two Pair
        elif counts[0] == 2:
            pair = max(rank for rank, count in rank_counts.items() if count == 2)
            kickers = sorted([rank for rank in ranks if rank != pair], reverse=True)
            return (2, pair, kickers)  # One Pair
        else:
            return (1, sorted_ranks)  # High Card