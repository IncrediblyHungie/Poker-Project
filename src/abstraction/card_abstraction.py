"""
Card abstraction system using K-means clustering and Earth Mover's Distance.
Groups similar hands into buckets based on equity distribution calculations.
"""

import numpy as np
import itertools
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
import pickle
import os
from ..engine.hand_evaluator import HandEvaluator, Card, Rank, Suit
from ..engine.game_state import BettingRound


class CardAbstraction:
    def __init__(self, config: dict):
        self.config = config
        self.hand_evaluator = HandEvaluator()
        
        # Bucket mappings for each betting round
        self.preflop_buckets = {}
        self.flop_buckets = {}
        self.turn_buckets = {}
        self.river_buckets = {}
        
        # Equity distributions cache
        self.equity_cache = {}
        
        # Initialize preflop hands (all 169 canonical hands)
        self.preflop_hands = self._generate_preflop_hands()
    
    def _generate_preflop_hands(self) -> List[Tuple[Card, Card]]:
        """Generate all 169 canonical preflop hands"""
        hands = []
        
        # Generate all unique two-card combinations
        ranks = list(Rank)
        
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i <= j:  # Avoid duplicates
                    if i == j:
                        # Pocket pair - use spades/hearts
                        hand = (Card(rank1, Suit.SPADES), Card(rank2, Suit.HEARTS))
                    else:
                        # Suited - use spades
                        suited_hand = (Card(rank1, Suit.SPADES), Card(rank2, Suit.SPADES))
                        # Offsuit - use spades/hearts
                        offsuit_hand = (Card(rank1, Suit.SPADES), Card(rank2, Suit.HEARTS))
                        hands.extend([suited_hand, offsuit_hand])
                        continue
                    hands.append(hand)
        
        return hands
    
    def compute_equity_distribution(self, hole_cards: List[Card], 
                                  board: List[Card],
                                  num_opponents: int = 1,
                                  num_simulations: int = 1000) -> np.ndarray:
        """
        Compute equity distribution using Monte Carlo sampling.
        Returns histogram of win probabilities.
        """
        cache_key = (tuple(hole_cards), tuple(board), num_opponents)
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]
        
        # Create remaining deck
        deck = self.hand_evaluator._create_deck()
        used_cards = set(hole_cards + board)
        remaining_deck = [card for card in deck if card not in used_cards]
        
        win_probabilities = []
        
        for _ in range(num_simulations):
            # Shuffle deck
            np.random.shuffle(remaining_deck)
            
            # Complete board if necessary
            sim_board = board.copy()
            deck_idx = 0
            
            while len(sim_board) < 5:
                sim_board.append(remaining_deck[deck_idx])
                deck_idx += 1
            
            # Deal opponent hands
            opponent_hands = []
            for _ in range(num_opponents):
                opp_hole = [remaining_deck[deck_idx], remaining_deck[deck_idx + 1]]
                opponent_hands.append(opp_hole)
                deck_idx += 2
            
            # Calculate equity for this simulation
            our_hand = self.hand_evaluator.evaluate_hand(hole_cards + sim_board)
            wins = 0
            ties = 0
            
            for opp_hole in opponent_hands:
                opp_hand = self.hand_evaluator.evaluate_hand(opp_hole + sim_board)
                comparison = self.hand_evaluator.compare_hands(our_hand, opp_hand)
                
                if comparison > 0:
                    wins += 1
                elif comparison == 0:
                    ties += 1
            
            equity = (wins + ties * 0.5) / num_opponents
            win_probabilities.append(equity)
        
        # Convert to histogram (distribution)
        hist, _ = np.histogram(win_probabilities, bins=20, range=(0, 1))
        distribution = hist / np.sum(hist)  # Normalize
        
        self.equity_cache[cache_key] = distribution
        return distribution
    
    def earth_movers_distance(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Calculate Earth Mover's Distance (Wasserstein distance) between distributions.
        """
        # Create position arrays for the distributions
        positions = np.arange(len(dist1))
        return wasserstein_distance(positions, positions, dist1, dist2)
    
    def create_clustering(self, betting_round: BettingRound, 
                         num_buckets: int,
                         sample_boards: List[List[Card]] = None) -> Dict:
        """
        Create card abstraction clustering for a specific betting round.
        """
        if betting_round == BettingRound.PREFLOP:
            return self._create_preflop_clustering()
        
        # For post-flop rounds, we need sample boards
        if sample_boards is None:
            sample_boards = self._generate_sample_boards(betting_round, num_samples=1000)
        
        # Collect all hand-board combinations
        hand_board_combos = []
        equity_distributions = []
        
        # Sample representative hole card combinations
        hole_card_samples = self._sample_hole_cards(num_samples=500)
        
        for hole_cards in hole_card_samples:
            for board in sample_boards:
                # Skip if hole cards overlap with board
                if any(card in board for card in hole_cards):
                    continue
                
                hand_board_combos.append((hole_cards, board))
                
                # Compute equity distribution
                equity_dist = self.compute_equity_distribution(
                    hole_cards, board, num_opponents=2, num_simulations=200
                )
                equity_distributions.append(equity_dist)
        
        # Convert to numpy array for clustering
        X = np.array(equity_distributions)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_buckets, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create mapping from hand-board combinations to buckets
        clustering = {}
        for i, (hole_cards, board) in enumerate(hand_board_combos):
            key = self._create_hand_board_key(hole_cards, board)
            clustering[key] = cluster_labels[i]
        
        return clustering
    
    def _create_preflop_clustering(self) -> Dict:
        """Create preflop clustering (use all 169 hands without abstraction)"""
        clustering = {}
        for i, hand in enumerate(self.preflop_hands):
            key = self._canonicalize_preflop_hand(hand)
            clustering[key] = i  # Each hand gets its own bucket
        
        return clustering
    
    def _canonicalize_preflop_hand(self, hand: Tuple[Card, Card]) -> str:
        """Convert preflop hand to canonical string representation"""
        card1, card2 = hand
        rank1, rank2 = card1.rank.value, card2.rank.value
        
        # Sort by rank (higher first)
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            card1, card2 = card2, card1
        
        # Determine if suited
        suited = card1.suit == card2.suit
        
        # Create canonical representation
        if rank1 == rank2:
            return f"{rank1}{rank1}"  # Pocket pair
        else:
            suffix = "s" if suited else "o"
            return f"{rank1}{rank2}{suffix}"
    
    def _sample_hole_cards(self, num_samples: int) -> List[List[Card]]:
        """Sample representative hole card combinations"""
        deck = self.hand_evaluator._create_deck()
        samples = []
        
        for _ in range(num_samples):
            np.random.shuffle(deck)
            hole_cards = deck[:2]
            samples.append(hole_cards)
        
        return samples
    
    def _generate_sample_boards(self, betting_round: BettingRound, 
                               num_samples: int) -> List[List[Card]]:
        """Generate sample boards for a betting round"""
        deck = self.hand_evaluator._create_deck()
        boards = []
        
        # Determine board size based on betting round
        if betting_round == BettingRound.FLOP:
            board_size = 3
        elif betting_round == BettingRound.TURN:
            board_size = 4
        elif betting_round == BettingRound.RIVER:
            board_size = 5
        else:
            return []
        
        for _ in range(num_samples):
            np.random.shuffle(deck)
            board = deck[:board_size]
            boards.append(board)
        
        return boards
    
    def _create_hand_board_key(self, hole_cards: List[Card], 
                              board: List[Card]) -> str:
        """Create unique key for hand-board combination"""
        # Convert cards to sortable representation
        all_cards = hole_cards + board
        card_strs = [f"{card.rank.value}{card.suit.value}" for card in all_cards]
        
        # Sort for consistency
        hole_strs = sorted(card_strs[:2])
        board_strs = sorted(card_strs[2:])
        
        return f"{''.join(hole_strs)}|{''.join(board_strs)}"
    
    def get_bucket(self, hole_cards: List[Card], board: List[Card], 
                   betting_round: BettingRound) -> int:
        """Get bucket number for given hand and board"""
        if betting_round == BettingRound.PREFLOP:
            key = self._canonicalize_preflop_hand((hole_cards[0], hole_cards[1]))
            return self.preflop_buckets.get(key, 0)
        
        # Post-flop rounds
        key = self._create_hand_board_key(hole_cards, board)
        
        if betting_round == BettingRound.FLOP:
            return self.flop_buckets.get(key, 0)
        elif betting_round == BettingRound.TURN:
            return self.turn_buckets.get(key, 0)
        elif betting_round == BettingRound.RIVER:
            return self.river_buckets.get(key, 0)
        
        return 0
    
    def train_abstractions(self):
        """Train all card abstractions"""
        print("Training card abstractions...")
        
        # Preflop (all 169 hands)
        print("Creating preflop abstraction...")
        self.preflop_buckets = self._create_preflop_clustering()
        
        # Flop
        print("Creating flop abstraction...")
        self.flop_buckets = self.create_clustering(
            BettingRound.FLOP, 
            self.config['card_buckets']['flop']
        )
        
        # Turn
        print("Creating turn abstraction...")
        self.turn_buckets = self.create_clustering(
            BettingRound.TURN,
            self.config['card_buckets']['turn']
        )
        
        # River
        print("Creating river abstraction...")
        self.river_buckets = self.create_clustering(
            BettingRound.RIVER,
            self.config['card_buckets']['river']
        )
        
        print("Card abstraction training complete!")
    
    def save_abstractions(self, filepath: str):
        """Save trained abstractions to disk"""
        abstractions = {
            'preflop_buckets': self.preflop_buckets,
            'flop_buckets': self.flop_buckets,
            'turn_buckets': self.turn_buckets,
            'river_buckets': self.river_buckets,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(abstractions, f)
    
    def load_abstractions(self, filepath: str):
        """Load trained abstractions from disk"""
        with open(filepath, 'rb') as f:
            abstractions = pickle.load(f)
        
        self.preflop_buckets = abstractions['preflop_buckets']
        self.flop_buckets = abstractions['flop_buckets']
        self.turn_buckets = abstractions['turn_buckets']
        self.river_buckets = abstractions['river_buckets']
        self.config = abstractions['config']
    
    def get_num_buckets(self, betting_round: BettingRound) -> int:
        """Get number of buckets for a betting round"""
        if betting_round == BettingRound.PREFLOP:
            return self.config['card_buckets']['preflop']
        elif betting_round == BettingRound.FLOP:
            return self.config['card_buckets']['flop']
        elif betting_round == BettingRound.TURN:
            return self.config['card_buckets']['turn']
        elif betting_round == BettingRound.RIVER:
            return self.config['card_buckets']['river']
        
        return 1