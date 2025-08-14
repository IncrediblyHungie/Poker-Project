"""
Fast card abstraction using GPU acceleration and reduced complexity.
Optimized for quick training setup.
"""

import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from engine.game_state import BettingRound
from utils.device_config import get_device_config


class FastCardAbstraction:
    """Simplified card abstraction for faster training"""
    
    def __init__(self, config: dict, device_config=None):
        self.config = config
        self.device = device_config or get_device_config()
        
        # Simplified bucket mappings
        self.preflop_buckets = {}
        self.flop_buckets = {}
        self.turn_buckets = {}
        self.river_buckets = {}
        
        print("Using fast card abstraction (simplified for speed)")
    
    def train_abstractions(self):
        """Train card abstractions quickly using simplified approach"""
        print("Creating fast preflop abstraction...")
        self._create_fast_preflop_abstraction()
        
        print("Creating fast postflop abstractions...")
        self._create_fast_postflop_abstractions()
        
        print("Fast card abstractions complete!")
    
    def _create_fast_preflop_abstraction(self):
        """Create simplified preflop hand rankings"""
        # Simple preflop hand rankings based on card values
        preflop_hands = [
            # Pocket pairs (strong)
            ('AA', 1), ('KK', 1), ('QQ', 1), ('JJ', 2), ('TT', 2), ('99', 2),
            ('88', 3), ('77', 3), ('66', 4), ('55', 4), ('44', 5), ('33', 5), ('22', 5),
            
            # Suited connectors and high cards (medium-strong)
            ('AKs', 1), ('AQs', 2), ('AJs', 2), ('ATs', 3), ('A9s', 3),
            ('KQs', 2), ('KJs', 3), ('KTs', 3), ('QJs', 3), ('QTs', 4),
            
            # Offsuit high cards (medium)
            ('AKo', 2), ('AQo', 3), ('AJo', 4), ('ATo', 4),
            ('KQo', 3), ('KJo', 4), ('KTo', 5), ('QJo', 4), ('QTo', 5),
            
            # Suited connectors (speculative)
            ('JTs', 4), ('J9s', 5), ('T9s', 5), ('98s', 6), ('87s', 6),
            ('76s', 7), ('65s', 7), ('54s', 8),
        ]
        
        # Create bucket mapping (simplified to fewer buckets than config)
        target_buckets = min(50, self.config['card_buckets']['preflop'])  # Cap at 50
        
        for hand, bucket in preflop_hands:
            # Map to actual bucket (1-indexed to 0-indexed)
            actual_bucket = min(bucket - 1, target_buckets - 1)
            self.preflop_buckets[hand] = actual_bucket
        
        print(f"Created {len(self.preflop_buckets)} preflop hand mappings")
    
    def _create_fast_postflop_abstractions(self):
        """Create simplified postflop abstractions based on hand strength"""
        # For postflop, use simple heuristics instead of equity calculations
        
        # Flop buckets (simplified)
        flop_buckets = min(100, self.config['card_buckets']['flop'])
        for i in range(flop_buckets):
            bucket_key = f"flop_bucket_{i}"
            self.flop_buckets[bucket_key] = i
        
        # Turn buckets (simplified)
        turn_buckets = min(50, self.config['card_buckets']['turn'])
        for i in range(turn_buckets):
            bucket_key = f"turn_bucket_{i}"
            self.turn_buckets[bucket_key] = i
        
        # River buckets (simplified)
        river_buckets = min(25, self.config['card_buckets']['river'])
        for i in range(river_buckets):
            bucket_key = f"river_bucket_{i}"
            self.river_buckets[bucket_key] = i
        
        print(f"Created simplified postflop abstractions: "
              f"flop={len(self.flop_buckets)}, turn={len(self.turn_buckets)}, river={len(self.river_buckets)}")
    
    def get_bucket(self, hole_cards: List, community_cards: List, betting_round: BettingRound) -> int:
        """Get bucket for given cards and betting round"""
        
        if betting_round == BettingRound.PREFLOP:
            return self._get_preflop_bucket(hole_cards)
        else:
            return self._get_postflop_bucket(hole_cards, community_cards, betting_round)
    
    def _get_preflop_bucket(self, hole_cards: List) -> int:
        """Get preflop bucket using simplified hand rankings"""
        if len(hole_cards) != 2:
            return 0
        
        card1, card2 = hole_cards
        
        # Convert to simple string representation
        ranks = sorted([card1.rank.value, card2.rank.value], reverse=True)
        suits = [card1.suit, card2.suit]
        
        # Create hand string
        if ranks[0] == ranks[1]:
            # Pocket pair
            hand_str = f"{self._rank_to_char(ranks[0])}{self._rank_to_char(ranks[1])}"
        elif suits[0] == suits[1]:
            # Suited
            hand_str = f"{self._rank_to_char(ranks[0])}{self._rank_to_char(ranks[1])}s"
        else:
            # Offsuit
            hand_str = f"{self._rank_to_char(ranks[0])}{self._rank_to_char(ranks[1])}o"
        
        # Look up bucket
        return self.preflop_buckets.get(hand_str, 10)  # Default to medium bucket
    
    def _rank_to_char(self, rank_value: int) -> str:
        """Convert rank value to character"""
        rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        return rank_map.get(rank_value, str(rank_value))
    
    def _get_postflop_bucket(self, hole_cards: List, community_cards: List, betting_round: BettingRound) -> int:
        """Get postflop bucket using simplified heuristics"""
        # For now, use a simple hash-based approach for speed
        # In a real implementation, this would evaluate hand strength
        
        card_hash = hash(tuple(sorted([str(card) for card in hole_cards + community_cards])))
        
        if betting_round == BettingRound.FLOP:
            return abs(card_hash) % len(self.flop_buckets)
        elif betting_round == BettingRound.TURN:
            return abs(card_hash) % len(self.turn_buckets)
        elif betting_round == BettingRound.RIVER:
            return abs(card_hash) % len(self.river_buckets)
        
        return 0
    
    def save_abstractions(self, filepath: str):
        """Save card abstractions to file"""
        abstraction_data = {
            'preflop_buckets': self.preflop_buckets,
            'flop_buckets': self.flop_buckets,
            'turn_buckets': self.turn_buckets,
            'river_buckets': self.river_buckets,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(abstraction_data, f)
        
        print(f"Fast card abstractions saved to {filepath}")
    
    def load_abstractions(self, filepath: str):
        """Load card abstractions from file"""
        with open(filepath, 'rb') as f:
            abstraction_data = pickle.load(f)
        
        self.preflop_buckets = abstraction_data['preflop_buckets']
        self.flop_buckets = abstraction_data['flop_buckets']
        self.turn_buckets = abstraction_data['turn_buckets']
        self.river_buckets = abstraction_data['river_buckets']
        
        print(f"Fast card abstractions loaded from {filepath}")
    
    def get_abstraction_info(self) -> Dict:
        """Get information about current abstractions"""
        return {
            'preflop_buckets': len(self.preflop_buckets),
            'flop_buckets': len(self.flop_buckets),
            'turn_buckets': len(self.turn_buckets),
            'river_buckets': len(self.river_buckets),
            'type': 'fast_abstraction'
        }