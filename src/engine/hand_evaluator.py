"""
Hand evaluator for 7-card Texas Hold'em poker.
Uses lookup tables for fast hand evaluation and comparison.
"""

from enum import Enum
from typing import List, Tuple, Optional
import itertools


class Suit(Enum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank.name[0]}{self.suit.name[0]}"
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))


class HandEvaluator:
    def __init__(self):
        self._lookup_table = self._build_lookup_table()
    
    def _build_lookup_table(self) -> dict:
        """Build lookup table for hand rankings (simplified version)"""
        # In a production implementation, this would be a comprehensive
        # lookup table with all possible hand combinations pre-computed
        return {}
    
    def evaluate_hand(self, cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a 7-card hand and return the best 5-card hand.
        Returns tuple of (hand_rank, kickers) where kickers are in descending order.
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate hand")
        
        # Generate all possible 5-card combinations
        best_hand = None
        best_rank = HandRank.HIGH_CARD
        best_kickers = []
        
        for combo in itertools.combinations(cards, 5):
            hand_rank, kickers = self._evaluate_5_cards(list(combo))
            
            if (hand_rank.value > best_rank.value or 
                (hand_rank.value == best_rank.value and kickers > best_kickers)):
                best_hand = combo
                best_rank = hand_rank
                best_kickers = kickers
        
        return best_rank, best_kickers
    
    def _evaluate_5_cards(self, cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate exactly 5 cards"""
        if len(cards) != 5:
            raise ValueError("Must evaluate exactly 5 cards")
        
        ranks = sorted([card.rank.value for card in cards], reverse=True)
        suits = [card.suit for card in cards]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight = self._is_straight(ranks)
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Sort ranks by count then by rank value
        sorted_counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        counts = [count for rank, count in sorted_counts]
        sorted_ranks = [rank for rank, count in sorted_counts]
        
        # Determine hand rank
        if is_straight and is_flush:
            if ranks == [14, 13, 12, 11, 10]:  # Royal flush
                return HandRank.ROYAL_FLUSH, [14]
            else:
                return HandRank.STRAIGHT_FLUSH, [ranks[0]]
        
        elif counts == [4, 1]:
            return HandRank.FOUR_OF_A_KIND, [sorted_ranks[0], sorted_ranks[1]]
        
        elif counts == [3, 2]:
            return HandRank.FULL_HOUSE, [sorted_ranks[0], sorted_ranks[1]]
        
        elif is_flush:
            return HandRank.FLUSH, ranks
        
        elif is_straight:
            return HandRank.STRAIGHT, [ranks[0]]
        
        elif counts == [3, 1, 1]:
            return HandRank.THREE_OF_A_KIND, [sorted_ranks[0]] + sorted_ranks[1:]
        
        elif counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, [max(sorted_ranks[0], sorted_ranks[1]), 
                                       min(sorted_ranks[0], sorted_ranks[1]), sorted_ranks[2]]
        
        elif counts == [2, 1, 1, 1]:
            return HandRank.PAIR, [sorted_ranks[0]] + sorted_ranks[1:]
        
        else:
            return HandRank.HIGH_CARD, ranks
    
    def _is_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks))
        
        if len(sorted_ranks) != 5:
            return False
        
        # Check for A-2-3-4-5 straight (wheel)
        if sorted_ranks == [2, 3, 4, 5, 14]:
            return True
        
        # Check for regular straight
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] != sorted_ranks[i-1] + 1:
                return False
        
        return True
    
    def compare_hands(self, hand1: Tuple[HandRank, List[int]], 
                     hand2: Tuple[HandRank, List[int]]) -> int:
        """
        Compare two hands.
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        rank1, kickers1 = hand1
        rank2, kickers2 = hand2
        
        if rank1.value > rank2.value:
            return 1
        elif rank1.value < rank2.value:
            return -1
        else:
            # Same hand rank, compare kickers
            for k1, k2 in zip(kickers1, kickers2):
                if k1 > k2:
                    return 1
                elif k1 < k2:
                    return -1
            return 0
    
    def get_hand_strength(self, hole_cards: List[Card], 
                         community_cards: List[Card]) -> float:
        """
        Calculate hand strength as a value between 0 and 1.
        Used for abstraction and evaluation.
        """
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 5:
            # Pre-flop or incomplete community cards
            return self._preflop_strength(hole_cards)
        
        hand_rank, kickers = self.evaluate_hand(all_cards)
        
        # Convert to normalized strength (simplified)
        # In production, this would use lookup tables with more precise values
        base_strength = hand_rank.value / 10.0
        
        # Add kicker strength
        kicker_strength = sum(k / 14.0 for k in kickers[:3]) / 3.0 * 0.1
        
        return min(base_strength + kicker_strength, 1.0)
    
    def _preflop_strength(self, hole_cards: List[Card]) -> float:
        """Calculate preflop hand strength"""
        if len(hole_cards) != 2:
            return 0.0
        
        card1, card2 = hole_cards
        rank1, rank2 = card1.rank.value, card2.rank.value
        
        # Pocket pairs
        if rank1 == rank2:
            return 0.5 + (rank1 / 14.0) * 0.4
        
        # Suited cards
        is_suited = card1.suit == card2.suit
        
        # High cards
        high_card = max(rank1, rank2)
        low_card = min(rank1, rank2)
        
        strength = (high_card / 14.0) * 0.6 + (low_card / 14.0) * 0.2
        
        if is_suited:
            strength += 0.1
        
        # Connected cards bonus
        if abs(rank1 - rank2) == 1:
            strength += 0.05
        
        return min(strength, 0.95)
    
    def calculate_equity(self, hole_cards: List[Card], 
                        community_cards: List[Card],
                        num_opponents: int,
                        num_simulations: int = 1000) -> float:
        """
        Calculate hand equity using Monte Carlo simulation.
        Returns win probability against random opponent hands.
        """
        if num_simulations <= 0:
            return 0.0
        
        # Create deck without known cards
        deck = self._create_deck()
        known_cards = set(hole_cards + community_cards)
        remaining_deck = [card for card in deck if card not in known_cards]
        
        wins = 0
        ties = 0
        
        for _ in range(num_simulations):
            # Shuffle remaining deck
            import random
            random.shuffle(remaining_deck)
            
            # Complete community cards if needed
            sim_community = community_cards.copy()
            deck_idx = 0
            
            while len(sim_community) < 5:
                sim_community.append(remaining_deck[deck_idx])
                deck_idx += 1
            
            # Deal opponent hands
            opponent_hands = []
            for _ in range(num_opponents):
                opponent_hole = [remaining_deck[deck_idx], remaining_deck[deck_idx + 1]]
                opponent_hands.append(opponent_hole)
                deck_idx += 2
            
            # Evaluate all hands
            our_hand = self.evaluate_hand(hole_cards + sim_community)
            opponent_results = [self.evaluate_hand(opp_hole + sim_community) 
                              for opp_hole in opponent_hands]
            
            # Compare against all opponents
            beats_all = True
            ties_with_any = False
            
            for opp_hand in opponent_results:
                comparison = self.compare_hands(our_hand, opp_hand)
                if comparison < 0:
                    beats_all = False
                    break
                elif comparison == 0:
                    ties_with_any = True
            
            if beats_all:
                if ties_with_any:
                    ties += 1
                else:
                    wins += 1
        
        return (wins + ties * 0.5) / num_simulations
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck"""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        return deck


def create_card(rank_str: str, suit_str: str) -> Card:
    """Helper function to create a card from strings"""
    rank_map = {
        '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
        '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
        'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE
    }
    
    suit_map = {
        'S': Suit.SPADES, 'H': Suit.HEARTS, 'D': Suit.DIAMONDS, 'C': Suit.CLUBS
    }
    
    return Card(rank_map[rank_str.upper()], suit_map[suit_str.upper()])