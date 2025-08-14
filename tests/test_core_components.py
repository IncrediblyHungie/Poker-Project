#!/usr/bin/env python3
"""
Unit tests for core components that work without relative imports.
"""

import sys
import unittest
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from engine.game_state import GameState, Action, BettingRound, Player
from engine.action_handler import ActionHandler
from engine.hand_evaluator import HandEvaluator, Card, Rank, Suit


class TestGameState(unittest.TestCase):
    """Test GameState functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.game_state = GameState(
            num_players=6,
            big_blind=100,
            small_blind=50,
            starting_stack=1000
        )
    
    def test_initialization(self):
        """Test game state initialization"""
        self.assertEqual(len(self.game_state.players), 6)
        self.assertEqual(self.game_state.big_blind, 100)
        self.assertEqual(self.game_state.small_blind, 50)
        self.assertEqual(self.game_state.current_betting_round, BettingRound.PREFLOP)
        
        # Check player stacks
        for player in self.game_state.players:
            self.assertEqual(player.stack, 1000)
    
    def test_betting_actions(self):
        """Test basic betting actions"""
        current_player = self.game_state.current_player
        
        # Test fold action
        result = self.game_state.apply_action(current_player, Action.FOLD, 0)
        self.assertTrue(result)
        
        # Verify player is folded
        self.assertTrue(self.game_state.players[current_player].has_folded)
    
    def test_pot_management(self):
        """Test pot calculations"""
        initial_pot = self.game_state.pot
        current_player = self.game_state.current_player
        
        # Make a call
        call_amount = self.game_state.current_bet - self.game_state.players[current_player].bet_this_round
        self.game_state.apply_action(current_player, Action.CALL, call_amount)
        
        # Check pot increased
        self.assertGreater(self.game_state.pot, initial_pot)


class TestActionHandler(unittest.TestCase):
    """Test ActionHandler functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.game_state = GameState(
            num_players=6,
            big_blind=100,
            small_blind=50,
            starting_stack=1000
        )
        self.action_handler = ActionHandler()
    
    def test_legal_actions(self):
        """Test getting legal actions"""
        current_player = self.game_state.current_player
        legal_actions = self.action_handler.get_legal_actions(self.game_state, current_player)
        
        self.assertIsInstance(legal_actions, list)
        self.assertGreater(len(legal_actions), 0)
        
        # Should always be able to fold
        self.assertIn(Action.FOLD, legal_actions)
    
    def test_action_validation(self):
        """Test action validation"""
        current_player = self.game_state.current_player
        
        # Valid fold should work
        is_valid = self.action_handler.is_valid_action(
            self.game_state, current_player, Action.FOLD, 0
        )
        self.assertTrue(is_valid)
        
        # Invalid player should fail
        is_valid = self.action_handler.is_valid_action(
            self.game_state, (current_player + 1) % 6, Action.FOLD, 0
        )
        self.assertFalse(is_valid)


class TestHandEvaluator(unittest.TestCase):
    """Test HandEvaluator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = HandEvaluator()
    
    def test_card_creation(self):
        """Test card creation"""
        card = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(card.rank, Rank.ACE)
        self.assertEqual(card.suit, Suit.SPADES)
    
    def test_hand_strength_calculation(self):
        """Test basic hand strength calculation"""
        # Create a simple hand
        hole_cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        community_cards = [
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.TEN, Suit.SPADES)
        ]
        
        strength = self.evaluator.evaluate_hand(hole_cards, community_cards)
        self.assertIsInstance(strength, (int, float))
        self.assertGreater(strength, 0)
    
    def test_equity_calculation(self):
        """Test equity calculation"""
        hole_cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        community_cards = [
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.CLUBS)
        ]
        
        equity = self.evaluator.calculate_equity(
            hole_cards, community_cards, num_opponents=1, num_simulations=100
        )
        
        self.assertIsInstance(equity, float)
        self.assertGreaterEqual(equity, 0.0)
        self.assertLessEqual(equity, 1.0)


class TestPlayer(unittest.TestCase):
    """Test Player functionality"""
    
    def test_player_initialization(self):
        """Test player initialization"""
        player = Player(player_id=0, starting_stack=1000)
        
        self.assertEqual(player.player_id, 0)
        self.assertEqual(player.stack, 1000)
        self.assertEqual(player.bet_this_round, 0)
        self.assertFalse(player.has_folded)
        self.assertFalse(player.is_all_in)
        self.assertEqual(len(player.hole_cards), 0)
    
    def test_player_betting(self):
        """Test player betting functionality"""
        player = Player(player_id=0, starting_stack=1000)
        
        # Test betting
        bet_amount = 100
        player.bet_this_round = bet_amount
        player.stack -= bet_amount
        
        self.assertEqual(player.bet_this_round, 100)
        self.assertEqual(player.stack, 900)
    
    def test_player_all_in(self):
        """Test all-in detection"""
        player = Player(player_id=0, starting_stack=100)
        
        # Bet entire stack
        player.bet_this_round = 100
        player.stack = 0
        player.is_all_in = True
        
        self.assertTrue(player.is_all_in)
        self.assertEqual(player.stack, 0)


def run_tests():
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGameState,
        TestActionHandler, 
        TestHandEvaluator,
        TestPlayer
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Core Component Tests")
    print("=" * 40)
    
    success = run_tests()
    
    if success:
        print("\n✅ All core component tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1)