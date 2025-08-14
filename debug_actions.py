#!/usr/bin/env python3
"""
Debug action abstraction specifically
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
import yaml

print("Testing action abstraction...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

# Create a simple game state
print("Creating game state...")
game_state = GameState(
    num_players=2,
    starting_stack=1000,
    small_blind=10,
    big_blind=20
)

# Deal simple hole cards
from engine.hand_evaluator import Card, Rank, Suit
game_state.players[0].hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
game_state.players[1].hole_cards = [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.JACK, Suit.CLUBS)]

print(f"Game state terminal: {game_state.is_terminal()}")
print(f"Current player: {game_state.current_player}")

# Test action generation
for i in range(10):  # Test several rounds
    print(f"\n--- Round {i+1} ---")
    print(f"Is terminal: {game_state.is_terminal()}")
    print(f"Current player: {game_state.current_player}")
    
    if game_state.is_terminal():
        print("Game is terminal, stopping")
        break
    
    actions = action_abs.get_abstract_actions(game_state, game_state.current_player)
    print(f"Available actions: {actions}")
    
    if not actions:
        print("❌ No actions available but game not terminal - this is the bug!")
        break
    
    # Apply first action to advance game
    desc, action_type, amount = actions[0]
    print(f"Applying action: {desc} ({action_type}, {amount})")
    success = game_state.apply_action(game_state.current_player, action_type, amount)
    print(f"Action success: {success}")
    
    if not success:
        print("Action failed, stopping")
        break

print("Action debug complete")