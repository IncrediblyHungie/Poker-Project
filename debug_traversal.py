#!/usr/bin/env python3
"""
Debug CFR traversal step by step
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
import yaml

print("Testing CFR traversal step by step...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

# Create CFR solver
cfr_solver = LinearCFR(card_abs, action_abs)

# Create a simple game state
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

print("✓ Initial game state created")

def debug_traverse(game_state, depth=0, max_depth=5):
    """Manual traversal with debugging"""
    print(f"{'  ' * depth}Depth {depth}: Terminal={game_state.is_terminal()}, Current={game_state.current_player}")
    
    if depth >= max_depth:
        print(f"{'  ' * depth}Stopping at max depth")
        return
    
    if game_state.is_terminal():
        payoffs = game_state.get_payoffs()
        print(f"{'  ' * depth}Terminal payoffs: {payoffs}")
        return
    
    current_player = game_state.current_player
    actions = action_abs.get_abstract_actions(game_state, current_player)
    print(f"{'  ' * depth}Actions for player {current_player}: {len(actions)}")
    
    if not actions:
        print(f"{'  ' * depth}❌ No actions but not terminal!")
        return
    
    # Test first action only to avoid explosion
    desc, action_type, amount = actions[0]
    print(f"{'  ' * depth}Testing action: {desc}")
    
    # Test copy mechanism
    new_state = game_state.copy()
    success = new_state.apply_action(current_player, action_type, amount)
    print(f"{'  ' * depth}Action success: {success}")
    
    if success:
        # Check if state actually changed
        same_terminal = new_state.is_terminal() == game_state.is_terminal()
        same_player = new_state.current_player == game_state.current_player
        print(f"{'  ' * depth}State changed: terminal={not same_terminal}, player={not same_player}")
        
        # Recurse
        debug_traverse(new_state, depth + 1, max_depth)

print("Starting manual traversal...")
debug_traverse(game_state)
print("Manual traversal complete")