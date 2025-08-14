#!/usr/bin/env python3
"""
Minimal CFR implementation to isolate the infinite loop
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
import yaml
import signal
import numpy as np

print("Testing minimal CFR implementation...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

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

print("✓ Game state created")

def minimal_cfr_traverse(game_state, depth=0):
    """Minimal CFR traversal with no strategy calculation"""
    print(f"Depth {depth}: terminal={game_state.is_terminal()}, current={game_state.current_player}")
    
    if depth > 3:
        print(f"Max depth reached at {depth}")
        return {0: 0.0, 1: 0.0}
    
    if game_state.is_terminal():
        payoffs = game_state.get_payoffs()
        print(f"Terminal payoffs: {payoffs}")
        return {0: payoffs[0], 1: payoffs[1]}
    
    current_player = game_state.current_player
    actions = action_abs.get_abstract_actions(game_state, current_player)
    
    print(f"Player {current_player} has {len(actions)} actions")
    
    if not actions:
        print("No actions available")
        return {0: 0.0, 1: 0.0}
    
    # Just test the first action to avoid loops
    desc, action_type, amount = actions[0]
    print(f"Testing action: {desc}")
    
    new_state = game_state.copy()
    success = new_state.apply_action(current_player, action_type, amount)
    
    if not success:
        print("Action failed")
        return {0: 0.0, 1: 0.0}
    
    # Recurse with just this one action
    return minimal_cfr_traverse(new_state, depth + 1)

def timeout_handler(signum, frame):
    raise TimeoutError("Minimal CFR timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10-second timeout

try:
    print("Starting minimal CFR traversal...")
    result = minimal_cfr_traverse(game_state)
    signal.alarm(0)
    print(f"✓ Minimal CFR completed: {result}")
except TimeoutError:
    print("❌ Minimal CFR timed out")
    signal.alarm(0)
except Exception as e:
    print(f"❌ Minimal CFR failed: {e}")
    signal.alarm(0)

print("Minimal CFR debug complete")