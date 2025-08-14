#!/usr/bin/env python3
"""
Test the safe SimpleCFR implementation
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.simple_cfr import SimpleCFR
import yaml
import signal

print("Testing SimpleCFR implementation...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

# Create SimpleCFR solver
cfr_solver = SimpleCFR(card_abs, action_abs)

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

def timeout_handler(signum, frame):
    raise TimeoutError("SimpleCFR timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10-second timeout

try:
    print("Testing SimpleCFR iteration...")
    utilities = cfr_solver.train_iteration(game_state)
    signal.alarm(0)
    print(f"✓ SimpleCFR completed: {utilities}")
    
    # Test multiple iterations
    print("Testing 10 iterations...")
    for i in range(10):
        utilities = cfr_solver.train_iteration(game_state)
        print(f"Iteration {i+1}: {utilities}")
    
    print("✓ SimpleCFR works correctly!")
    
except TimeoutError:
    print("❌ SimpleCFR timed out")
    signal.alarm(0)
except Exception as e:
    print(f"❌ SimpleCFR failed: {e}")
    signal.alarm(0)

print("SimpleCFR debug complete")