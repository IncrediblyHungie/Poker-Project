#!/usr/bin/env python3
"""
Debug CFR specifically
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
import yaml
import random

print("Testing CFR...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

# Create CFR solver
print("Creating CFR solver...")
cfr_solver = LinearCFR(card_abs, action_abs)
print("✓ CFR solver created")

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

print("✓ Game state with cards created")

# Test one CFR iteration with timeout
print("Testing CFR iteration (with 30-second timeout)...")
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("CFR iteration timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30-second timeout

try:
    utilities = cfr_solver.train_iteration(game_state)
    signal.alarm(0)  # Cancel timeout
    print(f"✓ CFR iteration completed! Utilities: {utilities}")
    print("✓ CFR is working correctly")
except TimeoutError:
    print("❌ CFR iteration timed out - there's an infinite loop in CFR traversal")
except Exception as e:
    print(f"❌ CFR iteration failed: {e}")
    signal.alarm(0)

print("CFR debug complete")