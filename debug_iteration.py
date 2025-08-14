#!/usr/bin/env python3
"""
Debug the train_iteration method specifically
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
import yaml
import signal

print("Testing train_iteration method...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create abstractions
print("Creating card abstraction...")
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
print("✓ Card abstraction complete")

print("Creating action abstraction...")
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])
print("✓ Action abstraction complete")

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

print("✓ Game state created")

# Test each part of train_iteration separately
print("Testing random player selection...")
import random
traversing_player = random.randint(0, game_state.num_players - 1)
print(f"✓ Traversing player: {traversing_player}")

print("Testing reach probabilities...")
reach_probs = [1.0] * game_state.num_players
print(f"✓ Reach probs: {reach_probs}")

print("Testing train_iteration method...")

def timeout_handler(signum, frame):
    raise TimeoutError("train_iteration timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5-second timeout

try:
    print("Calling train_iteration...")
    utilities = cfr_solver.train_iteration(game_state)
    signal.alarm(0)
    print(f"✓ train_iteration completed: {utilities}")
except TimeoutError:
    print("❌ train_iteration timed out")
    signal.alarm(0)
except Exception as e:
    print(f"❌ train_iteration failed: {e}")
    signal.alarm(0)

print("train_iteration debug complete")