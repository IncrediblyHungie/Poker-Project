#!/usr/bin/env python3
"""
Debug CFR with CPU-only mode to rule out GPU issues
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
from utils.device_config import setup_device
import yaml
import signal

print("Testing CFR with CPU-only mode...")

# Load config
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Force CPU-only device config
device_config = setup_device(force_cpu=True)
print(f"Device config: CPU-only mode, backend: {device_config.backend}")

# Create abstractions
card_abs = FastCardAbstraction(config['abstraction'], device_config=device_config)
card_abs.train_abstractions()
action_abs = ActionAbstraction(config['abstraction']['action_fractions'])

# Create CFR solver with CPU-only mode
cfr_solver = LinearCFR(card_abs, action_abs, device_config=device_config)

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
    raise TimeoutError("CPU-only CFR timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10-second timeout

try:
    print("Testing CPU-only CFR iteration...")
    utilities = cfr_solver.train_iteration(game_state)
    signal.alarm(0)
    print(f"✓ CPU-only CFR completed: {utilities}")
except TimeoutError:
    print("❌ CPU-only CFR timed out")
    signal.alarm(0)
except Exception as e:
    print(f"❌ CPU-only CFR failed: {e}")
    signal.alarm(0)

print("CPU-only CFR debug complete")