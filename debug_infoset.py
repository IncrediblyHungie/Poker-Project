#!/usr/bin/env python3
"""
Debug information set creation specifically
"""

import sys
sys.path.insert(0, "src")

from engine.game_state import GameState
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
import yaml
import signal

print("Testing information set creation...")

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

print("✓ Game state created")

# Test infoset key creation
print("Testing infoset key creation...")
try:
    infoset_key = cfr_solver.create_infoset_key(game_state, 0)
    print(f"✓ Infoset key: {infoset_key}")
except Exception as e:
    print(f"❌ Infoset key creation failed: {e}")
    sys.exit(1)

# Test infoset creation
print("Testing infoset creation...")
actions = action_abs.get_abstract_actions(game_state, 0)
num_actions = len(actions)
print(f"Number of actions: {num_actions}")

try:
    infoset = cfr_solver.get_infoset(infoset_key, num_actions)
    print(f"✓ Infoset created: {infoset.key}")
except Exception as e:
    print(f"❌ Infoset creation failed: {e}")
    sys.exit(1)

# Test strategy calculation
print("Testing strategy calculation...")
try:
    strategy = infoset.get_strategy(1.0)
    print(f"✓ Strategy: {strategy}")
except Exception as e:
    print(f"❌ Strategy calculation failed: {e}")
    sys.exit(1)

# Test the exact CFR traversal call that's hanging
print("Testing exact CFR traversal call...")

def timeout_handler(signum, frame):
    raise TimeoutError("CFR traversal timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10-second timeout

try:
    reach_probs = [1.0, 1.0]
    traversing_player = 0
    print("Calling _mccfr_traverse...")
    utilities = cfr_solver._mccfr_traverse(game_state, reach_probs, traversing_player)
    signal.alarm(0)
    print(f"✓ CFR traversal completed: {utilities}")
except TimeoutError:
    print("❌ CFR traversal timed out - infinite loop confirmed")
    signal.alarm(0)
except Exception as e:
    print(f"❌ CFR traversal failed: {e}")
    signal.alarm(0)

print("Infoset debug complete")