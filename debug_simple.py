#!/usr/bin/env python3
"""
Simple debug test to isolate the issue
"""

import sys
sys.path.insert(0, "src")

# Test basic imports
print("Testing imports...")
from engine.game_state import GameState, Action
from abstraction.fast_card_abstraction import FastCardAbstraction
from abstraction.action_abstraction import ActionAbstraction
import yaml

print("✓ Imports successful")

# Test config loading
with open('config/game_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("✓ Config loaded")

# Test basic game state
print("Testing game state creation...")
game_state = GameState(
    num_players=2,
    starting_stack=1000,
    small_blind=10,
    big_blind=20
)
print(f"✓ Game state created with {game_state.num_players} players")

# Test if game terminates properly
print("Testing game termination...")
print(f"Is terminal: {game_state.is_terminal()}")
print(f"Current player: {game_state.current_player}")

# Test abstractions
print("Testing abstractions...")
card_abs = FastCardAbstraction(config['abstraction'])
card_abs.train_abstractions()
print("✓ Card abstraction created")

action_abs = ActionAbstraction(config['abstraction']['action_fractions'])
print("✓ Action abstraction created")

# Test getting actions
print("Testing action generation...")
actions = action_abs.get_abstract_actions(game_state, 0)
print(f"Actions for player 0: {actions}")

# Test one action application
if actions:
    desc, action_type, amount = actions[0]
    print(f"Applying action: {desc}")
    success = game_state.apply_action(0, action_type, amount)
    print(f"Action success: {success}")
    print(f"Game terminal after action: {game_state.is_terminal()}")

print("✓ Basic functionality test complete")