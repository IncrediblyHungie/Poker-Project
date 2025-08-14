#!/usr/bin/env python3
"""
Play script for testing the trained Pluribus poker bot.
Allows bot vs bot testing and basic human vs bot play.
"""

import os
import sys
import argparse
import random
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator
from engine.game_state import GameState, Action
from abstraction.action_abstraction import ActionAbstraction
import yaml


class SimpleBot:
    """Simple bot that uses the trained blueprint strategy"""
    
    def __init__(self, blueprint_generator: BlueprintGenerator, player_id: int):
        self.blueprint_gen = blueprint_generator
        self.player_id = player_id
        self.cfr_solver = blueprint_generator.cfr_solver
        self.action_abstraction = blueprint_generator.action_abstraction
    
    def get_action(self, game_state: GameState):
        """Get action from bot using blueprint strategy"""
        if game_state.current_player != self.player_id:
            return None
        
        # Get strategy from blueprint
        infoset_key = self.cfr_solver.create_infoset_key(game_state, self.player_id)
        strategy = self.cfr_solver.get_strategy(infoset_key)
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, self.player_id
        )
        
        if not abstract_actions:
            return None
        
        # Sample action from strategy
        if strategy is not None and len(strategy) == len(abstract_actions):
            action_idx = random.choices(range(len(abstract_actions)), 
                                       weights=strategy)[0]
        else:
            # Fallback to random action
            action_idx = random.randint(0, len(abstract_actions) - 1)
        
        desc, action_type, amount = abstract_actions[action_idx]
        return desc, action_type, amount


def play_bot_vs_bot(blueprint_path: str, config_path: str, num_hands: int = 100):
    """Play bot vs bot games"""
    print(f"Playing {num_hands} hands of bot vs bot...")
    
    # Load blueprint
    blueprint_gen = BlueprintGenerator(config_path)
    
    if os.path.exists("data/card_abstractions.pkl"):
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    blueprint_gen.cfr_solver.load_strategy(blueprint_path)
    
    # Create bots
    bots = [SimpleBot(blueprint_gen, i) for i in range(6)]
    
    wins = [0] * 6
    total_utility = [0.0] * 6
    
    for hand_num in range(num_hands):
        if hand_num % 10 == 0:
            print(f"Hand {hand_num + 1}/{num_hands}")
        
        # Setup game
        game_state = blueprint_gen.setup_game(num_players=6)
        
        # Play hand
        actions_taken = 0
        max_actions = 1000  # Prevent infinite loops
        
        while not game_state.is_terminal() and actions_taken < max_actions:
            current_player = game_state.current_player
            bot = bots[current_player]
            
            # Get bot action
            bot_action = bot.get_action(game_state)
            if bot_action is None:
                break
            
            desc, action_type, amount = bot_action
            
            # Apply action
            success = game_state.apply_action(current_player, action_type, amount)
            if not success:
                print(f"Invalid action: {desc} by player {current_player}")
                break
            
            actions_taken += 1
        
        # Record results
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            
            for i, payoff in enumerate(payoffs):
                total_utility[i] += payoff
                if payoff > 0:
                    wins[i] += 1
    
    # Print results
    print("\nBot vs Bot Results:")
    print("=" * 30)
    for i in range(6):
        avg_utility = total_utility[i] / num_hands
        win_rate = wins[i] / num_hands * 100
        print(f"Player {i}: {wins[i]} wins ({win_rate:.1f}%), "
              f"Avg utility: {avg_utility:.2f}")


def play_human_vs_bot(blueprint_path: str, config_path: str):
    """Simple human vs bot interface"""
    print("Human vs Bot Poker")
    print("=" * 30)
    print("Actions: 'f' = fold, 'c' = check/call, 'r [amount]' = raise")
    print("Type 'quit' to exit\n")
    
    # Load blueprint
    blueprint_gen = BlueprintGenerator(config_path)
    
    if os.path.exists("data/card_abstractions.pkl"):
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    blueprint_gen.cfr_solver.load_strategy(blueprint_path)
    
    # Create bot (player 1)
    bot = SimpleBot(blueprint_gen, 1)
    human_player = 0
    
    while True:
        # Setup heads-up game
        game_state = blueprint_gen.setup_game(num_players=2)
        
        print(f"\nNew hand! Your cards: {game_state.players[human_player].hole_cards}")
        print(f"Community: {game_state.community_cards}")
        print(f"Pot: {game_state.pot}, Your stack: {game_state.players[human_player].stack}")
        
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            if current_player == human_player:
                # Human player
                legal_actions = game_state.get_legal_actions(current_player)
                print(f"\nLegal actions: {legal_actions}")
                
                action_input = input("Your action: ").strip().lower()
                
                if action_input == 'quit':
                    return
                
                # Parse human input
                if action_input == 'f':
                    action_type, amount = Action.FOLD, 0
                elif action_input == 'c':
                    # Check or call
                    call_amount = max(0, game_state.current_bet - 
                                    game_state.players[current_player].bet_this_round)
                    if call_amount == 0:
                        action_type, amount = Action.CHECK, 0
                    else:
                        action_type, amount = Action.CALL, call_amount
                elif action_input.startswith('r'):
                    # Raise
                    try:
                        parts = action_input.split()
                        if len(parts) > 1:
                            amount = int(parts[1])
                        else:
                            amount = game_state.big_blind * 2
                        action_type = Action.RAISE
                    except ValueError:
                        print("Invalid raise amount")
                        continue
                else:
                    print("Invalid action")
                    continue
                
                success = game_state.apply_action(current_player, action_type, amount)
                if not success:
                    print("Invalid action, try again")
                    continue
                
            else:
                # Bot player
                bot_action = bot.get_action(game_state)
                if bot_action is None:
                    break
                
                desc, action_type, amount = bot_action
                print(f"Bot action: {desc}")
                
                success = game_state.apply_action(current_player, action_type, amount)
                if not success:
                    break
        
        # Show results
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            if payoffs[human_player] > 0:
                print(f"You won ${payoffs[human_player]:.2f}!")
            else:
                print(f"You lost ${abs(payoffs[human_player]):.2f}")
        
        play_again = input("\nPlay another hand? (y/n): ").strip().lower()
        if play_again != 'y':
            break


def main():
    parser = argparse.ArgumentParser(description="Play with trained poker bot")
    parser.add_argument("--blueprint", default="data/blueprints/final_blueprint.pkl",
                       help="Path to trained blueprint")
    parser.add_argument("--config", default="config/game_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", choices=["bot_vs_bot", "human_vs_bot"], 
                       default="bot_vs_bot",
                       help="Game mode")
    parser.add_argument("--hands", type=int, default=100,
                       help="Number of hands for bot vs bot")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.blueprint):
        print(f"Blueprint not found: {args.blueprint}")
        print("Please train the bot first using scripts/train.py")
        return
    
    if args.mode == "bot_vs_bot":
        play_bot_vs_bot(args.blueprint, args.config, args.hands)
    elif args.mode == "human_vs_bot":
        play_human_vs_bot(args.blueprint, args.config)


if __name__ == "__main__":
    main()