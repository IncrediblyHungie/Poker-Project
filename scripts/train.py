#!/usr/bin/env python3
"""
Training script for Pluribus poker bot.
Trains card abstractions and blueprint strategy.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator
from abstraction.card_abstraction import CardAbstraction
import yaml


def train_abstractions(config_path: str):
    """Train card abstractions"""
    print("=" * 50)
    print("Training Card Abstractions")
    print("=" * 50)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize card abstraction
    card_abstraction = CardAbstraction(config['abstraction'])
    
    # Train abstractions
    card_abstraction.train_abstractions()
    
    # Save abstractions
    abstraction_path = "data/card_abstractions.pkl"
    card_abstraction.save_abstractions(abstraction_path)
    print(f"Card abstractions saved to {abstraction_path}")


def train_blueprint(config_path: str, iterations: int = None, 
                   load_abstractions: bool = True):
    """Train blueprint strategy"""
    print("=" * 50)
    print("Training Blueprint Strategy")
    print("=" * 50)
    
    # Initialize blueprint generator
    blueprint_gen = BlueprintGenerator(config_path)
    
    # Load pre-trained abstractions if available
    if load_abstractions and os.path.exists("data/card_abstractions.pkl"):
        print("Loading pre-trained card abstractions...")
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    # Train blueprint
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if iterations is None:
        iterations = config['training']['iterations']
    
    stats = blueprint_gen.train_blueprint(iterations)
    
    # Save final blueprint
    blueprint_path = "data/blueprints/final_blueprint.pkl"
    blueprint_gen.save_blueprint(blueprint_path)
    
    # Evaluate blueprint
    evaluation = blueprint_gen.evaluate_blueprint()
    
    return stats, evaluation


def main():
    parser = argparse.ArgumentParser(description="Train Pluribus poker bot")
    parser.add_argument("--config", default="config/game_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Number of training iterations")
    parser.add_argument("--skip-abstractions", action="store_true",
                       help="Skip card abstraction training")
    parser.add_argument("--abstractions-only", action="store_true",
                       help="Only train abstractions, skip blueprint")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("data/blueprints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Pluribus Poker Bot Training")
    print("=" * 50)
    
    # Train card abstractions
    if not args.skip_abstractions:
        train_abstractions(args.config)
    
    # Train blueprint strategy
    if not args.abstractions_only:
        stats, evaluation = train_blueprint(
            args.config, 
            args.iterations,
            load_abstractions=not args.skip_abstractions
        )
        
        print("\nTraining Summary:")
        print(f"Final exploitability: {stats['exploitability'][-1]:.6f}")
        print(f"Total information sets: {stats['total_infosets'][-1]}")
        print(f"Average utility: {evaluation['average_utility']:.2f}")
        print(f"Hands evaluated: {evaluation['hands_played']}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()