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
from abstraction.fast_card_abstraction import FastCardAbstraction
import yaml


def train_abstractions(config_path: str, use_gpu: bool = True, device_id: int = 0, use_fast: bool = True):
    """Train card abstractions"""
    print("=" * 50)
    print("Training Card Abstractions")
    print("=" * 50)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize card abstraction with GPU support
    from utils.device_config import setup_device
    device_config = setup_device(force_cpu=not use_gpu, device_id=device_id)
    
    if use_fast:
        print("Using fast card abstraction for quick training...")
        card_abstraction = FastCardAbstraction(config['abstraction'], device_config=device_config)
    else:
        print("Using full card abstraction (slower but more accurate)...")
        card_abstraction = CardAbstraction(config['abstraction'], device_config=device_config)
    
    # Train abstractions
    card_abstraction.train_abstractions()
    
    # Save abstractions
    abstraction_path = "data/card_abstractions.pkl"
    card_abstraction.save_abstractions(abstraction_path)
    print(f"Card abstractions saved to {abstraction_path}")


def train_blueprint(config_path: str, iterations: int = None, 
                   load_abstractions: bool = True, use_gpu: bool = True, device_id: int = 0,
                   batch_size: int = None, disable_batching: bool = False):
    """Train blueprint strategy"""
    print("=" * 50)
    print("Training Blueprint Strategy")
    print("=" * 50)
    
    # Initialize blueprint generator with GPU support
    blueprint_gen = BlueprintGenerator(
        config_path, 
        use_gpu=use_gpu, 
        device_id=device_id,
        use_batch_processing=not disable_batching,
        batch_size=batch_size
    )
    
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
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU-only training (disable GPU)")
    parser.add_argument("--device-id", type=int, default=0,
                       help="GPU device ID to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for GPU training (auto-detected by default)")
    parser.add_argument("--disable-batching", action="store_true",
                       help="Disable batch processing (use single iterations)")
    parser.add_argument("--slow-abstractions", action="store_true",
                       help="Use full card abstractions (slower but more accurate)")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("data/blueprints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Pluribus Poker Bot Training")
    print("=" * 50)
    
    # GPU/CPU configuration
    use_gpu = not args.cpu
    device_id = args.device_id
    
    # Train card abstractions
    if not args.skip_abstractions:
        train_abstractions(args.config, use_gpu=use_gpu, device_id=device_id, use_fast=not args.slow_abstractions)
    
    # Train blueprint strategy
    if not args.abstractions_only:
        stats, evaluation = train_blueprint(
            args.config, 
            args.iterations,
            load_abstractions=not args.skip_abstractions,
            use_gpu=use_gpu,
            device_id=device_id,
            batch_size=args.batch_size,
            disable_batching=args.disable_batching
        )
        
        print("\nTraining Summary:")
        print(f"Final exploitability: {stats['exploitability'][-1]:.6f}")
        print(f"Total information sets: {stats['total_infosets'][-1]}")
        print(f"Average utility: {evaluation['average_utility']:.2f}")
        print(f"Hands evaluated: {evaluation['hands_played']}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()