#!/usr/bin/env python3
"""
Evaluation script for analyzing poker bot strategy quality over training.
Measures exploitability, head-to-head performance, and convergence.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator
from evaluation.strategy_evaluator import StrategyEvaluator
from evaluation.metrics import PokerMetrics, EvaluationVisualizer
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction


def evaluate_checkpoints(config_path: str, checkpoint_dir: str, 
                        output_dir: str, use_gpu: bool = True):
    """Evaluate all checkpoints in a directory"""
    print("Poker Strategy Quality Evaluation")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    print("Initializing evaluation components...")
    card_abstraction = CardAbstraction(config['abstraction'])
    action_abstraction = ActionAbstraction(config['abstraction']['action_fractions'])
    
    # Load card abstractions if available
    if os.path.exists("data/card_abstractions.pkl"):
        print("Loading card abstractions...")
        card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(card_abstraction, action_abstraction)
    
    # Run evaluation
    print(f"Evaluating checkpoints in {checkpoint_dir}...")
    results = evaluator.evaluate_strategy_progression(checkpoint_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.pkl")
    evaluator.save_evaluation_results(results, results_path)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_path = os.path.join(output_dir, "training_progress.png")
    try:
        EvaluationVisualizer.plot_training_progress(results, plot_path)
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
        print("Install matplotlib to generate plots: pip install matplotlib")
    
    # Generate report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    EvaluationVisualizer.create_evaluation_report(results, report_path)
    
    # Print summary
    print_evaluation_summary(results)
    
    return results


def evaluate_single_strategy(blueprint_path: str, config_path: str, 
                           output_dir: str, use_gpu: bool = True):
    """Evaluate a single trained strategy"""
    print("Single Strategy Evaluation")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize blueprint generator
    blueprint_gen = BlueprintGenerator(config_path, use_gpu=use_gpu)
    
    # Load strategy
    if os.path.exists("data/card_abstractions.pkl"):
        blueprint_gen.card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    blueprint_gen.cfr_solver.load_strategy(blueprint_path)
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(
        blueprint_gen.card_abstraction, 
        blueprint_gen.action_abstraction
    )
    
    # Run evaluations
    print("Computing exploitability...")
    exploitability = evaluator.compute_exploitability(
        blueprint_gen.cfr_solver, num_samples=2000
    )
    
    print("Analyzing strategy properties...")
    entropy = PokerMetrics.strategy_entropy(blueprint_gen.cfr_solver)
    action_analysis = PokerMetrics.action_frequency_analysis(blueprint_gen.cfr_solver)
    
    # Compile results
    results = {
        'strategy_path': blueprint_path,
        'exploitability': exploitability,
        'strategy_entropy': entropy,
        'action_frequency': action_analysis,
        'total_infosets': len(blueprint_gen.cfr_solver.infosets),
        'training_iterations': blueprint_gen.cfr_solver.iterations,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, "single_strategy_evaluation.pkl")
    evaluator.save_evaluation_results(results, results_path)
    
    # Generate report
    report_path = os.path.join(output_dir, "single_strategy_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print_single_strategy_summary(results)
    
    return results


def compare_strategies(blueprint_paths: list, config_path: str, 
                      output_dir: str, use_gpu: bool = True):
    """Compare multiple strategies head-to-head"""
    print("Strategy Comparison Evaluation")
    print("=" * 50)
    
    if len(blueprint_paths) < 2:
        print("Error: Need at least 2 strategies to compare")
        return None
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    card_abstraction = CardAbstraction(config['abstraction'])
    action_abstraction = ActionAbstraction(config['abstraction']['action_fractions'])
    
    if os.path.exists("data/card_abstractions.pkl"):
        card_abstraction.load_abstractions("data/card_abstractions.pkl")
    
    evaluator = StrategyEvaluator(card_abstraction, action_abstraction)
    
    # Load all strategies
    strategies = []
    for i, path in enumerate(blueprint_paths):
        print(f"Loading strategy {i+1}: {path}")
        strategy = evaluator._load_strategy_from_checkpoint(path)
        if strategy:
            strategies.append((path, strategy))
        else:
            print(f"Warning: Could not load strategy from {path}")
    
    if len(strategies) < 2:
        print("Error: Could not load enough strategies for comparison")
        return None
    
    # Run pairwise comparisons
    comparison_results = {}
    
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            path1, strategy1 = strategies[i]
            path2, strategy2 = strategies[j]
            
            print(f"Comparing {os.path.basename(path1)} vs {os.path.basename(path2)}...")
            
            # Head-to-head evaluation
            h2h_result = evaluator.head_to_head_evaluation(
                strategy1, strategy2, num_hands=1000
            )
            
            # Nash distance
            nash_dist = PokerMetrics.nash_distance(strategy1, strategy2)
            
            comparison_key = f"{os.path.basename(path1)}_vs_{os.path.basename(path2)}"
            comparison_results[comparison_key] = {
                'strategy1_path': path1,
                'strategy2_path': path2,
                'head_to_head': h2h_result,
                'nash_distance': nash_dist
            }
    
    # Compile final results
    results = {
        'strategies_compared': len(strategies),
        'comparisons': comparison_results,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, "strategy_comparison.pkl")
    evaluator.save_evaluation_results(results, results_path)
    
    # Generate report
    report_path = os.path.join(output_dir, "comparison_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print_comparison_summary(results)
    
    return results


def print_evaluation_summary(results):
    """Print summary of checkpoint evaluation results"""
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    iterations = results.get('iterations', [])
    exploitability = results.get('exploitability', [])
    head_to_head = results.get('head_to_head_results', [])
    
    if iterations and exploitability:
        print(f"Checkpoints evaluated: {len(iterations)}")
        print(f"Training range: {iterations[0]:,} to {iterations[-1]:,} iterations")
        print(f"Initial exploitability: {exploitability[0]:.6f}")
        print(f"Final exploitability: {exploitability[-1]:.6f}")
        
        improvement = exploitability[0] - exploitability[-1]
        improvement_pct = (improvement / exploitability[0]) * 100
        print(f"Total improvement: {improvement:.6f} ({improvement_pct:.1f}%)")
        
        # Convergence analysis
        convergence = PokerMetrics.convergence_rate(exploitability)
        if convergence.get('is_converged', False):
            print("Status: ✓ Strategy appears to have converged")
        else:
            print("Status: ⚠ Strategy still improving")
        
        # Head-to-head performance
        valid_h2h = [h for h in head_to_head if h is not None]
        if valid_h2h:
            avg_winrate = sum(h['strategy1_winrate'] for h in valid_h2h) / len(valid_h2h)
            print(f"Avg improvement vs previous: {avg_winrate:.1%} winrate")
    
    print("\nFiles generated:")
    print("  - evaluation_results.pkl (raw data)")
    print("  - evaluation_report.json (detailed analysis)")
    print("  - training_progress.png (visualizations)")


def print_single_strategy_summary(results):
    """Print summary of single strategy evaluation"""
    print("\n" + "=" * 50)
    print("STRATEGY ANALYSIS")
    print("=" * 50)
    
    print(f"Strategy: {os.path.basename(results['strategy_path'])}")
    print(f"Training iterations: {results['training_iterations']:,}")
    print(f"Information sets: {results['total_infosets']:,}")
    print(f"Exploitability: {results['exploitability']:.6f}")
    print(f"Strategy entropy: {results['strategy_entropy']:.4f}")
    
    # Action frequency analysis
    action_freq = results['action_frequency']['frequencies']
    if action_freq:
        print("\nAction frequencies:")
        for action, freq in sorted(action_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"  {action}: {freq:.3f}")


def print_comparison_summary(results):
    """Print summary of strategy comparison"""
    print("\n" + "=" * 50)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 50)
    
    comparisons = results['comparisons']
    
    for comp_name, comp_data in comparisons.items():
        print(f"\n{comp_name}:")
        h2h = comp_data['head_to_head']
        print(f"  Strategy 1 winrate: {h2h['strategy1_winrate']:.1%}")
        print(f"  Strategy 2 winrate: {h2h['strategy2_winrate']:.1%}")
        print(f"  Tie rate: {h2h['tie_rate']:.1%}")
        print(f"  Nash distance: {comp_data['nash_distance']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate poker bot strategy quality")
    parser.add_argument("--config", default="config/game_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", choices=["checkpoints", "single", "compare"],
                       default="checkpoints",
                       help="Evaluation mode")
    parser.add_argument("--checkpoint-dir", default="data/blueprints",
                       help="Directory containing checkpoint files")
    parser.add_argument("--blueprint", 
                       help="Path to single blueprint file (for single mode)")
    parser.add_argument("--compare-blueprints", nargs="+",
                       help="Paths to blueprint files to compare")
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU-only evaluation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "single" and not args.blueprint:
        print("Error: --blueprint required for single mode")
        return
    
    if args.mode == "compare" and not args.compare_blueprints:
        print("Error: --compare-blueprints required for compare mode")
        return
    
    use_gpu = not args.cpu
    
    # Run evaluation based on mode
    if args.mode == "checkpoints":
        if not os.path.exists(args.checkpoint_dir):
            print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist")
            return
        
        results = evaluate_checkpoints(
            args.config, args.checkpoint_dir, args.output_dir, use_gpu
        )
    
    elif args.mode == "single":
        if not os.path.exists(args.blueprint):
            print(f"Error: Blueprint file {args.blueprint} does not exist")
            return
        
        results = evaluate_single_strategy(
            args.blueprint, args.config, args.output_dir, use_gpu
        )
    
    elif args.mode == "compare":
        for blueprint in args.compare_blueprints:
            if not os.path.exists(blueprint):
                print(f"Error: Blueprint file {blueprint} does not exist")
                return
        
        results = compare_strategies(
            args.compare_blueprints, args.config, args.output_dir, use_gpu
        )
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()