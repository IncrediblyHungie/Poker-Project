"""
Additional evaluation metrics for poker strategy analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os

from cfr.mccfr import MCCFR


class PokerMetrics:
    """Advanced metrics for poker strategy evaluation"""
    
    @staticmethod
    def nash_distance(strategy1: MCCFR, strategy2: MCCFR, 
                     sample_infosets: List[str] = None) -> float:
        """
        Compute Nash distance between two strategies.
        Measures how different the strategies are in terms of action probabilities.
        """
        if sample_infosets is None:
            # Use intersection of both strategies' information sets
            common_infosets = set(strategy1.infosets.keys()) & set(strategy2.infosets.keys())
            sample_infosets = list(common_infosets)
        
        if not sample_infosets:
            return float('inf')
        
        total_distance = 0.0
        valid_infosets = 0
        
        for infoset_key in sample_infosets:
            if infoset_key in strategy1.infosets and infoset_key in strategy2.infosets:
                # Get average strategies
                strategy1_probs = strategy1.infosets[infoset_key].get_average_strategy()
                strategy2_probs = strategy2.infosets[infoset_key].get_average_strategy()
                
                # Ensure same dimensions
                min_len = min(len(strategy1_probs), len(strategy2_probs))
                if min_len > 0:
                    s1 = strategy1_probs[:min_len]
                    s2 = strategy2_probs[:min_len]
                    
                    # Compute L2 distance
                    distance = np.linalg.norm(s1 - s2)
                    total_distance += distance
                    valid_infosets += 1
        
        return total_distance / valid_infosets if valid_infosets > 0 else float('inf')
    
    @staticmethod
    def strategy_entropy(strategy: MCCFR, sample_size: int = 1000) -> float:
        """
        Compute average entropy of strategy across information sets.
        Higher entropy indicates more random/mixed strategies.
        """
        if not strategy.infosets:
            return 0.0
        
        total_entropy = 0.0
        valid_infosets = 0
        
        # Sample information sets if too many
        infoset_keys = list(strategy.infosets.keys())
        if len(infoset_keys) > sample_size:
            infoset_keys = np.random.choice(infoset_keys, sample_size, replace=False)
        
        for infoset_key in infoset_keys:
            infoset = strategy.infosets[infoset_key]
            probs = infoset.get_average_strategy()
            
            # Compute entropy: -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            probs_safe = np.maximum(probs, epsilon)
            entropy = -np.sum(probs_safe * np.log(probs_safe))
            
            total_entropy += entropy
            valid_infosets += 1
        
        return total_entropy / valid_infosets if valid_infosets > 0 else 0.0
    
    @staticmethod
    def action_frequency_analysis(strategy: MCCFR) -> Dict:
        """
        Analyze how frequently different actions are taken.
        """
        action_counts = defaultdict(int)
        total_probability = 0.0
        
        for infoset_key, infoset in strategy.infosets.items():
            probs = infoset.get_average_strategy()
            
            for action_idx, prob in enumerate(probs):
                action_name = f"action_{action_idx}"
                action_counts[action_name] += prob
                total_probability += prob
        
        # Normalize to get frequencies
        if total_probability > 0:
            action_frequencies = {
                action: count / total_probability 
                for action, count in action_counts.items()
            }
        else:
            action_frequencies = {}
        
        return {
            'frequencies': action_frequencies,
            'total_infosets': len(strategy.infosets),
            'total_probability_mass': total_probability
        }
    
    @staticmethod
    def strategy_stability(strategies: List[MCCFR], window_size: int = 5) -> List[float]:
        """
        Measure strategy stability over time using rolling Nash distance.
        """
        if len(strategies) < 2:
            return []
        
        stability_scores = []
        
        for i in range(len(strategies) - 1):
            current_strategy = strategies[i]
            next_strategy = strategies[i + 1]
            
            # Compute Nash distance between consecutive strategies
            distance = PokerMetrics.nash_distance(current_strategy, next_strategy)
            stability_scores.append(distance)
        
        return stability_scores
    
    @staticmethod
    def convergence_rate(exploitability_history: List[float]) -> Dict:
        """
        Analyze convergence rate of the strategy.
        """
        if len(exploitability_history) < 10:
            return {'insufficient_data': True}
        
        # Compute moving averages
        window_size = max(10, len(exploitability_history) // 10)
        moving_avg = []
        
        for i in range(window_size, len(exploitability_history)):
            avg = np.mean(exploitability_history[i-window_size:i])
            moving_avg.append(avg)
        
        # Compute rate of improvement
        if len(moving_avg) >= 2:
            improvement_rate = (moving_avg[0] - moving_avg[-1]) / len(moving_avg)
        else:
            improvement_rate = 0.0
        
        # Check if converged (improvement rate below threshold)
        convergence_threshold = 1e-6
        is_converged = abs(improvement_rate) < convergence_threshold
        
        return {
            'improvement_rate': improvement_rate,
            'is_converged': is_converged,
            'final_exploitability': exploitability_history[-1],
            'best_exploitability': min(exploitability_history),
            'convergence_threshold': convergence_threshold
        }


class EvaluationVisualizer:
    """Create visualizations for strategy evaluation results"""
    
    @staticmethod
    def plot_training_progress(results: Dict, output_path: str = None):
        """Plot training progress metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Analysis', fontsize=16)
        
        iterations = results.get('iterations', [])
        exploitability = results.get('exploitability', [])
        head_to_head = results.get('head_to_head_results', [])
        
        # Plot 1: Exploitability over time
        if iterations and exploitability:
            axes[0, 0].plot(iterations, exploitability, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Training Iterations')
            axes[0, 0].set_ylabel('Exploitability')
            axes[0, 0].set_title('Exploitability Reduction')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
        
        # Plot 2: Win rate vs previous iteration
        if head_to_head:
            win_rates = [h2h['strategy1_winrate'] for h2h in head_to_head if h2h is not None]
            valid_iterations = iterations[1:len(win_rates)+1]
            
            if win_rates:
                axes[0, 1].plot(valid_iterations, win_rates, 'g-', linewidth=2)
                axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% (no improvement)')
                axes[0, 1].set_xlabel('Training Iterations')
                axes[0, 1].set_ylabel('Win Rate vs Previous')
                axes[0, 1].set_title('Head-to-Head Improvement')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
        
        # Plot 3: Utility differences
        if head_to_head:
            utility_diffs = [h2h['avg_utility_diff'] for h2h in head_to_head if h2h is not None]
            valid_iterations = iterations[1:len(utility_diffs)+1]
            
            if utility_diffs:
                axes[1, 0].plot(valid_iterations, utility_diffs, 'm-', linewidth=2)
                axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Training Iterations')
                axes[1, 0].set_ylabel('Avg Utility Difference')
                axes[1, 0].set_title('Utility Improvement vs Previous')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Convergence analysis
        if exploitability and len(exploitability) > 1:
            # Compute improvement rate
            improvements = [-1 * np.diff(exploitability)]
            if improvements:
                axes[1, 1].plot(iterations[1:], improvements[0], 'orange', linewidth=2)
                axes[1, 1].set_xlabel('Training Iterations')
                axes[1, 1].set_ylabel('Exploitability Improvement')
                axes[1, 1].set_title('Rate of Improvement')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_exploitability_comparison(strategies_data: List[Tuple[str, List[float]]], 
                                     output_path: str = None):
        """Compare exploitability across different training runs"""
        plt.figure(figsize=(12, 8))
        
        for name, exploitability_data in strategies_data:
            iterations = list(range(len(exploitability_data)))
            plt.plot(iterations, exploitability_data, linewidth=2, label=name)
        
        plt.xlabel('Training Iterations (scaled)')
        plt.ylabel('Exploitability')
        plt.title('Exploitability Comparison Across Training Runs')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Exploitability comparison saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_evaluation_report(results: Dict, output_path: str):
        """Create a comprehensive evaluation report"""
        report = {
            'evaluation_summary': {
                'total_iterations_evaluated': len(results.get('iterations', [])),
                'final_exploitability': results.get('exploitability', [0])[-1] if results.get('exploitability') else 0,
                'best_exploitability': min(results.get('exploitability', [float('inf')])),
                'evaluation_timestamp': results.get('timestamps', ['unknown'])[-1] if results.get('timestamps') else 'unknown'
            },
            'convergence_analysis': PokerMetrics.convergence_rate(results.get('exploitability', [])),
            'raw_data': results
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {output_path}")
        
        return report