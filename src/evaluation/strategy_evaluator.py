"""
Strategy evaluation metrics for measuring poker bot quality.
Includes exploitability, head-to-head comparisons, and Nash distance.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
import os
from datetime import datetime

from engine.game_state import GameState, Action, BettingRound
from cfr.mccfr import MCCFR
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction


class StrategyEvaluator:
    """Evaluates poker strategy quality using multiple metrics"""
    
    def __init__(self, card_abstraction: CardAbstraction, 
                 action_abstraction: ActionAbstraction):
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        
        # Evaluation history
        self.evaluation_history = []
    
    def compute_exploitability(self, cfr_solver: MCCFR, 
                             num_samples: int = 1000) -> float:
        """
        Compute more accurate exploitability using best response calculation.
        Lower values indicate stronger play.
        """
        if cfr_solver.iterations == 0:
            return float('inf')
        
        total_exploitability = 0.0
        samples_evaluated = 0
        
        # Sample different game situations
        for _ in range(num_samples):
            # Create random game state
            num_players = random.randint(2, 6)
            game_state = self._create_random_game_state(num_players)
            
            # Compute exploitability for each player
            for player_id in range(num_players):
                if game_state.players[player_id].is_folded:
                    continue
                
                # Get current strategy value
                current_value = self._evaluate_strategy_value(
                    cfr_solver, game_state, player_id
                )
                
                # Get best response value
                best_response_value = self._compute_best_response_value(
                    cfr_solver, game_state, player_id
                )
                
                # Exploitability is the difference
                player_exploitability = max(0, best_response_value - current_value)
                total_exploitability += player_exploitability
                samples_evaluated += 1
        
        if samples_evaluated == 0:
            return float('inf')
        
        return total_exploitability / samples_evaluated
    
    def _create_random_game_state(self, num_players: int) -> GameState:
        """Create a random game state for evaluation"""
        game_state = GameState(
            num_players=num_players,
            starting_stack=10000,
            small_blind=50,
            big_blind=100
        )
        
        # Deal random cards
        deck = self._create_shuffled_deck()
        
        # Deal hole cards
        card_idx = 0
        for player in game_state.players:
            player.hole_cards = [deck[card_idx], deck[card_idx + 1]]
            card_idx += 2
        
        # Randomly advance to different betting rounds
        betting_round = random.choice(list(BettingRound))
        game_state.betting_round = betting_round
        
        # Set appropriate community cards
        if betting_round == BettingRound.PREFLOP:
            game_state.community_cards = []
        elif betting_round == BettingRound.FLOP:
            game_state.community_cards = deck[card_idx:card_idx + 3]
        elif betting_round == BettingRound.TURN:
            game_state.community_cards = deck[card_idx:card_idx + 4]
        elif betting_round == BettingRound.RIVER:
            game_state.community_cards = deck[card_idx:card_idx + 5]
        
        return game_state
    
    def _create_shuffled_deck(self):
        """Create and shuffle a deck of cards"""
        from engine.hand_evaluator import Card, Rank, Suit
        
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        
        random.shuffle(deck)
        return deck
    
    def _evaluate_strategy_value(self, cfr_solver: MCCFR, 
                                game_state: GameState, player_id: int) -> float:
        """Evaluate expected value of current strategy for a player"""
        return self._traverse_strategy_value(cfr_solver, game_state, player_id, 1.0)
    
    def _traverse_strategy_value(self, cfr_solver: MCCFR, game_state: GameState,
                               player_id: int, reach_prob: float) -> float:
        """Recursively compute strategy value"""
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            return payoffs[player_id]
        
        current_player = game_state.current_player
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, current_player
        )
        
        if not abstract_actions:
            return self._traverse_strategy_value(cfr_solver, game_state, player_id, reach_prob)
        
        # Get strategy for current player
        infoset_key = cfr_solver.create_infoset_key(game_state, current_player)
        strategy = cfr_solver.get_strategy(infoset_key)
        
        if strategy is None:
            # Use uniform random strategy if no data
            strategy = np.ones(len(abstract_actions)) / len(abstract_actions)
        
        expected_value = 0.0
        
        # Compute expected value over all actions
        for action_idx, (desc, action_type, amount) in enumerate(abstract_actions):
            # Create new game state
            new_state = game_state.copy()
            success = new_state.apply_action(current_player, action_type, amount)
            
            if not success:
                continue
            
            # Compute probability and value
            action_prob = strategy[action_idx] if action_idx < len(strategy) else 0.1
            action_value = self._traverse_strategy_value(
                cfr_solver, new_state, player_id, reach_prob * action_prob
            )
            
            expected_value += action_prob * action_value
        
        return expected_value
    
    def _compute_best_response_value(self, cfr_solver: MCCFR, 
                                   game_state: GameState, player_id: int) -> float:
        """Compute best response value for a player"""
        return self._traverse_best_response(cfr_solver, game_state, player_id, 1.0)
    
    def _traverse_best_response(self, cfr_solver: MCCFR, game_state: GameState,
                              player_id: int, reach_prob: float) -> float:
        """Recursively compute best response value"""
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            return payoffs[player_id]
        
        current_player = game_state.current_player
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, current_player
        )
        
        if not abstract_actions:
            return self._traverse_best_response(cfr_solver, game_state, player_id, reach_prob)
        
        if current_player == player_id:
            # For the player we're computing best response for, choose best action
            best_value = float('-inf')
            
            for action_idx, (desc, action_type, amount) in enumerate(abstract_actions):
                new_state = game_state.copy()
                success = new_state.apply_action(current_player, action_type, amount)
                
                if not success:
                    continue
                
                action_value = self._traverse_best_response(
                    cfr_solver, new_state, player_id, reach_prob
                )
                
                best_value = max(best_value, action_value)
            
            return best_value if best_value != float('-inf') else 0.0
        
        else:
            # For other players, use their current strategy
            infoset_key = cfr_solver.create_infoset_key(game_state, current_player)
            strategy = cfr_solver.get_strategy(infoset_key)
            
            if strategy is None:
                strategy = np.ones(len(abstract_actions)) / len(abstract_actions)
            
            expected_value = 0.0
            
            for action_idx, (desc, action_type, amount) in enumerate(abstract_actions):
                new_state = game_state.copy()
                success = new_state.apply_action(current_player, action_type, amount)
                
                if not success:
                    continue
                
                action_prob = strategy[action_idx] if action_idx < len(strategy) else 0.1
                action_value = self._traverse_best_response(
                    cfr_solver, new_state, player_id, reach_prob * action_prob
                )
                
                expected_value += action_prob * action_value
            
            return expected_value
    
    def head_to_head_evaluation(self, strategy1: MCCFR, strategy2: MCCFR,
                              num_hands: int = 1000) -> Dict:
        """
        Evaluate two strategies against each other.
        Returns win rates and average utilities.
        """
        results = {
            'strategy1_wins': 0,
            'strategy2_wins': 0,
            'ties': 0,
            'strategy1_utility': 0.0,
            'strategy2_utility': 0.0,
            'hands_played': 0
        }
        
        for hand_num in range(num_hands):
            # Create heads-up game
            game_state = self._create_random_game_state(2)
            
            # Play hand with both strategies
            final_state = self._play_hand_with_strategies(
                game_state, [strategy1, strategy2]
            )
            
            if final_state and final_state.is_terminal():
                payoffs = final_state.get_payoffs()
                
                results['strategy1_utility'] += payoffs[0]
                results['strategy2_utility'] += payoffs[1]
                
                if payoffs[0] > payoffs[1]:
                    results['strategy1_wins'] += 1
                elif payoffs[1] > payoffs[0]:
                    results['strategy2_wins'] += 1
                else:
                    results['ties'] += 1
                
                results['hands_played'] += 1
        
        # Calculate rates
        if results['hands_played'] > 0:
            results['strategy1_winrate'] = results['strategy1_wins'] / results['hands_played']
            results['strategy2_winrate'] = results['strategy2_wins'] / results['hands_played']
            results['tie_rate'] = results['ties'] / results['hands_played']
            results['avg_utility_diff'] = (results['strategy1_utility'] - results['strategy2_utility']) / results['hands_played']
        
        return results
    
    def _play_hand_with_strategies(self, game_state: GameState, 
                                 strategies: List[MCCFR]) -> Optional[GameState]:
        """Play a hand using different strategies for each player"""
        max_actions = 1000  # Prevent infinite loops
        actions_taken = 0
        
        while not game_state.is_terminal() and actions_taken < max_actions:
            current_player = game_state.current_player
            strategy = strategies[current_player % len(strategies)]
            
            # Get available actions
            abstract_actions = self.action_abstraction.get_abstract_actions(
                game_state, current_player
            )
            
            if not abstract_actions:
                break
            
            # Get strategy for current information set
            infoset_key = strategy.create_infoset_key(game_state, current_player)
            action_probs = strategy.get_strategy(infoset_key)
            
            if action_probs is None or len(action_probs) != len(abstract_actions):
                # Use uniform random if no strategy available
                action_idx = random.randint(0, len(abstract_actions) - 1)
            else:
                # Sample action according to strategy
                action_idx = np.random.choice(len(abstract_actions), p=action_probs)
            
            desc, action_type, amount = abstract_actions[action_idx]
            
            # Apply action
            success = game_state.apply_action(current_player, action_type, amount)
            if not success:
                break
            
            actions_taken += 1
        
        return game_state
    
    def evaluate_strategy_progression(self, checkpoint_dir: str) -> Dict:
        """
        Evaluate how strategy quality improves across training checkpoints.
        """
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.pkl'):
                iteration = int(filename.split('_')[1].split('.')[0])
                checkpoint_files.append((iteration, filename))
        
        checkpoint_files.sort()  # Sort by iteration
        
        results = {
            'iterations': [],
            'exploitability': [],
            'head_to_head_results': [],
            'timestamps': []
        }
        
        previous_strategy = None
        
        for iteration, filename in checkpoint_files:
            print(f"Evaluating checkpoint at iteration {iteration}...")
            
            # Load strategy
            filepath = os.path.join(checkpoint_dir, filename)
            current_strategy = self._load_strategy_from_checkpoint(filepath)
            
            if current_strategy is None:
                continue
            
            # Compute exploitability
            exploitability = self.compute_exploitability(current_strategy, num_samples=500)
            
            results['iterations'].append(iteration)
            results['exploitability'].append(exploitability)
            results['timestamps'].append(datetime.now().isoformat())
            
            # Head-to-head comparison with previous strategy
            if previous_strategy is not None:
                h2h_result = self.head_to_head_evaluation(
                    current_strategy, previous_strategy, num_hands=500
                )
                results['head_to_head_results'].append(h2h_result)
            else:
                results['head_to_head_results'].append(None)
            
            previous_strategy = current_strategy
            
            print(f"  Exploitability: {exploitability:.6f}")
            if results['head_to_head_results'][-1]:
                winrate = results['head_to_head_results'][-1]['strategy1_winrate']
                print(f"  vs Previous: {winrate:.2%} winrate")
        
        return results
    
    def _load_strategy_from_checkpoint(self, filepath: str) -> Optional[MCCFR]:
        """Load CFR strategy from checkpoint file"""
        try:
            from cfr.linear_cfr import LinearCFR
            
            # Create temporary CFR solver
            cfr_solver = LinearCFR(self.card_abstraction, self.action_abstraction)
            cfr_solver.load_strategy(filepath)
            
            return cfr_solver
        except Exception as e:
            print(f"Error loading checkpoint {filepath}: {e}")
            return None
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation results saved to {output_path}")
    
    def load_evaluation_results(self, filepath: str) -> Dict:
        """Load evaluation results from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)