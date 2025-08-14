"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) implementation.
Core algorithm for computing poker strategies using self-play.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from engine.game_state import GameState, Action, BettingRound
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from utils.device_config import get_device_config


class InfoSet:
    """Information set for CFR algorithm"""
    
    def __init__(self, key: str, num_actions: int, device_config=None):
        self.key = key
        self.num_actions = num_actions
        
        # Get device configuration
        self.device = device_config or get_device_config()
        
        # CFR data structures on appropriate device
        self.regret_sum = self.device.zeros(num_actions, dtype=np.float32)
        self.strategy_sum = self.device.zeros(num_actions, dtype=np.float32)
        self.reach_count = 0
        
    def get_strategy(self, reach_prob: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        # Regret matching: strategy proportional to positive regrets
        positive_regrets = self.device.maximum(self.regret_sum, 0)
        normalizing_sum = self.device.sum(positive_regrets)
        
        # Convert to scalar for comparison
        normalizing_sum_scalar = float(self.device.to_numpy(normalizing_sum))
        
        if normalizing_sum_scalar > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            # Uniform random strategy if no positive regrets
            strategy = self.device.ones(self.num_actions, dtype=np.float32) / self.num_actions
        
        # Update strategy sum for averaging
        self.strategy_sum += reach_prob * strategy
        self.reach_count += 1
        
        return self.device.to_numpy(strategy)
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        normalizing_sum = self.device.sum(self.strategy_sum)
        normalizing_sum_scalar = float(self.device.to_numpy(normalizing_sum))
        
        if normalizing_sum_scalar > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = self.device.ones(self.num_actions, dtype=np.float32) / self.num_actions
        
        return self.device.to_numpy(avg_strategy)
    
    def update_regret(self, action_index: int, regret: float):
        """Update regret for a specific action"""
        self.regret_sum[action_index] += regret


class MCCFR:
    """Monte Carlo Counterfactual Regret Minimization solver"""
    
    def __init__(self, card_abstraction: CardAbstraction, 
                 action_abstraction: ActionAbstraction, device_config=None):
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        
        # Get device configuration
        self.device = device_config or get_device_config()
        
        # Information sets storage
        self.infosets: Dict[str, InfoSet] = {}
        
        # Training statistics
        self.iterations = 0
        self.total_utility = defaultdict(float)
        
    def get_infoset(self, key: str, num_actions: int) -> InfoSet:
        """Get or create information set"""
        if key not in self.infosets:
            self.infosets[key] = InfoSet(key, num_actions, self.device)
        
        return self.infosets[key]
    
    def create_infoset_key(self, game_state: GameState, player_id: int) -> str:
        """Create information set key for current game state"""
        player = game_state.players[player_id]
        
        # Get card buckets
        hole_cards_bucket = self.card_abstraction.get_bucket(
            player.hole_cards, 
            game_state.community_cards,
            game_state.betting_round
        )
        
        # Get board bucket (if post-flop)
        if game_state.betting_round == BettingRound.PREFLOP:
            board_bucket = 0
        else:
            board_bucket = self.card_abstraction.get_bucket(
                [], game_state.community_cards, game_state.betting_round
            )
        
        # Create betting history string
        betting_history = []
        for pid, action, amount in game_state.round_betting_history[game_state.betting_round]:
            betting_history.append(f"{pid}:{action.value}:{amount}")
        
        betting_str = "|".join(betting_history)
        
        return f"{hole_cards_bucket}#{board_bucket}#{betting_str}#{game_state.betting_round.value}"
    
    def train_iteration(self, game_state: GameState) -> Dict[int, float]:
        """
        Run one iteration of MCCFR training.
        Returns utility for each player.
        """
        self.iterations += 1
        
        # Sample traversing player (external sampling)
        traversing_player = random.randint(0, game_state.num_players - 1)
        
        # Initialize reach probabilities
        reach_probs = [1.0] * game_state.num_players
        
        # Run MCCFR traversal
        utilities = self._mccfr_traverse(game_state, reach_probs, traversing_player)
        
        # Update total utilities
        for player_id, utility in utilities.items():
            self.total_utility[player_id] += utility
        
        return utilities
    
    def _mccfr_traverse(self, game_state: GameState, reach_probs: List[float], 
                       traversing_player: int, depth: int = 0) -> Dict[int, float]:
        """
        Recursive MCCFR traversal of game tree.
        """
        # Prevent infinite recursion
        if depth > 100:  # Maximum depth limit
            # Return neutral utilities to avoid infinite loops
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Terminal node
        if game_state.is_terminal():
            payoffs = game_state.get_payoffs()
            return {i: payoffs[i] for i in range(len(payoffs))}
        
        current_player = game_state.current_player
        
        # Get available actions
        abstract_actions = self.action_abstraction.get_abstract_actions(
            game_state, current_player
        )
        
        if not abstract_actions:
            # No legal actions, advance game state
            return self._mccfr_traverse(game_state, reach_probs, traversing_player, depth + 1)
        
        num_actions = len(abstract_actions)
        
        # Create information set
        infoset_key = self.create_infoset_key(game_state, current_player)
        infoset = self.get_infoset(infoset_key, num_actions)
        
        # Get strategy
        strategy = infoset.get_strategy(reach_probs[current_player])
        
        # Initialize utilities
        action_utilities = np.zeros(num_actions)
        node_utility = defaultdict(float)
        
        # Traverse each action
        for action_idx, (desc, action_type, amount) in enumerate(abstract_actions):
            # Create new game state
            new_state = game_state.copy()
            
            # Apply action
            success = new_state.apply_action(current_player, action_type, amount)
            if not success:
                continue
            
            # Update reach probabilities
            new_reach_probs = reach_probs.copy()
            if current_player == traversing_player:
                # Don't update reach prob for traversing player (external sampling)
                pass
            else:
                new_reach_probs[current_player] *= strategy[action_idx]
            
            # Recursive call
            child_utilities = self._mccfr_traverse(new_state, new_reach_probs, traversing_player, depth + 1)
            
            # Store action utility for traversing player
            if current_player == traversing_player:
                action_utilities[action_idx] = child_utilities.get(current_player, 0)
            
            # Accumulate weighted utilities
            for player_id, utility in child_utilities.items():
                if current_player == traversing_player:
                    node_utility[player_id] += utility
                else:
                    node_utility[player_id] += strategy[action_idx] * utility
        
        # Update regrets for traversing player
        if current_player == traversing_player:
            # Calculate counterfactual reach probability
            cfr_reach = 1.0
            for i, prob in enumerate(reach_probs):
                if i != current_player:
                    cfr_reach *= prob
            
            # Calculate regrets
            node_util = node_utility.get(current_player, 0)
            for action_idx in range(num_actions):
                regret = action_utilities[action_idx] - node_util
                infoset.update_regret(action_idx, cfr_reach * regret)
        
        return dict(node_utility)
    
    def get_strategy(self, infoset_key: str) -> Optional[np.ndarray]:
        """Get average strategy for an information set"""
        if infoset_key in self.infosets:
            return self.infosets[infoset_key].get_average_strategy()
        return None
    
    def get_exploitability(self) -> float:
        """
        Estimate exploitability of current strategy.
        Simplified version - full implementation would compute best response.
        """
        if self.iterations == 0:
            return float('inf')
        
        # Simple approximation based on regret sums
        total_regret = 0
        total_infosets = 0
        
        for infoset in self.infosets.values():
            if infoset.reach_count > 0:
                avg_regret = np.mean(np.maximum(infoset.regret_sum, 0))
                total_regret += avg_regret
                total_infosets += 1
        
        if total_infosets == 0:
            return float('inf')
        
        return total_regret / total_infosets
    
    def save_strategy(self, filepath: str):
        """Save current strategy to disk"""
        import pickle
        import os
        
        strategy_data = {
            'infosets': {key: {
                'regret_sum': self.device.to_numpy(infoset.regret_sum),
                'strategy_sum': self.device.to_numpy(infoset.strategy_sum),
                'reach_count': infoset.reach_count,
                'num_actions': infoset.num_actions
            } for key, infoset in self.infosets.items()},
            'iterations': self.iterations,
            'total_utility': dict(self.total_utility)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(strategy_data, f)
    
    def load_strategy(self, filepath: str):
        """Load strategy from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            strategy_data = pickle.load(f)
        
        # Reconstruct infosets
        self.infosets = {}
        for key, data in strategy_data['infosets'].items():
            infoset = InfoSet(key, data['num_actions'], self.device)
            infoset.regret_sum = self.device.array(data['regret_sum'], dtype=np.float32)
            infoset.strategy_sum = self.device.array(data['strategy_sum'], dtype=np.float32)
            infoset.reach_count = data['reach_count']
            self.infosets[key] = infoset
        
        self.iterations = strategy_data['iterations']
        self.total_utility = defaultdict(float, strategy_data['total_utility'])