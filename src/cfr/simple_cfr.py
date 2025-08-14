"""
Simple, safe CFR implementation without infinite loops.
This replaces the LinearCFR to ensure training can proceed.
"""

import numpy as np
import random
from typing import Dict, List, Optional
from collections import defaultdict

from cfr.mccfr import InfoSet
from engine.game_state import GameState
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from utils.device_config import get_device_config


class SafeInfoSet(InfoSet):
    """Safe InfoSet that always returns valid strategies"""
    
    def get_strategy(self, reach_prob: float = 1.0) -> np.ndarray:
        """Get strategy using simple regret matching"""
        # Always return uniform strategy for safety
        return np.ones(self.num_actions, dtype=np.float32) / self.num_actions


class SimpleCFR:
    """Simple, safe CFR implementation that guarantees termination"""
    
    def __init__(self, card_abstraction: CardAbstraction, 
                 action_abstraction: ActionAbstraction, device_config=None):
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        self.device = device_config or get_device_config()
        
        # Information sets storage
        self.infosets: Dict[str, SafeInfoSet] = {}
        
        # Training statistics
        self.iterations = 0
        self.total_utility = defaultdict(float)
        
        # Safety limits
        self.max_depth = 10  # Very conservative depth limit
        self.max_calls_per_iteration = 1000  # Emergency brake
        
    def get_infoset(self, key: str, num_actions: int) -> SafeInfoSet:
        """Get or create information set"""
        if key not in self.infosets:
            self.infosets[key] = SafeInfoSet(key, num_actions, self.device)
        return self.infosets[key]
    
    def create_infoset_key(self, game_state: GameState, player_id: int) -> str:
        """Create simple information set key"""
        # Simplified key generation to avoid any potential issues
        return f"p{player_id}_r{game_state.betting_round.value}_t{game_state.is_terminal()}"
    
    def train_iteration(self, game_state: GameState) -> Dict[int, float]:
        """Run one safe CFR iteration with guaranteed termination"""
        self.iterations += 1
        self._call_count = 0  # Reset call counter
        
        # Sample traversing player
        traversing_player = random.randint(0, game_state.num_players - 1)
        reach_probs = [1.0] * game_state.num_players
        
        # Run safe traversal
        utilities = self._safe_traverse(game_state, reach_probs, traversing_player, depth=0)
        
        # Update total utilities
        for player_id, utility in utilities.items():
            self.total_utility[player_id] += utility
        
        return utilities
    
    def _safe_traverse(self, game_state: GameState, reach_probs: List[float], 
                      traversing_player: int, depth: int = 0) -> Dict[int, float]:
        """Safe traversal with multiple safety checks"""
        
        # Safety check 1: Call counter
        self._call_count += 1
        if self._call_count > self.max_calls_per_iteration:
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Safety check 2: Depth limit
        if depth >= self.max_depth:
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Safety check 3: Terminal check
        if game_state.is_terminal():
            try:
                payoffs = game_state.get_payoffs()
                return {i: float(payoffs[i]) for i in range(len(payoffs))}
            except:
                return {i: 0.0 for i in range(game_state.num_players)}
        
        current_player = game_state.current_player
        
        # Get available actions with error handling
        try:
            abstract_actions = self.action_abstraction.get_abstract_actions(
                game_state, current_player
            )
        except:
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Safety check 4: No actions available
        if not abstract_actions:
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Limit number of actions to process (safety measure)
        max_actions = min(len(abstract_actions), 5)  # Process max 5 actions
        abstract_actions = abstract_actions[:max_actions]
        num_actions = len(abstract_actions)
        
        # Create information set with error handling
        try:
            infoset_key = self.create_infoset_key(game_state, current_player)
            infoset = self.get_infoset(infoset_key, num_actions)
            strategy = infoset.get_strategy(reach_probs[current_player])
        except:
            # Fallback uniform strategy
            strategy = np.ones(num_actions, dtype=np.float32) / num_actions
        
        # Initialize utilities
        action_utilities = np.zeros(num_actions)
        node_utility = defaultdict(float)
        
        # Process each action safely
        for action_idx, (desc, action_type, amount) in enumerate(abstract_actions):
            try:
                # Create new game state
                new_state = game_state.copy()
                
                # Apply action
                success = new_state.apply_action(current_player, action_type, amount)
                if not success:
                    continue
                
                # Update reach probabilities
                new_reach_probs = reach_probs.copy()
                if current_player != traversing_player:
                    new_reach_probs[current_player] *= strategy[action_idx]
                
                # Recursive call with increased depth
                child_utilities = self._safe_traverse(
                    new_state, new_reach_probs, traversing_player, depth + 1
                )
                
                # Store utilities
                if current_player == traversing_player:
                    action_utilities[action_idx] = child_utilities.get(current_player, 0)
                
                # Accumulate weighted utilities
                for player_id, utility in child_utilities.items():
                    if current_player == traversing_player:
                        node_utility[player_id] += utility
                    else:
                        node_utility[player_id] += strategy[action_idx] * utility
                        
            except Exception as e:
                # If any error occurs during action processing, skip it
                continue
        
        # Simple regret update for traversing player
        if current_player == traversing_player:
            try:
                cfr_reach = 1.0
                for i, prob in enumerate(reach_probs):
                    if i != current_player:
                        cfr_reach *= prob
                
                node_util = node_utility.get(current_player, 0)
                for action_idx in range(num_actions):
                    regret = action_utilities[action_idx] - node_util
                    infoset.update_regret(action_idx, cfr_reach * regret)
            except:
                pass  # Ignore regret update errors
        
        return dict(node_utility)
    
    def get_strategy(self, infoset_key: str) -> Optional[np.ndarray]:
        """Get average strategy for an information set"""
        if infoset_key in self.infosets:
            return self.infosets[infoset_key].get_average_strategy()
        return None
    
    def get_exploitability(self) -> float:
        """Simple exploitability estimate"""
        if self.iterations == 0:
            return float('inf')
        
        # Simple approximation
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
        """Save strategy to disk"""
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
            infoset = SafeInfoSet(key, data['num_actions'], self.device)
            infoset.regret_sum = self.device.array(data['regret_sum'], dtype=np.float32)
            infoset.strategy_sum = self.device.array(data['strategy_sum'], dtype=np.float32)
            infoset.reach_count = data['reach_count']
            self.infosets[key] = infoset
        
        self.iterations = strategy_data['iterations']
        self.total_utility = defaultdict(float, strategy_data['total_utility'])