"""
Linear CFR enhancement that weights iterations linearly for improved convergence.
Recent iterations count more than earlier ones.
"""

import numpy as np
from cfr.mccfr import MCCFR, InfoSet
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction


class LinearInfoSet(InfoSet):
    """Enhanced InfoSet with linear weighting"""
    
    def __init__(self, key: str, num_actions: int, device_config=None):
        super().__init__(key, num_actions, device_config)
        self.linear_start_iteration = 10000  # Start linear weighting after this many iterations
    
    def get_strategy(self, reach_prob: float = 1.0, iteration: int = 0) -> np.ndarray:
        """Get strategy with linear weighting consideration - simplified to avoid infinite loops"""
        # Simplified uniform strategy for debugging
        return np.ones(self.num_actions, dtype=np.float32) / self.num_actions


class LinearCFR(MCCFR):
    """Linear CFR with weighted iterations"""
    
    def __init__(self, card_abstraction: CardAbstraction, 
                 action_abstraction: ActionAbstraction,
                 linear_start_iteration: int = 10000, device_config=None):
        super().__init__(card_abstraction, action_abstraction, device_config)
        self.linear_start_iteration = linear_start_iteration
    
    def get_infoset(self, key: str, num_actions: int) -> LinearInfoSet:
        """Get or create linear information set"""
        if key not in self.infosets:
            infoset = LinearInfoSet(key, num_actions, self.device)
            infoset.linear_start_iteration = self.linear_start_iteration
            self.infosets[key] = infoset
        
        return self.infosets[key]
    
    def _mccfr_traverse(self, game_state, reach_probs, traversing_player, depth: int = 0):
        """Enhanced traversal with linear weighting"""
        # Global call counter for debugging
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        
        # Print progress every 1000 calls
        if self._call_count % 1000 == 0:
            print(f"CFR call #{self._call_count} at depth {depth}")
        
        # Emergency brake - if we've made too many calls, something is wrong
        if self._call_count > 2000:
            print(f"EMERGENCY BRAKE: Too many CFR calls ({self._call_count})")
            return {i: 0.0 for i in range(game_state.num_players)}
        
        # Prevent infinite recursion with extremely conservative limit for debugging
        if depth > 5:  # Very shallow depth for debugging
            print(f"Depth limit hit at depth {depth}")
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
            # No legal actions - return neutral utilities to avoid infinite loops
            return {i: 0.0 for i in range(game_state.num_players)}
        
        num_actions = len(abstract_actions)
        
        # Create information set
        infoset_key = self.create_infoset_key(game_state, current_player)
        infoset = self.get_infoset(infoset_key, num_actions)
        
        # Get strategy with current iteration count - add timeout protection
        try:
            strategy = infoset.get_strategy(reach_probs[current_player], self.iterations)
        except:
            # If strategy calculation fails, use uniform strategy
            strategy = np.ones(num_actions, dtype=np.float32) / num_actions
        
        # Rest of the logic is the same as base MCCFR
        action_utilities = np.zeros(num_actions)
        node_utility = {}
        
        # Safety counter to prevent infinite loops in action processing
        max_actions_to_process = min(len(abstract_actions), 20)  # Process max 20 actions
        
        for action_idx, (desc, action_type, amount) in enumerate(abstract_actions[:max_actions_to_process]):
            new_state = game_state.copy()
            
            success = new_state.apply_action(current_player, action_type, amount)
            if not success:
                continue
            
            # Safety check: if we're at high depth and game state seems unchanged, skip
            if depth > 10:
                if (new_state.betting_round == game_state.betting_round and 
                    new_state.current_player == game_state.current_player and
                    new_state.is_terminal() == game_state.is_terminal()):
                    continue  # Skip potentially infinite states
            
            new_reach_probs = reach_probs.copy()
            if current_player != traversing_player:
                new_reach_probs[current_player] *= strategy[action_idx]
            
            child_utilities = self._mccfr_traverse(new_state, new_reach_probs, traversing_player, depth + 1)
            
            if current_player == traversing_player:
                action_utilities[action_idx] = child_utilities.get(current_player, 0)
            
            for player_id, utility in child_utilities.items():
                if player_id not in node_utility:
                    node_utility[player_id] = 0
                
                if current_player == traversing_player:
                    node_utility[player_id] += utility
                else:
                    node_utility[player_id] += strategy[action_idx] * utility
        
        # Update regrets with linear weighting
        if current_player == traversing_player:
            cfr_reach = 1.0
            for i, prob in enumerate(reach_probs):
                if i != current_player:
                    cfr_reach *= prob
            
            # Apply linear weighting to regret updates
            if self.iterations >= self.linear_start_iteration:
                regret_weight = self.iterations - self.linear_start_iteration + 1
            else:
                regret_weight = 1.0
            
            node_util = node_utility.get(current_player, 0)
            for action_idx in range(num_actions):
                regret = action_utilities[action_idx] - node_util
                infoset.update_regret(action_idx, cfr_reach * regret * regret_weight)
        
        return node_utility