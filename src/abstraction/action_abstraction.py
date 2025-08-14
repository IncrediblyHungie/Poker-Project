"""
Action abstraction system for poker.
Maps continuous bet sizes to discrete abstract actions based on pot fractions.
"""

from typing import List, Tuple, Dict
from ..engine.game_state import GameState, Action
from ..engine.action_handler import ActionHandler


class ActionAbstraction:
    def __init__(self, action_fractions: List[float]):
        """
        Initialize action abstraction.
        
        Args:
            action_fractions: List of pot fractions for betting (e.g., [0.25, 0.5, 1.0, 2.0])
        """
        self.action_fractions = sorted(action_fractions)
        
    def get_abstract_actions(self, game_state: GameState, player_id: int) -> List[Tuple[str, Action, int]]:
        """
        Get available abstract actions for a player.
        
        Returns:
            List of (description, action_type, amount) tuples
        """
        if game_state.is_terminal():
            return []
        
        action_handler = ActionHandler(game_state)
        abstract_actions = []
        
        # Always include fold if facing a bet
        if action_handler.is_facing_bet(player_id):
            abstract_actions.append(("Fold", Action.FOLD, 0))
        
        # Check/Call
        call_amount = action_handler.get_call_amount(player_id)
        if call_amount == 0:
            abstract_actions.append(("Check", Action.CHECK, 0))
        else:
            abstract_actions.append(("Call", Action.CALL, call_amount))
        
        # Betting/Raising actions based on pot fractions
        for fraction in self.action_fractions:
            bet_amount = action_handler.calculate_bet_sizing(player_id, fraction)
            min_raise = action_handler.get_min_raise_amount(player_id)
            
            if bet_amount >= call_amount + min_raise:
                if fraction <= 1.0:
                    description = f"Bet {int(fraction * 100)}% pot"
                else:
                    description = f"Bet {fraction}x pot"
                
                abstract_actions.append((description, Action.RAISE, bet_amount))
        
        # Always include all-in if player has chips
        player = game_state.players[player_id]
        if player.stack > 0:
            max_bet = action_handler.get_max_raise_amount(player_id)
            if max_bet > call_amount:
                abstract_actions.append(("All-in", Action.ALL_IN, max_bet))
        
        return abstract_actions
    
    def map_to_abstract_action(self, game_state: GameState, player_id: int, 
                              action: Action, amount: int) -> Tuple[str, Action, int]:
        """
        Map a concrete action to the nearest abstract action.
        """
        abstract_actions = self.get_abstract_actions(game_state, player_id)
        
        if action in [Action.FOLD, Action.CHECK]:
            for desc, abs_action, abs_amount in abstract_actions:
                if abs_action == action:
                    return desc, abs_action, abs_amount
        
        elif action == Action.CALL:
            for desc, abs_action, abs_amount in abstract_actions:
                if abs_action == Action.CALL:
                    return desc, abs_action, abs_amount
        
        elif action in [Action.RAISE, Action.ALL_IN]:
            # Find closest betting action by amount
            betting_actions = [(desc, abs_action, abs_amount) 
                             for desc, abs_action, abs_amount in abstract_actions
                             if abs_action in [Action.RAISE, Action.ALL_IN]]
            
            if not betting_actions:
                return None
            
            # Find closest by amount
            closest_action = min(betting_actions, 
                               key=lambda x: abs(x[2] - amount))
            return closest_action
        
        return None
    
    def get_action_index(self, game_state: GameState, player_id: int, 
                        action_desc: str) -> int:
        """Get index of action in abstract action list"""
        abstract_actions = self.get_abstract_actions(game_state, player_id)
        
        for i, (desc, _, _) in enumerate(abstract_actions):
            if desc == action_desc:
                return i
        
        return 0  # Default to first action
    
    def get_num_actions(self) -> int:
        """Get maximum number of abstract actions"""
        # Fold, Check/Call, Betting fractions, All-in
        return 2 + len(self.action_fractions) + 1