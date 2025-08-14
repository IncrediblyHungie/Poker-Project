"""
Action handler for processing and validating poker actions.
Handles fold, check, call, raise, and all-in actions according to poker rules.
"""

from typing import List, Tuple, Optional
from .game_state import GameState, Action, Player


class ActionHandler:
    def __init__(self, game_state: GameState):
        self.game_state = game_state
    
    def validate_action(self, player_id: int, action: Action, amount: int = 0) -> bool:
        """
        Validate if an action is legal for the given player.
        
        Args:
            player_id: ID of the player taking the action
            action: The action being taken
            amount: Bet/raise amount (if applicable)
            
        Returns:
            True if action is legal, False otherwise
        """
        if self.game_state.is_terminal():
            return False
        
        if player_id != self.game_state.current_player:
            return False
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return False
        
        legal_actions = self.game_state.get_legal_actions(player_id)
        
        # Check if this specific action and amount combination is legal
        for legal_action, legal_amount in legal_actions:
            if action == legal_action:
                if action in [Action.FOLD, Action.CHECK]:
                    return amount == 0
                elif action == Action.CALL:
                    return amount == legal_amount
                elif action == Action.RAISE:
                    return amount >= legal_amount
                elif action == Action.ALL_IN:
                    return amount == player.stack
        
        return False
    
    def process_action(self, player_id: int, action: Action, amount: int = 0) -> bool:
        """
        Process and apply a player action to the game state.
        
        Args:
            player_id: ID of the player taking the action
            action: The action being taken
            amount: Bet/raise amount (if applicable)
            
        Returns:
            True if action was successfully processed, False otherwise
        """
        if not self.validate_action(player_id, action, amount):
            return False
        
        return self.game_state.apply_action(player_id, action, amount)
    
    def get_call_amount(self, player_id: int) -> int:
        """Get the amount needed for a player to call"""
        if self.game_state.is_terminal():
            return 0
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return 0
        
        call_amount = self.game_state.current_bet - player.bet_this_round
        return min(call_amount, player.stack)
    
    def get_min_raise_amount(self, player_id: int) -> int:
        """Get the minimum raise amount for a player"""
        if self.game_state.is_terminal():
            return 0
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return 0
        
        call_amount = self.get_call_amount(player_id)
        remaining_stack = player.stack - call_amount
        
        if remaining_stack <= 0:
            return 0
        
        min_raise = max(self.game_state.min_raise, self.game_state.big_blind)
        return min(min_raise, remaining_stack)
    
    def get_max_raise_amount(self, player_id: int) -> int:
        """Get the maximum raise amount for a player (all-in)"""
        if self.game_state.is_terminal():
            return 0
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return 0
        
        return player.stack
    
    def get_pot_odds(self, player_id: int) -> float:
        """
        Calculate pot odds for a player facing a bet.
        Returns the ratio of pot size to call amount.
        """
        call_amount = self.get_call_amount(player_id)
        if call_amount == 0:
            return float('inf')  # No cost to see next card
        
        total_pot = self.game_state.pot + call_amount
        return total_pot / call_amount
    
    def get_effective_stack_size(self, player_id: int) -> int:
        """
        Get effective stack size (smallest stack among active players).
        This is important for determining maximum possible action sizes.
        """
        active_players = self.game_state.get_active_players()
        if not active_players:
            return 0
        
        stack_sizes = [p.stack for p in active_players if p.player_id != player_id]
        if not stack_sizes:
            return self.game_state.players[player_id].stack
        
        return min(min(stack_sizes), self.game_state.players[player_id].stack)
    
    def calculate_bet_sizing(self, player_id: int, pot_fraction: float) -> int:
        """
        Calculate bet size based on pot fraction.
        
        Args:
            player_id: ID of the player
            pot_fraction: Fraction of pot to bet (e.g., 0.5 for half pot)
            
        Returns:
            Bet amount in chips
        """
        if self.game_state.is_terminal():
            return 0
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return 0
        
        current_pot = self.game_state.pot
        call_amount = self.get_call_amount(player_id)
        
        # Calculate total pot after calling
        total_pot_after_call = current_pot + call_amount
        
        # Calculate raise amount based on pot fraction
        raise_amount = int(total_pot_after_call * pot_fraction)
        
        # Ensure minimum raise requirements
        min_raise = self.get_min_raise_amount(player_id)
        if raise_amount < min_raise:
            raise_amount = min_raise
        
        # Ensure we don't exceed stack
        max_raise = self.get_max_raise_amount(player_id)
        if call_amount + raise_amount > max_raise:
            raise_amount = max_raise - call_amount
        
        return max(0, call_amount + raise_amount)
    
    def get_available_bet_sizes(self, player_id: int, 
                               pot_fractions: List[float]) -> List[Tuple[str, int]]:
        """
        Get available bet sizes based on pot fractions.
        
        Args:
            player_id: ID of the player
            pot_fractions: List of pot fractions to calculate
            
        Returns:
            List of (description, amount) tuples
        """
        if self.game_state.is_terminal():
            return []
        
        player = self.game_state.players[player_id]
        if not player.can_bet():
            return []
        
        bet_sizes = []
        call_amount = self.get_call_amount(player_id)
        min_raise = self.get_min_raise_amount(player_id)
        max_raise = self.get_max_raise_amount(player_id)
        
        # Add call option if there's a bet to call
        if call_amount > 0:
            bet_sizes.append(("Call", call_amount))
        else:
            bet_sizes.append(("Check", 0))
        
        # Add raise options based on pot fractions
        for fraction in pot_fractions:
            bet_amount = self.calculate_bet_sizing(player_id, fraction)
            
            if bet_amount >= call_amount + min_raise and bet_amount <= max_raise:
                if fraction <= 1.0:
                    description = f"{int(fraction * 100)}% pot"
                else:
                    description = f"{fraction}x pot"
                bet_sizes.append((description, bet_amount))
        
        # Always add all-in option if we have chips
        if player.stack > 0 and max_raise > call_amount:
            bet_sizes.append(("All-in", max_raise))
        
        return bet_sizes
    
    def is_facing_bet(self, player_id: int) -> bool:
        """Check if player is facing a bet (needs to call to continue)"""
        return self.get_call_amount(player_id) > 0
    
    def is_heads_up(self) -> bool:
        """Check if action is heads up (only 2 active players)"""
        active_players = self.game_state.get_active_players()
        return len(active_players) == 2
    
    def get_position_type(self, player_id: int) -> str:
        """
        Get position type for the player.
        Returns position description like 'button', 'small_blind', etc.
        """
        active_players = self.game_state.get_active_players()
        num_active = len(active_players)
        
        if num_active < 2:
            return "unknown"
        
        # Find player's position relative to dealer
        dealer_pos = self.game_state.dealer_position
        
        # Calculate position
        if num_active == 2:
            if player_id == dealer_pos:
                return "small_blind"  # In heads-up, dealer is small blind
            else:
                return "big_blind"
        
        # Multi-way pot positions
        if player_id == dealer_pos:
            return "button"
        elif player_id == (dealer_pos + 1) % self.game_state.num_players:
            return "small_blind"
        elif player_id == (dealer_pos + 2) % self.game_state.num_players:
            return "big_blind"
        elif player_id == (dealer_pos + 3) % self.game_state.num_players:
            return "under_the_gun"
        else:
            # Calculate if early, middle, or late position
            position_offset = (player_id - dealer_pos) % self.game_state.num_players
            if position_offset <= num_active // 3:
                return "early_position"
            elif position_offset <= 2 * num_active // 3:
                return "middle_position"
            else:
                return "late_position"
    
    def get_action_history_summary(self) -> dict:
        """
        Get summary of actions taken in current betting round.
        Useful for strategy decisions.
        """
        current_round = self.game_state.betting_round
        actions = self.game_state.round_betting_history[current_round]
        
        summary = {
            'num_raises': 0,
            'num_calls': 0,
            'num_folds': 0,
            'num_checks': 0,
            'aggressive_players': set(),
            'total_action_count': len(actions)
        }
        
        for player_id, action, amount in actions:
            if action == Action.RAISE or action == Action.ALL_IN:
                summary['num_raises'] += 1
                summary['aggressive_players'].add(player_id)
            elif action == Action.CALL:
                summary['num_calls'] += 1
            elif action == Action.FOLD:
                summary['num_folds'] += 1
            elif action == Action.CHECK:
                summary['num_checks'] += 1
        
        return summary