"""
Blueprint strategy generator using Linear MCCFR.
Runs self-play training to create a baseline strategy.
"""

import yaml
import random
from typing import Dict, List
from tqdm import tqdm
from ..engine.game_state import GameState, BettingRound
from ..engine.hand_evaluator import Card, Rank, Suit, create_card
from ..abstraction.card_abstraction import CardAbstraction
from ..abstraction.action_abstraction import ActionAbstraction
from ..cfr.linear_cfr import LinearCFR


class BlueprintGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize abstractions
        self.card_abstraction = CardAbstraction(self.config['abstraction'])
        self.action_abstraction = ActionAbstraction(
            self.config['abstraction']['action_fractions']
        )
        
        # Initialize CFR solver
        self.cfr_solver = LinearCFR(self.card_abstraction, self.action_abstraction)
        
        # Game configuration
        self.game_config = self.config['game']
        self.training_config = self.config['training']
        
    def setup_game(self, num_players: int = 6) -> GameState:
        """Create a new game with random cards dealt"""
        game_state = GameState(
            num_players=num_players,
            starting_stack=self.game_config['starting_stack'],
            small_blind=self.game_config['small_blind'],
            big_blind=self.game_config['big_blind']
        )
        
        # Deal random cards
        deck = self._create_shuffled_deck()
        
        # Deal hole cards
        card_idx = 0
        for player in game_state.players:
            player.hole_cards = [deck[card_idx], deck[card_idx + 1]]
            card_idx += 2
        
        # Set community cards based on betting round
        if game_state.betting_round == BettingRound.PREFLOP:
            game_state.community_cards = []
        elif game_state.betting_round == BettingRound.FLOP:
            game_state.community_cards = deck[card_idx:card_idx + 3]
        elif game_state.betting_round == BettingRound.TURN:
            game_state.community_cards = deck[card_idx:card_idx + 4]
        elif game_state.betting_round == BettingRound.RIVER:
            game_state.community_cards = deck[card_idx:card_idx + 5]
        
        return game_state
    
    def _create_shuffled_deck(self) -> List[Card]:
        """Create and shuffle a deck of cards"""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        
        random.shuffle(deck)
        return deck
    
    def train_blueprint(self, iterations: int = None, 
                       checkpoint_frequency: int = None) -> Dict:
        """
        Train blueprint strategy using self-play.
        
        Args:
            iterations: Number of training iterations (default from config)
            checkpoint_frequency: How often to save checkpoints (default from config)
            
        Returns:
            Training statistics
        """
        if iterations is None:
            iterations = self.training_config['iterations']
        
        if checkpoint_frequency is None:
            checkpoint_frequency = self.training_config['checkpoint_frequency']
        
        print(f"Training blueprint for {iterations} iterations...")
        
        # Training statistics
        stats = {
            'iterations': [],
            'exploitability': [],
            'total_infosets': [],
            'avg_utility': []
        }
        
        # Training loop
        for iteration in tqdm(range(iterations), desc="Training"):
            # Sample random game situation
            num_players = random.randint(
                self.game_config['min_players'], 
                self.game_config['max_players']
            )
            
            game_state = self.setup_game(num_players)
            
            # Run CFR iteration
            utilities = self.cfr_solver.train_iteration(game_state)
            
            # Collect statistics
            if iteration % 1000 == 0:
                exploitability = self.cfr_solver.get_exploitability()
                avg_utility = sum(utilities.values()) / len(utilities)
                
                stats['iterations'].append(iteration)
                stats['exploitability'].append(exploitability)
                stats['total_infosets'].append(len(self.cfr_solver.infosets))
                stats['avg_utility'].append(avg_utility)
                
                if iteration % 10000 == 0:
                    print(f"Iteration {iteration}: "
                          f"Exploitability={exploitability:.6f}, "
                          f"InfoSets={len(self.cfr_solver.infosets)}, "
                          f"AvgUtility={avg_utility:.2f}")
            
            # Save checkpoint
            if iteration % checkpoint_frequency == 0 and iteration > 0:
                checkpoint_path = f"data/blueprints/checkpoint_{iteration}.pkl"
                self.cfr_solver.save_strategy(checkpoint_path)
                print(f"Saved checkpoint at iteration {iteration}")
        
        print("Blueprint training complete!")
        return stats
    
    def save_blueprint(self, filepath: str):
        """Save final blueprint strategy"""
        self.cfr_solver.save_strategy(filepath)
        print(f"Blueprint saved to {filepath}")
    
    def evaluate_blueprint(self, num_hands: int = 1000) -> Dict:
        """
        Evaluate blueprint strategy by playing random hands.
        """
        print(f"Evaluating blueprint over {num_hands} hands...")
        
        total_utility = 0
        hands_played = 0
        
        for _ in tqdm(range(num_hands), desc="Evaluating"):
            game_state = self.setup_game()
            
            # Play hand using blueprint strategy
            while not game_state.is_terminal():
                current_player = game_state.current_player
                
                # Get strategy from blueprint
                infoset_key = self.cfr_solver.create_infoset_key(game_state, current_player)
                strategy = self.cfr_solver.get_strategy(infoset_key)
                
                # Get available actions
                abstract_actions = self.action_abstraction.get_abstract_actions(
                    game_state, current_player
                )
                
                if not abstract_actions:
                    break
                
                # Sample action from strategy
                if strategy is not None and len(strategy) == len(abstract_actions):
                    action_idx = random.choices(range(len(abstract_actions)), 
                                               weights=strategy)[0]
                else:
                    action_idx = random.randint(0, len(abstract_actions) - 1)
                
                desc, action_type, amount = abstract_actions[action_idx]
                
                # Apply action
                success = game_state.apply_action(current_player, action_type, amount)
                if not success:
                    break
            
            # Collect results
            payoffs = game_state.get_payoffs()
            total_utility += sum(payoffs)
            hands_played += 1
        
        avg_utility = total_utility / hands_played if hands_played > 0 else 0
        
        evaluation_results = {
            'hands_played': hands_played,
            'average_utility': avg_utility,
            'total_infosets': len(self.cfr_solver.infosets)
        }
        
        print(f"Evaluation complete: {evaluation_results}")
        return evaluation_results