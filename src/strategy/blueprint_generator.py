"""
Blueprint strategy generator using Linear MCCFR.
Runs self-play training to create a baseline strategy.
"""

import yaml
import random
import os
from typing import Dict, List
from tqdm import tqdm
from engine.game_state import GameState, BettingRound
from engine.hand_evaluator import Card, Rank, Suit, create_card
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from cfr.linear_cfr import LinearCFR
from cfr.batch_mccfr import BatchMCCFR
from utils.device_config import setup_device


class BlueprintGenerator:
    def __init__(self, config_path: str, use_gpu: bool = True, device_id: int = 0, 
                 use_batch_processing: bool = True, batch_size: int = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device configuration
        self.device_config = setup_device(force_cpu=not use_gpu, device_id=device_id)
        
        # Determine batch size based on GPU memory and CPU cores
        if batch_size is None:
            if self.device_config.use_gpu:
                _, total_mem = self.device_config.get_memory_info()
                import multiprocessing
                cpu_cores = multiprocessing.cpu_count()
                
                # Scale batch size based on both GPU memory and CPU cores
                if total_mem > 80e9 and cpu_cores >= 32:  # GH200-class with many CPUs
                    batch_size = 8192  # Massive batches for supercomputing setups
                elif total_mem > 30e9 and cpu_cores >= 16:  # A100-class with many CPUs
                    batch_size = 4096
                elif total_mem > 30e9:  # High-end GPU
                    batch_size = 1024
                elif total_mem > 10e9:  # Mid-range GPU
                    batch_size = 512
                else:
                    batch_size = 256
            else:
                batch_size = 128  # CPU only
        
        self.use_batch_processing = use_batch_processing and self.device_config.use_gpu
        self.batch_size = batch_size
        
        print(f"Using {'GPU' if self.device_config.use_gpu else 'CPU'} for computation "
              f"(backend: {self.device_config.backend})")
        
        if self.use_batch_processing:
            print(f"Batch processing enabled with batch size: {batch_size}")
        
        # Initialize abstractions with device config
        self.card_abstraction = CardAbstraction(
            self.config['abstraction'], 
            device_config=self.device_config
        )
        self.action_abstraction = ActionAbstraction(
            self.config['abstraction']['action_fractions']
        )
        
        # Initialize CFR solver with device config
        if self.use_batch_processing:
            self.cfr_solver = BatchMCCFR(
                self.card_abstraction, 
                self.action_abstraction,
                device_config=self.device_config,
                batch_size=batch_size
            )
        else:
            self.cfr_solver = LinearCFR(
                self.card_abstraction, 
                self.action_abstraction,
                device_config=self.device_config
            )
        
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
        if self.use_batch_processing:
            # Batch training for GPU acceleration
            batch_iterations = max(1, iterations // self.batch_size)
            
            for batch_idx in tqdm(range(batch_iterations), desc="Batch Training"):
                # Run batch of iterations
                utilities = self.cfr_solver.batch_train_iteration(self.batch_size)
                
                # Update iteration counter
                iteration = (batch_idx + 1) * self.batch_size
                
                # Collect statistics every few batches
                if batch_idx % max(1, 1000 // self.batch_size) == 0:
                    self._collect_training_stats(iteration, utilities, stats)
                
                # Save checkpoint
                if iteration % checkpoint_frequency == 0 and iteration > 0:
                    checkpoint_path = f"data/blueprints/checkpoint_{iteration}.pkl"
                    self.cfr_solver.save_strategy(checkpoint_path)
                    print(f"Saved checkpoint at iteration {iteration}")
                
                # Memory cleanup for large batches
                if batch_idx % 10 == 0 and hasattr(self.cfr_solver, 'batch_memory_cleanup'):
                    self.cfr_solver.batch_memory_cleanup()
        else:
            # Standard single-iteration training
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
                    self._collect_training_stats(iteration, utilities, stats)
                
                # Save checkpoint
                if iteration % checkpoint_frequency == 0 and iteration > 0:
                    checkpoint_path = f"data/blueprints/checkpoint_{iteration}.pkl"
                    self.cfr_solver.save_strategy(checkpoint_path)
                    print(f"Saved checkpoint at iteration {iteration}")
    
    def _collect_training_stats(self, iteration: int, utilities: Dict, stats: Dict):
        """Collect and display training statistics"""
        exploitability = self.cfr_solver.get_exploitability()
        avg_utility = sum(utilities.values()) / len(utilities) if utilities else 0.0
        
        stats['iterations'].append(iteration)
        stats['exploitability'].append(exploitability)
        stats['total_infosets'].append(len(self.cfr_solver.infosets))
        stats['avg_utility'].append(avg_utility)
        
        if iteration % 10000 == 0:
            # Get memory info if using GPU
            if self.device_config.use_gpu:
                used_mem, total_mem = self.device_config.get_memory_info()
                mem_usage = f"GPU Memory: {used_mem / 1e9:.1f}GB / {total_mem / 1e9:.1f}GB"
                
                # Additional memory stats for batch processing
                if hasattr(self.cfr_solver, 'get_memory_usage_stats'):
                    mem_stats = self.cfr_solver.get_memory_usage_stats()
                    utilization = mem_stats.get('gpu_utilization', 0)
                    mem_usage += f" ({utilization:.1f}% util)"
            else:
                mem_usage = "CPU Mode"
            
            print(f"Iteration {iteration}: "
                  f"Exploitability={exploitability:.6f}, "
                  f"InfoSets={len(self.cfr_solver.infosets)}, "
                  f"AvgUtility={avg_utility:.2f}, "
                  f"{mem_usage}")
            
            # Additional quality metrics
            if iteration > 0:
                # Strategy stability (Nash distance from previous checkpoint)
                if iteration >= 20000:  # Have previous checkpoint to compare
                    prev_checkpoint = f"data/blueprints/checkpoint_{iteration-10000}.pkl"
                    if os.path.exists(prev_checkpoint):
                        try:
                            from evaluation.metrics import PokerMetrics
                            from cfr.linear_cfr import LinearCFR
                            
                            # Load previous strategy
                            prev_cfr = LinearCFR(self.card_abstraction, self.action_abstraction)
                            prev_cfr.load_strategy(prev_checkpoint)
                            
                            # Compute Nash distance
                            nash_dist = PokerMetrics.nash_distance(self.cfr_solver, prev_cfr)
                            print(f"         Nash distance from prev: {nash_dist:.6f}")
                            
                            # Store in stats
                            if 'nash_distance' not in stats:
                                stats['nash_distance'] = []
                            stats['nash_distance'].append(nash_dist)
                            
                        except Exception as e:
                            print(f"         Could not compute Nash distance: {e}")
            
            # Synchronize GPU operations for accurate timing
            if self.device_config.use_gpu:
                self.device_config.synchronize()
        
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