"""
Batch Monte Carlo CFR for GPU acceleration.
Processes multiple game situations simultaneously to maximize GPU utilization.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import concurrent.futures
import threading

from cfr.mccfr import MCCFR, InfoSet
from engine.game_state import GameState, Action, BettingRound
from abstraction.card_abstraction import CardAbstraction
from abstraction.action_abstraction import ActionAbstraction
from utils.device_config import get_device_config


class BatchInfoSet(InfoSet):
    """InfoSet optimized for batch operations"""
    
    def __init__(self, key: str, num_actions: int, device_config=None):
        super().__init__(key, num_actions, device_config)
        
        # Pre-allocate batch computation arrays
        self.batch_size = 1024  # Process 1024 scenarios at once
        self.batch_regrets = self.device.zeros((self.batch_size, num_actions), dtype=np.float32)
        self.batch_strategies = self.device.zeros((self.batch_size, num_actions), dtype=np.float32)
        
    def batch_get_strategy(self, reach_probs_batch):
        """Get strategies for a batch of reach probabilities"""
        batch_size = len(reach_probs_batch)
        
        # Expand regret_sum for batch processing
        regret_batch = self.device.array([self.regret_sum] * batch_size)
        positive_regrets = self.device.maximum(regret_batch, 0)
        
        # Normalize each row
        normalizing_sums = self.device.sum(positive_regrets, axis=1)
        
        # Handle zero normalization
        uniform_strategy = self.device.ones((batch_size, self.num_actions), dtype=np.float32) / self.num_actions
        
        # Create mask for valid normalizations
        valid_mask = normalizing_sums > 0
        
        strategies = uniform_strategy.copy()
        if self.device.sum(valid_mask) > 0:
            # Normalize valid strategies
            valid_indices = self.device.to_numpy(valid_mask).nonzero()[0]
            for idx in valid_indices:
                strategies[idx] = positive_regrets[idx] / normalizing_sums[idx]
        
        # Update strategy sum (weighted by reach probabilities)
        reach_array = self.device.array(reach_probs_batch).reshape(-1, 1)
        weighted_strategies = strategies * reach_array
        strategy_update = self.device.sum(weighted_strategies, axis=0)
        self.strategy_sum += strategy_update
        self.reach_count += len(reach_probs_batch)
        
        return self.device.to_numpy(strategies)
    
    def batch_update_regrets(self, action_indices, regret_values):
        """Update regrets for multiple actions at once"""
        if len(action_indices) == 0:
            return
        
        # Convert to device arrays
        action_array = self.device.array(action_indices)
        regret_array = self.device.array(regret_values)
        
        # Batch update using advanced indexing
        for i, (action_idx, regret) in enumerate(zip(action_indices, regret_values)):
            self.regret_sum[action_idx] += regret


class BatchMCCFR(MCCFR):
    """Batch Monte Carlo CFR for parallel GPU processing"""
    
    def __init__(self, card_abstraction: CardAbstraction, 
                 action_abstraction: ActionAbstraction, 
                 device_config=None, batch_size: int = 256):
        super().__init__(card_abstraction, action_abstraction, device_config)
        
        self.batch_size = batch_size
        self.parallel_workers = 4  # Number of CPU threads for game generation
        
        # Pre-allocated memory pools
        self._initialize_memory_pools()
    
    def _initialize_memory_pools(self):
        """Pre-allocate GPU memory for batch operations"""
        max_actions = 10  # Maximum actions per decision point
        max_depth = 50    # Maximum game tree depth
        
        # Pre-allocate arrays for batch processing
        self.batch_utilities = self.device.zeros((self.batch_size, max_depth), dtype=np.float32)
        self.batch_strategies = self.device.zeros((self.batch_size, max_actions), dtype=np.float32)
        self.batch_regrets = self.device.zeros((self.batch_size, max_actions), dtype=np.float32)
        
        print(f"Initialized batch processing with size {self.batch_size}")
        if self.device.use_gpu:
            used, total = self.device.get_memory_info()
            print(f"GPU memory after initialization: {used / 1e9:.1f}GB / {total / 1e9:.1f}GB")
    
    def get_infoset(self, key: str, num_actions: int) -> BatchInfoSet:
        """Get or create batch-optimized information set"""
        if key not in self.infosets:
            self.infosets[key] = BatchInfoSet(key, num_actions, self.device)
        
        return self.infosets[key]
    
    def batch_train_iteration(self, num_iterations: int = None) -> Dict[int, float]:
        """
        Run multiple CFR iterations in parallel batches.
        """
        if num_iterations is None:
            num_iterations = self.batch_size
        
        # Generate batch of game states in parallel
        game_states = self._generate_game_states_parallel(num_iterations)
        
        # Process batches
        total_utilities = defaultdict(float)
        
        for batch_start in range(0, len(game_states), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(game_states))
            batch_games = game_states[batch_start:batch_end]
            
            batch_utilities = self._process_game_batch(batch_games)
            
            # Accumulate utilities
            for player_utilities in batch_utilities:
                for player_id, utility in player_utilities.items():
                    total_utilities[player_id] += utility
        
        # Average utilities
        num_games = len(game_states)
        if num_games > 0:
            for player_id in total_utilities:
                total_utilities[player_id] /= num_games
        
        self.iterations += num_iterations
        return dict(total_utilities)
    
    def _generate_game_states_parallel(self, num_games: int) -> List[GameState]:
        """Generate multiple game states in parallel"""
        games_per_worker = max(1, num_games // self.parallel_workers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []
            
            for _ in range(self.parallel_workers):
                future = executor.submit(self._generate_game_states_worker, games_per_worker)
                futures.append(future)
            
            # Collect results
            all_games = []
            for future in concurrent.futures.as_completed(futures):
                games = future.result()
                all_games.extend(games)
        
        return all_games[:num_games]  # Trim to exact number needed
    
    def _generate_game_states_worker(self, num_games: int) -> List[GameState]:
        """Worker function to generate game states"""
        games = []
        
        for _ in range(num_games):
            # Create random game
            num_players = random.randint(2, 6)
            game_state = self._create_random_game_state(num_players)
            games.append(game_state)
        
        return games
    
    def _create_random_game_state(self, num_players: int) -> GameState:
        """Create a random game state for training"""
        game_state = GameState(
            num_players=num_players,
            starting_stack=10000,
            small_blind=50,
            big_blind=100
        )
        
        # Deal random cards efficiently
        deck = self._create_shuffled_deck()
        
        # Deal hole cards
        card_idx = 0
        for player in game_state.players:
            player.hole_cards = [deck[card_idx], deck[card_idx + 1]]
            card_idx += 2
        
        # Random betting round
        betting_round = random.choice(list(BettingRound))
        game_state.betting_round = betting_round
        
        # Set community cards
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
    
    def _process_game_batch(self, game_batch: List[GameState]) -> List[Dict[int, float]]:
        """Process a batch of games simultaneously"""
        batch_results = []
        
        # Group games by similarity for more efficient processing
        grouped_games = self._group_similar_games(game_batch)
        
        for game_group in grouped_games:
            # Process similar games together for better GPU utilization
            group_results = self._process_similar_games(game_group)
            batch_results.extend(group_results)
        
        return batch_results
    
    def _group_similar_games(self, games: List[GameState]) -> List[List[GameState]]:
        """Group games by similar characteristics for efficient batch processing"""
        # Simple grouping by number of players and betting round
        groups = defaultdict(list)
        
        for game in games:
            key = (game.num_players, game.betting_round)
            groups[key].append(game)
        
        return list(groups.values())
    
    def _process_similar_games(self, games: List[GameState]) -> List[Dict[int, float]]:
        """Process a group of similar games efficiently"""
        results = []
        
        for game in games:
            # Sample traversing player
            traversing_player = random.randint(0, game.num_players - 1)
            reach_probs = [1.0] * game.num_players
            
            # Run MCCFR traversal (using parent class method for now)
            utilities = self._mccfr_traverse(game, reach_probs, traversing_player)
            results.append(utilities)
        
        return results
    
    def batch_memory_cleanup(self):
        """Clean up GPU memory periodically"""
        if self.device.use_gpu:
            self.device.clear_cache()
            
            # Optionally defragment information sets
            if len(self.infosets) > 100000:  # If too many infosets
                self._defragment_infosets()
    
    def _defragment_infosets(self):
        """Remove rarely used information sets to free memory"""
        # Remove infosets with very low reach counts
        min_reach_count = max(1, self.iterations // 10000)
        
        keys_to_remove = []
        for key, infoset in self.infosets.items():
            if infoset.reach_count < min_reach_count:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.infosets[key]
        
        if keys_to_remove:
            print(f"Removed {len(keys_to_remove)} rarely used information sets")
    
    def get_memory_usage_stats(self) -> Dict:
        """Get detailed memory usage statistics"""
        stats = {
            'total_infosets': len(self.infosets),
            'batch_size': self.batch_size,
            'parallel_workers': self.parallel_workers
        }
        
        if self.device.use_gpu:
            used, total = self.device.get_memory_info()
            stats.update({
                'gpu_memory_used_gb': used / 1e9,
                'gpu_memory_total_gb': total / 1e9,
                'gpu_utilization': (used / total) * 100
            })
        
        # Estimate memory per infoset
        if self.infosets:
            sample_infoset = next(iter(self.infosets.values()))
            # Rough estimate: regret_sum + strategy_sum + metadata
            memory_per_infoset = sample_infoset.num_actions * 2 * 4  # 2 arrays * 4 bytes per float32
            stats['estimated_infoset_memory_mb'] = (len(self.infosets) * memory_per_infoset) / 1e6
        
        return stats