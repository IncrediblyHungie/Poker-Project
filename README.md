# Pluribus-Quality Poker Bot

A comprehensive implementation of a poker AI bot using the same techniques as Facebook's Pluribus: Monte Carlo Counterfactual Regret Minimization (MCCFR), abstraction, and real-time search.

## Features

- **Game Engine**: Full Texas Hold'em implementation supporting 2-10 players
- **Card Abstraction**: K-means clustering with Earth Mover's Distance for grouping similar hands
- **Action Abstraction**: Discrete betting actions based on pot fractions
- **Monte Carlo CFR**: Advanced CFR implementation with linear weighting
- **Blueprint Strategy**: Self-play training to generate baseline strategy
- **Real-time Search**: Depth-limited subgame solving for online play
- **Modular Design**: Clean, extensible architecture

## Project Structure

```
pluribus_poker_bot/
├── src/
│   ├── engine/           # Game engine and rules
│   │   ├── game_state.py
│   │   ├── hand_evaluator.py
│   │   └── action_handler.py
│   ├── abstraction/      # Card and action abstraction
│   │   ├── card_abstraction.py
│   │   └── action_abstraction.py
│   ├── cfr/             # CFR algorithms
│   │   ├── mccfr.py
│   │   └── linear_cfr.py
│   ├── strategy/        # Strategy computation
│   │   ├── blueprint_generator.py
│   │   └── blueprint_strategy.py
│   └── bot/             # Main bot implementation
│       ├── pluribus_bot.py
│       └── player_manager.py
├── config/
│   └── game_config.yaml
├── scripts/
│   ├── train.py         # Training script
│   └── play.py          # Testing script
├── tests/               # Unit tests
├── data/                # Training data and models
└── requirements.txt
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pluribus_poker_bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Train the Bot

Train card abstractions and blueprint strategy:

```bash
python scripts/train.py --config config/game_config.yaml
```

Options:
- `--iterations N`: Number of training iterations (default: 1,000,000)
- `--skip-abstractions`: Skip card abstraction training
- `--abstractions-only`: Only train abstractions

### 2. Test the Bot

Play bot vs bot games:

```bash
python scripts/play.py --mode bot_vs_bot --hands 100
```

Play against the bot (basic interface):

```bash
python scripts/play.py --mode human_vs_bot
```

## Configuration

Edit `config/game_config.yaml` to customize:

```yaml
game:
  min_players: 2
  max_players: 10
  starting_stack: 10000
  small_blind: 50
  big_blind: 100

abstraction:
  card_buckets:
    preflop: 169      # All canonical hands
    flop: 5000        # Flop abstractions
    turn: 2000        # Turn abstractions  
    river: 1000       # River abstractions
  action_fractions: [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0]

training:
  iterations: 1000000
  checkpoint_frequency: 10000
  num_threads: 8
```

## Algorithm Details

### Card Abstraction

The bot uses K-means clustering to group similar poker hands:

1. **Equity Distribution**: For each hand-board combination, compute win probability distribution using Monte Carlo simulation
2. **Earth Mover's Distance**: Use Wasserstein distance as similarity metric between hands
3. **K-means Clustering**: Group similar hands into buckets for each betting round
4. **Information Sets**: Use (hole_cards_bucket, board_bucket, betting_history) as keys

### Action Abstraction

Betting actions are abstracted to discrete choices:

- **Fold/Check/Call**: Always available when legal
- **Bet/Raise**: Based on pot fractions (e.g., 25%, 50%, 100%, 200% of pot)
- **All-in**: Always available as an option
- **Mapping**: Continuous bet sizes mapped to nearest abstract action

### Monte Carlo CFR

Core algorithm for strategy computation:

1. **External Sampling**: Sample opponent actions, enumerate traverser actions
2. **Regret Matching**: Compute strategy proportional to positive regrets
3. **Linear Weighting**: Recent iterations weighted more heavily (after 10k iterations)
4. **Information Sets**: Store regrets and strategies for each unique game situation

### Training Process

1. **Initialize**: Set up abstractions and CFR solver
2. **Sample Games**: Generate random poker situations
3. **CFR Iteration**: Run MCCFR traversal for each game
4. **Update Regrets**: Accumulate counterfactual regrets
5. **Strategy Averaging**: Compute time-averaged strategies
6. **Checkpointing**: Save progress periodically

## Performance Targets

- **Blueprint Generation**: 100k iterations in < 1 hour (8 cores)
- **Real-time Search**: < 5 seconds per decision
- **Memory Usage**: < 16GB for small abstraction, < 128GB for full
- **Convergence**: Low exploitability after sufficient training

## Testing Progression

Recommended testing sequence:

1. **Kuhn Poker**: Verify CFR works on simple game
2. **Leduc Hold'em**: Test on medium complexity game  
3. **Limited Hold'em**: Test with betting limits
4. **6-Player No-Limit**: Scale to multi-player
5. **10-Player Full**: Complete implementation

## Implementation Status

### Completed ✅
- [x] Project structure and configuration
- [x] Game engine (GameState, HandEvaluator, ActionHandler)
- [x] Card abstraction with clustering
- [x] Action abstraction system
- [x] Basic MCCFR implementation
- [x] Linear CFR enhancement
- [x] Blueprint generator
- [x] Training and testing scripts
- [x] Basic bot implementation

### In Progress 🚧
- [ ] Real-time search implementation
- [ ] Advanced bot logic with search decisions
- [ ] Player manager for dynamic games
- [ ] Strategy storage optimization
- [ ] Performance optimizations

### Future Enhancements 🔮
- [ ] Exploitability computation
- [ ] Opponent modeling
- [ ] Multi-threading optimization
- [ ] Advanced abstractions
- [ ] Tournament play support

## Usage Examples

### Basic Training

```python
from src.strategy.blueprint_generator import BlueprintGenerator

# Initialize and train
generator = BlueprintGenerator("config/game_config.yaml")
stats = generator.train_blueprint(iterations=100000)
generator.save_blueprint("my_blueprint.pkl")
```

### Custom Game

```python
from src.engine.game_state import GameState
from src.engine.hand_evaluator import create_card

# Create custom game
game = GameState(num_players=6, starting_stack=1000, 
                small_blind=5, big_blind=10)

# Deal specific cards
game.players[0].hole_cards = [
    create_card('A', 'S'), create_card('K', 'S')
]
```

## Contributing

1. Follow existing code style and patterns
2. Add unit tests for new functionality
3. Update documentation for API changes
4. Test with different game configurations

## Performance Notes

- Start with smaller abstractions for faster training
- Use checkpointing for long training runs
- Monitor memory usage with large card abstractions
- Consider distributed training for production use

## References

- [Pluribus Paper](https://science.sciencemag.org/content/365/6456/885)
- [Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- [Monte Carlo CFR](https://papers.nips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)

## License

MIT License - see LICENSE file for details.