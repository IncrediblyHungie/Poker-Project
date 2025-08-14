#!/usr/bin/env python3
"""
Test blueprint training with the fixed CFR implementation
"""

import sys
sys.path.insert(0, "src")

from strategy.blueprint_generator import BlueprintGenerator
import signal

print("Testing blueprint training...")

def timeout_handler(signum, frame):
    raise TimeoutError("Blueprint training timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60-second timeout

try:
    # Create blueprint generator (using CPU mode to be safe)
    generator = BlueprintGenerator(
        config_path='config/game_config.yaml',
        use_gpu=False,  # Use CPU to avoid any GPU issues
        use_batch_processing=False  # Use SimpleCFR
    )
    
    print("✓ Blueprint generator created")
    
    # Test a small training run
    print("Starting small training run (100 iterations)...")
    stats = generator.train_blueprint(iterations=100, checkpoint_frequency=50)
    
    signal.alarm(0)
    print("✓ Blueprint training completed successfully!")
    print(f"Training stats: {stats}")
    
except TimeoutError:
    print("❌ Blueprint training timed out")
    signal.alarm(0)
except Exception as e:
    print(f"❌ Blueprint training failed: {e}")
    signal.alarm(0)

print("Blueprint training test complete")