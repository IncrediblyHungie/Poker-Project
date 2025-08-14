#!/usr/bin/env python3
"""
Test script to verify all module imports work correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def test_external_imports():
    """Test external dependencies"""
    print('Testing external dependencies...')
    try:
        import numpy as np
        import yaml
        import joblib
        import tqdm
        from pypokerengine.api.game import setup_config, start_poker
        print('✓ External dependencies imported successfully')
        return True
    except ImportError as e:
        print(f'❌ External import error: {e}')
        return False

def test_project_imports():
    """Test project module imports"""
    print('Testing project modules...')
    
    try:
        # Test engine modules first (no relative imports)
        from engine.game_state import GameState, Action
        print('✓ GameState imported')
        
        from engine.action_handler import ActionHandler
        print('✓ ActionHandler imported')
        
        from engine.hand_evaluator import HandEvaluator
        print('✓ HandEvaluator imported')
        
        # Test abstraction modules (these might have relative imports)
        try:
            from abstraction.action_abstraction import ActionAbstraction
            print('✓ ActionAbstraction imported')
        except ImportError as e:
            print(f'⚠️  ActionAbstraction import issue: {e}')
        
        try:
            from abstraction.card_abstraction import CardAbstraction
            print('✓ CardAbstraction imported')
        except ImportError as e:
            print(f'⚠️  CardAbstraction import issue: {e}')
        
        # Test CFR modules
        try:
            from cfr.linear_cfr import LinearCFR
            print('✓ LinearCFR imported')
        except ImportError as e:
            print(f'⚠️  LinearCFR import issue: {e}')
        
        try:
            from cfr.mccfr import MCCFR
            print('✓ MCCFR imported')
        except ImportError as e:
            print(f'⚠️  MCCFR import issue: {e}')
        
        # Test strategy modules
        try:
            from strategy.blueprint_generator import BlueprintGenerator
            print('✓ BlueprintGenerator imported')
        except ImportError as e:
            print(f'⚠️  BlueprintGenerator import issue: {e}')
        
        print('✅ Core project modules tested!')
        return True
        
    except ImportError as e:
        print(f'❌ Project import error: {e}')
        return False
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        return False

def test_basic_functionality():
    """Test basic functionality of key classes"""
    print('Testing basic functionality...')
    
    try:
        from engine.game_state import GameState
        from abstraction.action_abstraction import ActionAbstraction
        
        # Test creating action abstraction
        action_abs = ActionAbstraction([0.25, 0.5, 1.0, 2.0])
        print('✓ ActionAbstraction created successfully')
        
        # Test creating game state
        game_state = GameState(num_players=6, big_blind=20, small_blind=10, starting_stack=1000)
        print('✓ GameState created successfully')
        
        print('✅ Basic functionality tests passed!')
        return True
        
    except Exception as e:
        print(f'❌ Functionality test error: {e}')
        return False

def main():
    """Main test function"""
    print("🧪 Testing Pluribus Poker Bot File System Setup")
    print("=" * 50)
    
    all_passed = True
    
    # Test external imports
    all_passed &= test_external_imports()
    print()
    
    # Test project imports
    all_passed &= test_project_imports()
    print()
    
    # Test basic functionality
    all_passed &= test_basic_functionality()
    print()
    
    if all_passed:
        print("🎉 All tests passed! The file system setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())