#!/usr/bin/env python3
# Grid search script for SpecAugment parameters

import itertools
import argparse
from tune_augment import main_fn

def grid_search(target_wer, margin=0.05):
    """
    Perform grid search over SpecAugment parameters to find a configuration that
    produces a WER within the specified margin of the target.
    
    Args:
        target_wer: Target word error rate to achieve
        margin: Acceptable margin around target WER (default: 0.05)
    
    Returns:
        Best parameters found and their corresponding WER
    """
    # Define parameter ranges to search
    # These are common ranges for SpecAugment parameters
    param_grid = {
        'freq_mask_param': [10, 20, 30, 40, 50],      # Frequency mask maximum width
        'time_mask_param': [10, 20, 40, 80, 100],     # Time mask maximum width
        'n_freq_mask': [1, 2, 3, 4],                  # Number of frequency masks
        'n_time_mask': [1, 2, 4, 8, 10],              # Number of time masks
        'warp_param': [0, 5, 10, 20, 40]              # Time warping parameter (0 for no warping)
    }
    
    # Generate all combinations of parameters
    keys = list(param_grid.keys())
    param_combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    
    print(f"Starting grid search to find WER close to {target_wer} (margin: {margin})")
    print(f"Total combinations to test: {len(param_combinations)}")
    
    best_params = None
    best_wer_diff = float('inf')
    
    # Track all results within margin
    results_within_margin = []
    
    for i, values in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(keys, values))
        
        # Always run with augmentation enabled
        params['augment'] = True
        
        # Run the main function with these parameters
        wer = main_fn(**params)
        
        # Calculate difference from target
        wer_diff = abs(wer - target_wer)
        
        # Print progress update
        print(f"[{i+1}/{len(param_combinations)}] WER: {wer:.4f}, " + 
              f"Params: {', '.join([f'{k}={v}' for k, v in params.items() if k != 'augment'])}")
        
        # Check if this is the best result so far
        if wer_diff < best_wer_diff:
            best_wer_diff = wer_diff
            best_params = params.copy()
            best_params['wer'] = wer
            
        # Store results within margin
        if wer_diff <= margin:
            result = params.copy()
            result['wer'] = wer
            result['wer_diff'] = wer_diff
            results_within_margin.append(result)
            
            # Break if we found a result within margin
            print(f"\nFound WER within margin: {wer:.4f} (target: {target_wer})")
            print(f"Parameters: {', '.join([f'{k}={v}' for k, v in params.items() if k != 'augment'])}")
            break
    
    if not results_within_margin:
        print("\nNo results found within the specified margin.")
        print(f"Best result: WER = {best_params['wer']:.4f} (difference: {best_wer_diff:.4f})")
        print(f"Parameters: {', '.join([f'{k}={v}' for k, v in best_params.items() if k not in ['augment', 'wer']])}")
    else:
        # Print all results within margin
        print(f"\nFound {len(results_within_margin)} results within margin:")
        for i, result in enumerate(sorted(results_within_margin, key=lambda x: x['wer_diff'])):
            print(f"{i+1}. WER = {result['wer']:.4f} (difference: {result['wer_diff']:.4f})")
            print(f"   Parameters: {', '.join([f'{k}={v}' for k, v in result.items() if k not in ['augment', 'wer', 'wer_diff']])}")
    
    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search for SpecAugment parameters')
    parser.add_argument('--target-wer', default=0.0749, type=float,
                        help='Target WER to achieve')
    parser.add_argument('--margin', type=float, default=0.003,
                        help='Acceptable margin around target WER (default: 0.003)')
    
    args = parser.parse_args()
    
    best_params = grid_search(args.target_wer, args.margin)
