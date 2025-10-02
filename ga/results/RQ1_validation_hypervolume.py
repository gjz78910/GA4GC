#!/usr/bin/env python3
"""
Validation Hypervolume Calculator using PyMOO
This script computes individual hypervolumes for validation results
"""

import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

def calculate_hypervolume_pymoo(points, reference_point):
    """Calculate hypervolume using PyMOO library"""
    if len(points) == 0:
        return 0.0
    
    points = np.array(points)
    reference_point = np.array(reference_point)
    
    # PyMOO expects minimization objectives and reference point to be dominated by all points
    # Convert our maximization-oriented normalized points to minimization format
    pymoo_points = 1.0 - points  # Invert for minimization
    pymoo_reference = 1.0 - reference_point  # Adjust reference accordingly
    
    # Create hypervolume indicator
    hv_indicator = HV(ref_point=pymoo_reference)
    
    # Calculate hypervolume
    return hv_indicator(pymoo_points)

def normalize_objectives(objectives, minimize_objectives):
    """Normalize objectives to [0, 1] range"""
    normalized = np.array(objectives)
    
    for i in range(normalized.shape[1]):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        
        if max_val > min_val:
            if minimize_objectives[i]:
                # For minimization: smaller is better, so invert
                normalized[:, i] = (max_val - normalized[:, i]) / (max_val - min_val)
            else:
                # For maximization: larger is better
                normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 0.5  # All values are the same
    
    return normalized

def compute_validation_hypervolumes():
    """
    Compute individual hypervolumes for all validation results
    """
    
    # Load all validation summary files
    validation_files = [
        ('validation_default_summary.csv', 'default'),
        ('validation_pareto_eval4_summary.csv', 4),
        ('validation_pareto_eval5_summary.csv', 5), 
        ('validation_pareto_eval9_summary.csv', 9),
        ('validation_pareto_eval15_summary.csv', 15),
        ('validation_pareto_eval16_summary.csv', 16)
    ]
    
    print("=== VALIDATION HYPERVOLUME ANALYSIS (PyMOO) ===")
    
    # Load baseline for normalization context
    baseline_df = pd.read_csv('single_config_summary_20250920_173830.csv')
    baseline_obj = baseline_df[['objective1_execution_time', 'objective2_model_improved', 'objective3_success']].values[0]
    
    # Load original GA results for normalization context
    ga_df = pd.read_csv('ga_results_20250919_154742.csv')
    ga_objectives = ga_df[['objective1_execution_time', 'objective2_model_improved', 'objective3_success']].values
    
    validation_data = []
    validation_objectives = []
    
    # Collect all validation data
    for file_name, eval_id in validation_files:
        try:
            df = pd.read_csv(file_name)
            obj_values = [
                df['objective1_execution_time'].iloc[0],
                df['objective2_model_improved'].iloc[0], 
                df['objective3_success'].iloc[0]
            ]
            validation_data.append((eval_id, obj_values))
            validation_objectives.append(obj_values)
            print(f"Loaded Eval {eval_id}: RT={obj_values[0]:.1f}s, Perf={obj_values[1]:.4f}, Succ={obj_values[2]:.1f}")
        except FileNotFoundError:
            print(f"Warning: {file_name} not found, skipping...")
    
    if not validation_data:
        print("No validation files found!")
        return {}
    
    # Create combined dataset for consistent normalization
    # Include GA results, baseline, and validation results for proper normalization
    all_objectives = np.vstack([
        ga_objectives,
        baseline_obj.reshape(1, -1),
        np.array(validation_objectives)
    ])
    
    minimize_objectives = [True, False, False]  # minimize time, maximize others
    normalized_all = normalize_objectives(all_objectives, minimize_objectives)
    
    # Extract normalized validation points (last len(validation_data) points)
    validation_normalized = normalized_all[-len(validation_data):]
    
    # Reference point (same as used in training)
    reference_point = [-0.1, -0.1, -0.1]
    
    print(f"\nUsing reference point: {reference_point}")
    print(f"Normalized validation points shape: {validation_normalized.shape}")
    
    validation_hypervolumes = {}
    
    # Calculate individual hypervolumes using PyMOO
    for i, (eval_id, _) in enumerate(validation_data):
        point = validation_normalized[i]
        point_hv = calculate_hypervolume_pymoo([point], reference_point)
        validation_hypervolumes[eval_id] = point_hv
        
        print(f"Validation Eval {eval_id}: {point_hv:.4f} ({point_hv*100:.2f}%)")
    
    # Calculate average validation hypervolume
    avg_hv = np.mean(list(validation_hypervolumes.values()))
    print(f"\nAverage Validation Hypervolume: {avg_hv:.4f} ({avg_hv*100:.2f}%)")
    
    # Compare with original training hypervolumes
    print("\n=== TRAINING vs VALIDATION COMPARISON ===")
    training_hvs = {
        'default': 0.52, 4: 5.82, 5: 70.28, 9: 9.25, 15: 33.42, 16: 1.10  # From training results
    }
    
    print("Config | Training HV (%) | Validation HV (%) | Retention (%)")
    print("-------|-----------------|-------------------|---------------")
    total_retention = 0
    valid_comparisons = 0
    
    # Sort keys with default first, then numeric
    numeric_keys = [k for k in validation_hypervolumes.keys() if k != 'default']
    sorted_keys = (['default'] if 'default' in validation_hypervolumes else []) + sorted(numeric_keys)
    for eval_id in sorted_keys:
        if eval_id in training_hvs:
            training_hv = training_hvs[eval_id]
            validation_hv = validation_hypervolumes[eval_id] * 100
            retention = (validation_hv / training_hv) * 100 if training_hv > 0 else 0
            eval_str = f"#{eval_id}" if eval_id != 'default' else "default"
            print(f"  {eval_str:7} |      {training_hv:5.2f}      |       {validation_hv:5.2f}       |    {retention:6.1f}")
            total_retention += retention
            valid_comparisons += 1
    
    if valid_comparisons > 0:
        avg_retention = total_retention / valid_comparisons
        print(f"Average retention: {avg_retention:.1f}%")
    
    # Print values for table update
    print("\n=== VALUES FOR RQ1 TABLE UPDATE ===")
    print("Replace XX.X in the VHV (%) column with:")
    # Use same sorted keys as above
    numeric_keys = [k for k in validation_hypervolumes.keys() if k != 'default']
    sorted_keys = (['default'] if 'default' in validation_hypervolumes else []) + sorted(numeric_keys)
    for eval_id in sorted_keys:
        hv_percent = validation_hypervolumes[eval_id] * 100
        if eval_id == 'default':
            print(f"Default: {hv_percent:.1f}")
        else:
            print(f"Eval #{eval_id}: {hv_percent:.1f}")
    
    return validation_hypervolumes

if __name__ == "__main__":
    validation_hvs = compute_validation_hypervolumes()