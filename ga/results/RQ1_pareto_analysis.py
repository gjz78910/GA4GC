import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

def is_dominated(point1, point2, minimize_objectives):
    """Check if point1 is dominated by point2"""
    better_in_all = True
    strictly_better_in_at_least_one = False
    
    for i, minimize in enumerate(minimize_objectives):
        if minimize:  # Minimization objective
            if point1[i] > point2[i]:  # point1 is worse
                strictly_better_in_at_least_one = True
            elif point1[i] < point2[i]:  # point1 is better
                better_in_all = False
        else:  # Maximization objective
            if point1[i] < point2[i]:  # point1 is worse
                strictly_better_in_at_least_one = True
            elif point1[i] > point2[i]:  # point1 is better
                better_in_all = False
    
    return better_in_all and strictly_better_in_at_least_one

def find_pareto_front(objectives, minimize_objectives):
    """Find Pareto optimal solutions"""
    n_points = len(objectives)
    pareto_front = []
    
    for i in range(n_points):
        is_pareto_optimal = True
        for j in range(n_points):
            if i != j and is_dominated(objectives[i], objectives[j], minimize_objectives):
                is_pareto_optimal = False
                break
        if is_pareto_optimal:
            pareto_front.append(i)
    
    return pareto_front

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

def calculate_individual_hypervolume_pymoo(point, reference_point):
    """Calculate individual hypervolume of a single point using PyMOO"""
    return calculate_hypervolume_pymoo([point], reference_point)

def analyze_ga_results(csv_file, baseline_file=None):
    """Comprehensive analysis of GA results with optional baseline comparison"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Load baseline data if provided
    baseline_data = None
    if baseline_file:
        baseline_df = pd.read_csv(baseline_file)
        baseline_data = {
            'execution_time': baseline_df['objective1_execution_time'].iloc[0],
            'model_improved': baseline_df['objective2_model_improved'].iloc[0],
            'success': baseline_df['objective3_success'].iloc[0]
        }
        print("=== BASELINE COMPARISON ===")
        print(f"Default Configuration Performance:")
        print(f"  Execution Time: {baseline_data['execution_time']:.1f}s")
        print(f"  Model Improved: {baseline_data['model_improved']:.4f}")
        print(f"  Success Rate: {baseline_data['success']:.1f}")
        print()
    
    # Extract objectives (we want to minimize time, maximize improvement and success)
    objectives = df[['objective1_execution_time', 'objective2_model_improved', 'objective3_success']].values
    minimize_objectives = [True, False, False]  # minimize time, maximize others
    
    # Find Pareto front
    pareto_indices = find_pareto_front(objectives, minimize_objectives)
    pareto_front_df = df.iloc[pareto_indices].copy()
    
    print("=== PARETO FRONT ANALYSIS ===")
    print(f"Total evaluations: {len(df)}")
    print(f"Pareto optimal solutions: {len(pareto_indices)}")
    print(f"Pareto efficiency: {len(pareto_indices)/len(df)*100:.1f}%")
    print()
    
    # Normalize objectives for hypervolume calculation
    normalized_objectives = normalize_objectives(objectives, minimize_objectives)
    normalized_pareto_points = normalized_objectives[pareto_indices]
    
    # Calculate hypervolume using PyMOO
    # Reference point should be DOMINATED by all solutions (worse than worst)
    reference_point = [-0.1, -0.1, -0.1]  # Worse than any normalized solution
    hypervolume = calculate_hypervolume_pymoo(normalized_pareto_points, reference_point)
    
    print(f"Hypervolume: {hypervolume:.4f}")
    print(f"Hypervolume percentage: {hypervolume*100:.2f}%")
    print()
    
    # Compare with baseline if available
    if baseline_data:
        print("=== PARETO vs BASELINE ANALYSIS ===")
        best_pareto_time = min(objectives[pareto_indices, 0])
        best_pareto_improved = max(objectives[pareto_indices, 1])
        best_pareto_success = max(objectives[pareto_indices, 2])
        
        time_improvement = (baseline_data['execution_time'] - best_pareto_time) / baseline_data['execution_time'] * 100
        improved_improvement = (best_pareto_improved - baseline_data['model_improved']) / (baseline_data['model_improved'] + 1e-10) * 100
        success_improvement = (best_pareto_success - baseline_data['success']) / (baseline_data['success'] + 1e-10) * 100
        
        print(f"Speed Improvement: {time_improvement:.1f}% faster ({best_pareto_time:.1f}s vs {baseline_data['execution_time']:.1f}s)")
        print(f"Quality Improvement: +{best_pareto_improved:.4f} vs {baseline_data['model_improved']:.4f} ({improved_improvement:.1f}% better)")
        print(f"Success Improvement: +{best_pareto_success:.1f} vs {baseline_data['success']:.1f} ({success_improvement:.1f}% better)")
        
        # Check if baseline is dominated by any Pareto solution
        baseline_objectives = [baseline_data['execution_time'], baseline_data['model_improved'], baseline_data['success']]
        dominated_by = []
        for i, pareto_idx in enumerate(pareto_indices):
            pareto_point = objectives[pareto_idx]
            if is_dominated(baseline_objectives, pareto_point, minimize_objectives):
                dominated_by.append(df.iloc[pareto_idx]['evaluation_id'])
        
        if dominated_by:
            print(f"[SUCCESS] Baseline is DOMINATED by Pareto solutions: {dominated_by}")
        else:
            print("[WARNING] Baseline is NOT dominated by any Pareto solution")
        print()
    
    # Calculate individual hypervolume contributions
    print("=== INDIVIDUAL HYPERVOLUME ANALYSIS ===")
    individual_hypervolumes = {}
    
    # CORRECTED: Include baseline in normalization for fair comparison
    if baseline_data:
        # Include baseline in the same normalization as GA results
        all_objectives_with_baseline = np.vstack([
            objectives,
            [[baseline_data['execution_time'], baseline_data['model_improved'], baseline_data['success']]]
        ])
        normalized_all = normalize_objectives(all_objectives_with_baseline, minimize_objectives)
        
        # Re-extract normalized pareto points and baseline
        normalized_pareto_points = normalized_all[pareto_indices]
        baseline_normalized = normalized_all[-1]  # Last point is baseline
        
        # Recalculate total hypervolume with PyMOO
        hypervolume = calculate_hypervolume_pymoo(normalized_pareto_points, reference_point)
        print(f"Total hypervolume (PyMOO): {hypervolume:.4f} ({hypervolume*100:.2f}%)")
        
        # Calculate individual hypervolume for baseline using PyMOO
        baseline_hv = calculate_individual_hypervolume_pymoo(baseline_normalized, reference_point)
        individual_hypervolumes['baseline'] = baseline_hv
        print(f"Baseline individual hypervolume: {baseline_hv:.4f} ({baseline_hv*100:.2f}%)")
    
    # Calculate INDIVIDUAL hypervolume for each Pareto solution using PyMOO
    for i, idx in enumerate(pareto_indices):
        eval_id = df.iloc[idx]['evaluation_id']
        # Use PyMOO for consistent calculation
        point_normalized = normalized_pareto_points[i] if baseline_data else normalized_pareto_points[i]
        individual_hv = calculate_individual_hypervolume_pymoo(point_normalized, reference_point)
        individual_hypervolumes[eval_id] = individual_hv
        print(f"Evaluation #{eval_id} individual hypervolume: {individual_hv:.4f} ({individual_hv*100:.2f}%)")
    print()
    
    # Analyze Pareto front solutions
    print("=== PARETO OPTIMAL CONFIGURATIONS ===")
    pareto_analysis = []
    
    for idx, (_, row) in enumerate(pareto_front_df.iterrows()):
        eval_id = row['evaluation_id']
        config = {
            'evaluation_id': eval_id,
            'execution_time': row['objective1_execution_time'],
            'model_improved': row['objective2_model_improved'],
            'success': row['objective3_success'],
            'step_limit': row['config1_step_limit'],
            'cost_limit': row['config2_cost_limit'],
            'temperature': row['config3_temperature'],
            'top_p': row['config4_top_p'],
            'max_tokens': row['config5_max_tokens'],
            'model_timeout': row['config6_model_timeout'],
            'env_timeout': row['config7_env_timeout'],
            'baseline': row['config8_baseline'],
            'hypervolume': individual_hypervolumes[eval_id]
        }
        pareto_analysis.append(config)
        
        print(f"Evaluation #{eval_id}:")
        print(f"  Objectives: Time={config['execution_time']:.1f}s, Improved={config['model_improved']:.4f}, Success={config['success']:.1f}")
        contribution_pct = (config['hypervolume'] / hypervolume * 100) if hypervolume > 0 else 0
        print(f"  HV Contribution: {config['hypervolume']:.4f} ({contribution_pct:.1f}% of total)")
        print(f"  Config: Steps={config['step_limit']}, Temp={config['temperature']:.3f}, Tokens={config['max_tokens']}, Baseline={config['baseline']}")
        print()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Objective Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 2D projections of objectives
    obj_names = ['Execution Time (s)', 'Model Improved', 'Success Rate']
    obj_pairs = [(0, 1), (0, 2), (1, 2)]
    
    for i, (obj1, obj2) in enumerate(obj_pairs):
        ax = axes[i//2, i%2]
        
        # Plot all points
        ax.scatter(objectives[:, obj1], objectives[:, obj2], 
                  alpha=0.6, color='lightblue', s=50, label='All Solutions')
        
        # Plot Pareto front
        pareto_obj = objectives[pareto_indices]
        ax.scatter(pareto_obj[:, obj1], pareto_obj[:, obj2], 
                  color='red', s=100, label='Pareto Front', marker='*')
        
        # Annotate Pareto points
        for j, idx in enumerate(pareto_indices):
            ax.annotate(f'{df.iloc[idx]["evaluation_id"]}', 
                       (objectives[idx, obj1], objectives[idx, obj2]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(obj_names[obj1])
        ax.set_ylabel(obj_names[obj2])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hypervolume evolution (if we had multiple generations)
    axes[1, 1].text(0.1, 0.5, f'Hypervolume: {hypervolume:.4f}\n({hypervolume*100:.2f}%)', 
                    fontsize=14, transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 1].set_title('Hypervolume Metric')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pareto_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pareto_analysis, hypervolume, pareto_front_df, baseline_data, individual_hypervolumes

# Run analysis
if __name__ == "__main__":
    pareto_configs, hv, pareto_df, baseline, individual_hvs = analyze_ga_results(
        'ga_results_20250919_154742.csv', 
        'single_config_summary_20250920_173830.csv'
    )
    
    # Save Pareto front to CSV
    pareto_df.to_csv('pareto_front_20250919.csv', index=False)
    print(f"Pareto front saved to pareto_front_20250919.csv")
    
    # Print individual hypervolumes for table
    print("\n=== HYPERVOLUME VALUES FOR TABLE ===")
    if 'baseline' in individual_hvs:
        print(f"Baseline: {individual_hvs['baseline']*100:.2f}%")
    for config in pareto_configs:
        eval_id = config['evaluation_id']
        print(f"Eval #{eval_id}: {individual_hvs[eval_id]*100:.2f}%")
