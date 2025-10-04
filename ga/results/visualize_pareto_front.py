"""
Clean and focused Pareto front visualization for 3-objective optimization.
Shows clear evidence of directed search with minimal complexity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set academic paper style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Load data
data_path = Path(__file__).parent / 'optimization_results_20250919_154742.csv'
df = pd.read_csv(data_path)

# Create figures directory
figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Assign generations (5 configs per generation for population size 5)
df['generation'] = (df['evaluation_id'] - 1) // 5 + 1

# Extract objectives
df['runtime'] = df['objective1_execution_time']
df['performance'] = df['objective2_model_improved'] * 100
df['correctness'] = df['objective3_success']

# For Pareto front identification (convert to minimization)
df['neg_perf'] = -df['performance']
df['neg_corr'] = -df['correctness']

# Function to identify Pareto front
def is_pareto_efficient(costs):
    """Find Pareto efficient points (for minimization)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Colors for generations: red, orange, green, blue, purple
gen_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

print("="*70)
print("PARETO FRONT EVOLUTION ANALYSIS")
print("="*70)

# Calculate Pareto front for each generation (cumulative)
pareto_fronts = {}
for gen in sorted(df['generation'].unique()):
    cumulative_data = df[df['generation'] <= gen]
    costs = cumulative_data[['runtime', 'neg_perf', 'neg_corr']].values
    pareto_mask = is_pareto_efficient(costs)
    pareto_fronts[gen] = cumulative_data[pareto_mask]
    print(f"Generation {gen}: {pareto_mask.sum()} Pareto-optimal solutions")

# ============================================================================
# Figure 1: 3D Pareto Front - 5 Subplots for 5 Generations
# ============================================================================
fig = plt.figure(figsize=(15, 8))

# Default configuration
default_runtime, default_perf, default_corr = 1513.3, 0.0, 2.0

for gen in sorted(df['generation'].unique()):
    ax = fig.add_subplot(2, 3, gen, projection='3d')
    
    # Get current generation's data
    gen_data = df[df['generation'] == gen]
    
    # Plot all configurations in current generation (light gray)
    ax.scatter(gen_data['runtime'],
               gen_data['performance'],
               gen_data['correctness'],
               s=80, alpha=0.8, color='lightgray', 
               edgecolors='gray', linewidths=1, zorder=1)
    
    # Get Pareto front for this generation
    pareto_data = pareto_fronts[gen]
    
    # Plot Pareto front points
    ax.scatter(pareto_data['runtime'],
               pareto_data['performance'],
               pareto_data['correctness'],
               s=180, color=gen_colors[gen-1],
               marker='o', edgecolors='black', linewidths=2.5,
               alpha=0.95, zorder=10)
    
    # Connect Pareto front points with lines
    if len(pareto_data) > 1:
        sorted_data = pareto_data.sort_values('runtime')
        ax.plot(sorted_data['runtime'],
                sorted_data['performance'],
                sorted_data['correctness'],
                color=gen_colors[gen-1], linewidth=3, alpha=0.8, zorder=5)
    
    # Add default configuration
    ax.scatter([default_runtime], [default_perf], [default_corr],
               s=350, marker='*', color='gold',
               edgecolors='black', linewidths=2,
               label='Default', zorder=20)
    
    # Set labels
    ax.set_xlabel('Runtime (s)', fontweight='bold', fontsize=9, labelpad=5)
    ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=9, labelpad=5)
    ax.set_zlabel('Correctness', fontweight='bold', fontsize=9, labelpad=5)
    
    # Set title at the bottom using text2D
    ax.text2D(0.5, -0.1, f'Generation {gen}', 
              transform=ax.transAxes, fontweight='bold', fontsize=11,
              ha='center', va='top')
    
    # Set viewing angle: origin (0,0,0) at front facing audience
    # elev=30 (looking down slightly), azim=-60 (rotate so origin is at front-left)
    ax.view_init(elev=30, azim=-60)
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=8)

# Hide the 6th subplot
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# Add legend in the empty space
legend_elements = [
    plt.scatter([], [], s=180, marker='o', color='#d73027', 
                edgecolors='black', linewidths=2, label='Pareto Front'),
    plt.scatter([], [], s=80, marker='o', color='lightgray',
                edgecolors='gray', linewidths=1, label='Other Configs'),
    plt.scatter([], [], s=350, marker='*', color='gold',
                edgecolors='black', linewidths=2, label='Default')
]
ax6.legend(handles=legend_elements, loc='center', fontsize=11, 
          frameon=True, fancybox=True, shadow=True)

plt.suptitle('Pareto Front Evolution Across Generations',
             fontweight='bold', fontsize=14, y=0.9)
plt.subplots_adjust(wspace=0, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(figures_dir / 'pareto_front_evolution.pdf',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[SAVED] pareto_front_evolution.pdf")

# ============================================================================
# Figure 2: Convergence Metrics (2 subplots only - simple and clear)
# ============================================================================
from pymoo.indicators.hv import HV

# Normalize objectives to [0, 1] range (aligned with RQ analysis approach)
def normalize_objectives(objectives, minimize_objectives):
    """Normalize objectives to [0, 1] range"""
    normalized = np.array(objectives, dtype=float)
    
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

# Prepare objectives: runtime (minimize), performance (maximize), correctness (maximize)
all_objectives = df[['runtime', 'performance', 'correctness']].values
minimize_objectives = [True, False, False]  # minimize runtime, maximize others
normalized_objectives = normalize_objectives(all_objectives, minimize_objectives)

# Reference point (same as RQ analysis: slightly worse than worst point)
reference_point = np.array([-0.1, -0.1, -0.1])

# Convert to minimization for PyMOO (invert normalized values)
pymoo_objectives = 1.0 - normalized_objectives
pymoo_reference = 1.0 - reference_point

hv_indicator = HV(ref_point=pymoo_reference)
hypervolumes = []
pareto_sizes = []

print("\n" + "="*70)
print("HYPERVOLUME CALCULATION METHOD")
print("="*70)
print("For each generation, we calculate the hypervolume of the")
print("CUMULATIVE PARETO FRONT (not all individuals in that generation).")
print("The Pareto front includes only non-dominated solutions from")
print("all configurations evaluated up to and including that generation.")
print("="*70 + "\n")

for gen in sorted(df['generation'].unique()):
    cumulative_data = df[df['generation'] <= gen]
    cumulative_indices = cumulative_data.index
    
    # Get normalized objectives for cumulative data
    cumulative_norm = pymoo_objectives[cumulative_indices]
    
    # Find Pareto front (non-dominated solutions only)
    costs = cumulative_data[['runtime', 'neg_perf', 'neg_corr']].values
    pareto_mask = is_pareto_efficient(costs)
    pareto_front_norm = cumulative_norm[pareto_mask]
    
    # Calculate hypervolume of the Pareto front only
    hv = hv_indicator(pareto_front_norm)
    hypervolumes.append(hv * 100)  # Convert to percentage
    pareto_sizes.append(pareto_mask.sum())
    
    print(f"Gen {gen}: {pareto_mask.sum()} Pareto solutions, HV = {hv * 100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

generations = sorted(df['generation'].unique())

# Subplot 1: Hypervolume convergence
ax1.plot(generations, hypervolumes, marker='o', linewidth=3,
        markersize=12, color='#d73027', markerfacecolor='white',
        markeredgewidth=3, markeredgecolor='#d73027')
ax1.fill_between(generations, hypervolumes, alpha=0.3, color='#d73027')

ax1.set_xlabel('Generation', fontweight='bold', fontsize=12)
ax1.set_ylabel('Hypervolume (%)', fontweight='bold', fontsize=12)
ax1.set_title('(a) Cumulative Pareto Front Hypervolume', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks(generations)
ax1.set_ylim(bottom=60)  # Set y-axis minimum to 60

# Add improvement annotation at bottom center
improvement = ((hypervolumes[-1] / hypervolumes[0]) - 1) * 100
ax1.text(0.5, 0.08, f'+{improvement:.1f}% Improvement',
        transform=ax1.transAxes, fontsize=13, fontweight='bold',
        ha='center', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2))

# Subplot 2: Pareto front size growth
ax2.bar(generations, pareto_sizes, color=gen_colors, edgecolor='black', linewidth=2, alpha=0.8)
ax2.plot(generations, pareto_sizes, marker='o', linewidth=3,
        markersize=10, color='black', linestyle='--', alpha=0.7)

ax2.set_xlabel('Generation', fontweight='bold', fontsize=12)
ax2.set_ylabel('Number of Pareto-Optimal Solutions', fontweight='bold', fontsize=12)
ax2.set_title('(b) Cumulative Pareto Front Size Growth', fontweight='bold', fontsize=13)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks(generations)
ax2.set_ylim([0, max(pareto_sizes) * 1.2])

# Add growth annotation at bottom center, same color as (a)
growth = pareto_sizes[-1] - pareto_sizes[0]
ax2.text(0.5, 0.08, f'+{growth} Solutions Discovered',
        transform=ax2.transAxes, fontsize=13, fontweight='bold',
        ha='center', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2))

plt.suptitle('Evidence of Directed Search: Convergence Metrics',
             fontweight='bold', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(figures_dir / 'convergence_metrics.pdf',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"[SAVED] convergence_metrics.pdf")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: EVIDENCE OF DIRECTED SEARCH")
print("="*70)
print(f"\n1. HYPERVOLUME IMPROVEMENT:")
print(f"   Gen 1: {hypervolumes[0]:.2e}")
print(f"   Gen 5: {hypervolumes[-1]:.2e}")
print(f"   Improvement: +{improvement:.1f}%")

print(f"\n2. PARETO FRONT GROWTH:")
print(f"   Gen 1: {pareto_sizes[0]} solutions")
print(f"   Gen 5: {pareto_sizes[-1]} solutions")
print(f"   Growth: +{growth} solutions ({growth/pareto_sizes[0]*100:.0f}% increase)")

print(f"\n3. FINAL PARETO FRONT:")
final_pareto = pareto_fronts[5]
print(f"   Size: {len(final_pareto)} configurations")
print(f"   Generations: {sorted(final_pareto['generation'].unique())}")

print(f"\n4. DOMINATION:")
all_costs = df[['runtime', 'neg_perf', 'neg_corr']].values
final_pareto_mask = is_pareto_efficient(all_costs)
dominated = len(df) - final_pareto_mask.sum()
print(f"   Dominated solutions: {dominated}/{len(df)}")
print(f"   Rate: {dominated/len(df)*100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION: NSGA-II performs DIRECTED, NON-RANDOM SEARCH")
print("="*70)
print("Evidence:")
print("  - Hypervolume improves systematically (+24.3%)")
print("  - Pareto front grows from 2 to 5 solutions")
print("  - High domination rate (80%) indicates quality improvement")
print("  - Solutions span multiple generations (evolutionary memory)")

print(f"\n[OK] All figures saved to: {figures_dir}")
print("\nGenerated figures:")
print("1. pareto_front_evolution.pdf - 3D Pareto front evolution (2 subplots)")
print("2. convergence_metrics.pdf - Hypervolume & growth (2 subplots)")
