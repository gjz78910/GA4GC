# Pareto Front Visualization

## Purpose

Demonstrates that GA4GC performs **directed search** (not random search) for 3-objective optimization, addressing Review #20A's concern about baseline comparison.

---

## NSGA-II Figures

### 1. `pareto_front_evolution.jpg`
**3D Pareto Front Evolution Across 5 Generations**

Shows cumulative Pareto fronts in 5 separate subplots:
- **Gray points**: All configurations evaluated in each generation
- **Colored points**: Pareto-optimal solutions (connected with lines)
- **Gold star**: Default configuration baseline
- **Colors**: Red, Orange, Green, Blue, Purple (Gen 1-5)

**Evidence**: Pareto front grows systematically from 2→5 solutions, with solutions spanning multiple generations (evolutionary memory).

### 2. `convergence_metrics.jpg`
**Hypervolume Convergence and Growth**

Two subplots showing quantitative improvement:
- **(a) Hypervolume**: Increases from 70.6% to 83.0% (+17.5%)
- **(b) Pareto Front Size**: Grows monotonically from 2 to 5 solutions (+150%)

**Evidence**: Clear upward trends in both metrics, with 80% of all configurations dominated by the final Pareto front.

---

## Baseline Comparison: NSGA-II vs Random Search

**Addressing Review #20A**: "The study lacks comparison with other search algorithms to validate whether NSGA-II is performing better than simpler alternatives."

### 3. `comparison_pareto_fronts.jpg`
**5-Generation Evolution Comparison (2 rows)**

Side-by-side visualization of NSGA-II vs Random Search across all 5 generations:
- **Row 1**: NSGA-II shows systematic growth (2→3→4→5→5 solutions)
- **Row 2**: Random Search shows stagnation (2→2→2→3→3 solutions)

**Evidence**: NSGA-II discovers 2 more Pareto solutions with better coverage.

### 4. `comparison_convergence.jpg`
**Convergence Metrics Comparison**

Two subplots comparing NSGA-II vs Random Search:
- **(a) Cumulative Hypervolume**: NSGA-II 83.0% vs Random 53.1% (+29.9% advantage)
- **(b) Pareto Front Size**: NSGA-II 5 solutions vs Random 3 solutions (+67% more)

**Evidence**: NSGA-II shows smooth convergence while Random Search shows erratic fluctuation.

---

## Key Results

### NSGA-II Performance

| Metric | Gen 1 | Gen 5 | Improvement |
|--------|-------|-------|-------------|
| Hypervolume | 70.6% | 83.0% | +17.5% |
| Pareto Solutions | 2 | 5 | +150% |
| Dominated Configs | - | 20/25 | 80% |

### NSGA-II vs Random Search

| Metric | NSGA-II | Random | Advantage |
|--------|---------|--------|-----------|
| Final Hypervolume | 83.0% | 53.1% | **+29.9%** |
| Pareto Solutions | 5 | 3 | **+2 (67% more)** |
| Best Correctness | 8 | 6 | **+33%** |
| Best Performance | 10.67% | 7.05% | **+51%** |

---

## Conclusion

**NSGA-II performs directed search, NOT random behavior:**

1. **29.9% higher hypervolume** than random search
2. **67% more Pareto solutions** discovered
3. **Systematic convergence** (+17.5%) vs erratic fluctuation (+1297% from terrible start)
4. **Superior solution quality** across all objectives

This directly addresses Review #20A's concern with empirical evidence.

---

## Technical Notes

- **Hypervolume**: Cumulative Pareto front only, normalized to [0,1] with reference point [-0.1,-0.1,-0.1]
- **Pareto front**: Cumulative strategy (all evaluations through current generation)
- **Objectives**: Minimize runtime, maximize performance gain, maximize correctness
- **Baseline**: Random search with same budget (25 evaluations)
