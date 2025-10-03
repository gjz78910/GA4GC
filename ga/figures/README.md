# Pareto Front Visualization

## Purpose

Two figures demonstrate that GA4GC performs **directed search** (not random search) for 3-objective optimization.

---

## Figures

### 1. `pareto_front_evolution.pdf`
**3D Pareto Front Evolution Across 5 Generations**

Shows cumulative Pareto fronts in 5 separate subplots:
- **Gray points**: All configurations evaluated in each generation
- **Colored points**: Pareto-optimal solutions (connected with lines)
- **Gold star**: Default configuration baseline
- **Colors**: Red, Orange, Green, Blue, Purple (Gen 1-5)

**Evidence of directed search**: Pareto front grows systematically from 2→5 solutions, with solutions spanning multiple generations (evolutionary memory).

### 2. `convergence_metrics.pdf`
**Hypervolume Convergence and Growth**

Two subplots showing quantitative improvement:
- **(a) Hypervolume**: Increases from 70.6% to 83.0% (+17.5%)
- **(b) Pareto Front Size**: Grows monotonically from 2 to 5 solutions (+150%)

**Evidence of directed search**: Clear upward trends in both metrics, with 80% of all configurations dominated by the final Pareto front.

---

## Key Results

| Metric | Gen 1 | Gen 5 | Improvement |
|--------|-------|-------|-------------|
| Hypervolume | 70.6% | 83.0% | +17.5% |
| Pareto Solutions | 2 | 5 | +150% |
| Dominated Configs | - | 20/25 | 80% |

**Conclusion**: GA4GC performs directed optimization through:
1. Systematic hypervolume improvement
2. Monotonic Pareto front growth
3. High domination rate (80% of solutions dominated)
4. Multi-generation solutions in final front (evolutionary memory)

---

## Technical Notes

- **Hypervolume calculation**: Cumulative Pareto front only (not all individuals), normalized to [0,1] with reference point [-0.1,-0.1,-0.1]
- **Pareto front strategy**: Cumulative (considers all evaluations through current generation)
- **Objectives**: Minimize runtime, maximize performance gain, maximize correctness
- **Viewing angle**: elev=30°, azim=-60° (origin facing audience)
