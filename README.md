# GreenerAgent

GreenerAgent explores sustainable AI-driven software engineering by optimizing the energy consumption of LLM-based coding agents while maintaining performance. This project adapts [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) for the [SWE-Perf](https://swe-perf.github.io/) benchmark, focusing on performance optimization tasks.

## Current Setup

- **Framework**: Mini-SWE-Agent with SWE-Perf integration
- **Models**:

| Name | Access | Context Window |
|------|---------|----------|
| `Codestral 2508` | Mistral API | 256K |
| `codestral:22b` | Ollama (local) | 32K |

Our goal is to use NSGA-II to find Pareto optimal hyperparameter and prompt configurations that achieve the trade-off between:
- **Agent Energy Efficiency**: Reduce computational costs and carbon footprint
- **SWE-Perf Performance Score**: Maintain optimization quality on SWE-Perf tasks

## Possible Energy Efficiency Metrics

- Execution Time - Most easy to measure, strong energy correlation across all models
- API Costs - Direct monetary proxy for Mistral API energy consumption
- CPU/Memory Usage - Resource indicators for local Ollama models

## Next Steps for the Paper

### Phase 1: LLM Validation
- [x] Test if Mistral and Ollama models work with Mini-SWE-Agent
- [x] Run basic functionality tests on simple SWE-Perf tasks
- [x] Verify model API connections and local Ollama setup

### Phase 2: Baseline Testing
- [x] Run Mini-SWE-Agent with default hyperparameters on SWE-Perf tasks
- [x] Measure baseline SWE-Perf performance scores
- [ ] Measure baseline energy metrics: execution time, API costs (Mistral), CPU/memory (Ollama)
- [ ] Establish automated measurement procedures for consistent evaluation

### Phase 3: NSGA-II Optimization
- [ ] Implement NSGA-II framework for multi-objective optimization
- [ ] Define objectives: minimize energy consumption, maximize SWE-Perf performance
- [ ] Search for Pareto optimal hyperparameter and prompt configurations
- [ ] Decision variables: model selection, temperature, step limits, sampling parameters, prompt templates

### Phase 4: Evaluation & Analysis
- [ ] Evaluate hypervolume of Pareto front solutions
- [ ] Analyze trade-offs between execution time/API costs and SWE-Perf performance
- [ ] Identify optimal configurations and most influential hyperparameters
- [ ] Statistical validation that energy savings don't degrade performance

## Key Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `sweperf_oracle.yaml` | Main configuration | `SWE-Perf/mini-swe-agent/src/minisweagent/config/extra/` |
| `sweperf.py` | Runner script | `SWE-Perf/mini-swe-agent/src/minisweagent/run/extra/` |

## Resources
- **SWE-Perf**: [Homepage](https://swe-perf.github.io/) • [Leaderboard](https://swe-perf.github.io/leaderboard.html) • [Paper](https://arxiv.org/pdf/2507.12415)
- **Mini-SWE-Agent**: [GitHub](https://github.com/SWE-agent/mini-swe-agent) • [Documentation](https://mini-swe-agent.com) 

## Other ideas during group discussion:

Coding agents’ idea:
### Plain:
-	1st Open-source mistral (Ollama)
-	Benchmark: the LLM generates the patch and the benchmark runs everything in Docker to evaluate the performance: Result -> Fail / Optimised
Paid version:
-	Check the results in the paid version, we can maybe compare the quality with respect to the free one
### GIN:
-	Code improvement. 
### Green:
-	Selection of the smallest LLMs to get the same results.
-	Python for carbon print measurement
### Search Component:
-	Parameter fine-tuning for the LLMs
-	Parameters to optimise: runtime, costs (energy usage, $$$), number of agent turns 
