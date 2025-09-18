from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from problem import ConfigProblem
import copy
import sys
import time

print("="*80)
print("GENETIC ALGORITHM OPTIMIZATION FOR SWE-AGENT CONFIGURATIONS")
print("="*80)

# Instantiate problem
print("Instantiating problem...")
problem = ConfigProblem()
print("Problem instantiated successfully!")

# Configure NSGA2
print("Configuring NSGA2 algorithm...")
algorithm = NSGA2(
    pop_size=5,
    eliminate_duplicates=True
)
print(f"Algorithm configured: population_size={5}")

termination = get_termination("n_gen", 5)  # run for 5 generations
print(f"Termination configured: max_generations={5}")

print("\nStarting optimization...")
print("This will evaluate multiple configurations using the SWE-Agent...")
print("Each evaluation includes running the agent and performance measurement.")
print("="*80)

# Flush stdout to ensure immediate output
sys.stdout.flush()

start_time = time.time()

# Run optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

total_time = time.time() - start_time
print(f"\n{'='*80}")
print(f"OPTIMIZATION COMPLETED in {total_time:.2f} seconds")
print(f"{'='*80}")


# Pareto front solutions
print("Pareto Front Objectives (F):")
print(res.F)

print("\nCorresponding Decision Variables (X):")
print(res.X)




def genome_to_dict(genome, base_config):
    cfg = copy.deepcopy(base_config)
    cfg["step_limit"] = int(genome[0])
    cfg["cost_limit"] = float(genome[1])
    cfg["model"]["model_kwargs"]["temperature"] = float(genome[2])
    cfg["model"]["model_kwargs"]["top_p"] = float(genome[3])
    cfg["model"]["model_kwargs"]["max_tokens"] = int(genome[4])
    cfg["model"]["model_kwargs"]["timeout"] = int(genome[5])
    cfg["environment"]["timeout"] = int(genome[6])
    return cfg

# Convert the first Pareto-optimal solution
if len(res.X) > 0:
    best_config = genome_to_dict(res.X[0], problem.base_config)
    print("\nBest Config Dict Example:\n", best_config)
else:
    print("\nNo solutions found!")

# Print CSV file information
print(f"\n{'='*80}")
print("RESULTS SAVED TO CSV:")
print(f"File: {problem.csv_path}")
print(f"Total evaluations: {problem.evaluation_count}")
print("CSV contains configurations and objectives for each evaluation.")
print(f"{'='*80}")


