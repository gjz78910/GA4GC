from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from problem import ConfigProblem
import copy

# Instantiate problem
problem = ConfigProblem()

# Configure NSGA2
algorithm = NSGA2(
    pop_size=2,
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 2)  # run for 30 generations

# Run optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)


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
best_config = genome_to_dict(res.X[0], your_config_dict)
print("\nBest Config Dict Example:\n", best_config)


