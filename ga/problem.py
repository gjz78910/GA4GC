import numpy as np
from pymoo.core.problem import ElementwiseProblem
import copy
from agentsRunner import AgentRunner

class ConfigProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=5,               # number of decision variables
            n_obj=2,               # number of objectives
            n_constr=0,            # no constraints for now
            xl=np.array([10, 0.1, 0.0, 0.1, 512]),   # lower bounds
            xu=np.array([100, 5.0, 1.0, 1.0, 4096])  # upper bounds
        )
        self.base_config = {
            "step_limit": 40,
            "cost_limit": 3.0,
            "environment": {
                "timeout": 60,
                "env": {
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "LESS": "-R",
                    "PIP_PROGRESS_BAR": "off",
                    "TQDM_DISABLE": "1",
                },
                "environment_class": "local",
            },
            "model": {
                "model_name": "ollama/qwen3",
                "model_kwargs": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 4096,
                    "stop": None,  # translates to YAML "null"
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "timeout": 60,
                    "drop_params": True,
                },
            },
        }


    
    def genome_to_dict(self,genome):
        cfg = copy.deepcopy(self.base_config)
        cfg["step_limit"] = int(genome[0])
        cfg["cost_limit"] = float(genome[1])
        cfg["model"]["model_kwargs"]["temperature"] = float(genome[2])
        cfg["model"]["model_kwargs"]["top_p"] = float(genome[3])
        cfg["model"]["model_kwargs"]["max_tokens"] = int(genome[4])
        return cfg

        
    def _evaluate(self, x, out, *args, **kwargs):
        config_dict = self.genome_to_dict(x)
        config_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/template.yaml"
        baseline_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/baseline.yaml"
        agent=AgentRunner(config_file,config_dict)
        final_config= agent.render(baseline_file)
        result = agent.runner(final_config)

        # Toy objectives
        f1 = result[0]
        f2 = result[1]

        out["F"] = [f1, f2]

# def main():
#     out={}
#     genome = [40,3.0,0.0,1.0,4096]
#     problem = ConfigProblem()
#     problem._evaluate(genome, out)

# main()
