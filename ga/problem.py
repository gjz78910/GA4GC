import numpy as np
from pymoo.core.problem import ElementwiseProblem
import copy
import time
from agentsRunner import AgentRunner

class ConfigProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=8,               # number of decision variables
            n_obj=3,               # number of objectives
            n_constr=0,            # no constraints for now
            xl=np.array([10, 3.0, 0.0, 0.1, 512,40,40,1]),   # lower bounds
            xu=np.array([40, 10.0, 1.0, 1.0, 4096,60,60,3])  # upper bounds
        )
        self.base_config = {
            "step_limit": 40,
            "cost_limit": 1000000,
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
        cfg["model"]["model_kwargs"]["timeout"] = int(genome[5])
        cfg["environment"]["timeout"] = int(genome[6])
        return cfg

        
    def _evaluate(self, x, out, *args, **kwargs):
        config_dict = self.genome_to_dict(x)
        config_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/template.yaml"
        baseline_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/baseline" + str(round(x[7])) + ".yaml"
        agent=AgentRunner(config_file,config_dict)
        final_config= agent.render(baseline_file)

        start_time = time.time()
        result = agent.runner(final_config)
        elapsed_time = time.time() - start_time 
        print(result)
        # Toy objectives
        f1 = elapsed_time 
        f2 = result.iloc[1]["model_improved"]
        f3 = result.iloc[1]["success"]
        
        out["F"] = [f1, -f2, -f3]

#def main():
#    out={}
#    genome = [40,100000,0.0,1.0,4096,600,600,1]
#    problem = ConfigProblem()
#    problem._evaluate(genome, out)
#
#main()
