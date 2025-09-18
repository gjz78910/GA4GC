import numpy as np
from pymoo.core.problem import ElementwiseProblem
import copy
import time
import sys
import pandas as pd
import os
from datetime import datetime
from agentsRunner import AgentRunner

class ConfigProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=8,               # number of decision variables
            n_obj=3,               # number of objectives
            n_constr=0,            # no constraints for now
            xl=np.array([10, 3.0, 0.0, 0.1, 512,40,40,0]),   # lower bounds
            xu=np.array([40, 10.0, 1.0, 1.0, 4096,60,60,0])  # upper bounds
        )
        
        # Initialize CSV logging
        self.ga_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"results/ga_results_{self.ga_run_id}.csv"
        self.evaluation_count = 0
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Initialize CSV with headers
        self._initialize_csv()
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
                "model_name": "claude-3-5-sonnet-20241022",
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

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'evaluation_id', 'timestamp', 'agent_run_id',
            'config1_step_limit', 'config2_cost_limit', 'config3_temperature', 
            'config4_top_p', 'config5_max_tokens', 'config6_model_timeout',
            'config7_env_timeout', 'config8_baseline',
            'objective1_execution_time', 'objective2_model_improved', 'objective3_success',
            'status', 'error_message'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.csv_path, index=False)
        print(f"Initialized GA results CSV: {self.csv_path}")

    def _log_evaluation(self, x, objectives, status='success', error_message='', agent_run_id=''):
        """Log evaluation results to CSV"""
        self.evaluation_count += 1
        
        row_data = {
            'evaluation_id': self.evaluation_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'agent_run_id': agent_run_id,
            'config1_step_limit': int(x[0]),
            'config2_cost_limit': float(x[1]),
            'config3_temperature': float(x[2]),
            'config4_top_p': float(x[3]),
            'config5_max_tokens': int(x[4]),
            'config6_model_timeout': int(x[5]),
            'config7_env_timeout': int(x[6]),
            'config8_baseline': int(round(x[7]) + 1),
            'objective1_execution_time': objectives[0] if objectives else 0,
            'objective2_model_improved': -objectives[1] if objectives else 0,  # Convert back to positive
            'objective3_success': -objectives[2] if objectives else 0,  # Convert back to positive
            'status': status,
            'error_message': error_message
        }
        
        # Append to CSV
        df = pd.DataFrame([row_data])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        
        print(f"Logged evaluation {self.evaluation_count} to {self.csv_path} (agent_run_id: {agent_run_id})")

    
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
        print(f"\n{'='*80}")
        print(f"EVALUATING CONFIGURATION: {x}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        config_dict = self.genome_to_dict(x)
        print(f"Config dict generated: {config_dict}")
        
        config_file="template.yaml"
        baseline_file="../configurations/baseline" + str(round(x[7]) + 1) + ".yaml"
        print(f"Using baseline file: {baseline_file}")
        
        print("Creating AgentRunner...")
        agent=AgentRunner(config_file,config_dict)
        
        print("Rendering configuration...")
        final_config= agent.render(baseline_file)
        print(f"Final config rendered to: {final_config}")

        print("Starting agent execution...")
        start_time = time.time()
        try:
            result, agent_run_id = agent.runner(final_config)
            elapsed_time = time.time() - start_time 
            print(f"Agent execution completed in {elapsed_time:.2f} seconds")
            print("Result:")
            print(result)
            
            # Toy objectives
            f1 = elapsed_time 
            f2 = result.iloc[1]["model_improved"]
            f3 = result.iloc[1]["success"]
            
            print(f"Objectives: f1(time)={f1:.2f}, f2(model_improved)={f2}, f3(success)={f3}")
            objectives = [f1, -f2, -f3]
            out["F"] = objectives
            
            # Log successful evaluation with agent run_id
            self._log_evaluation(x, objectives, status='success', agent_run_id=agent_run_id)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"ERROR during agent execution after {elapsed_time:.2f} seconds: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            # Return poor objectives on failure
            objectives = [elapsed_time, -0.0, -0.0]
            out["F"] = objectives
            
            # Log failed evaluation (no agent_run_id available on failure)
            self._log_evaluation(x, objectives, status='failed', error_message=str(e), agent_run_id='N/A')
            
        print(f"{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}\n")
        sys.stdout.flush()

#def main():
#    out={}
#    genome = [40,100000,0.0,1.0,4096,600,600,1]
#    problem = ConfigProblem()
#    problem._evaluate(genome, out)
#
#main()
