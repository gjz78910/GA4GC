import django
from django.conf import settings
from django.template.loader import render_to_string
import os
from datetime import datetime
import subprocess
import pandas as pd
import re 
class AgentRunner():

    def __init__(self,config_file,config_dict):
        self.config_dict = config_dict
        self.config_file = config_file
        
        if not settings.configured:
            settings.configure(
                TEMPLATES=[
                    {
                        "BACKEND": "django.template.backends.django.DjangoTemplates",
                        "DIRS": [".",".."],  # look for templates in current dir
                        "APP_DIRS": True,
                    }
                ]
            )
        django.setup()

    def render(self, baseline_file):
        yaml_text = render_to_string(self.config_file, {"config": self.config_dict})
        # Read baseline file
        with open(baseline_file, "r") as f:
            baseline_content = f.read()
        # Concatenate baseline + rendered yaml
        combined_content = baseline_content.strip() + "\n\n" + yaml_text.strip()
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build new file path (e.g., baseline.yaml -> baseline_rendered.yaml)
        base, ext = os.path.splitext(baseline_file)
        new_file = f"{base}_rendered_{timestamp}{ext or '.yaml'}"
        print(new_file)
        # Write out combined file
        with open(new_file, "w") as f:
            f.write(combined_content)

        return new_file

    def runner(self, config_file):
        cmd1 = [
            "sudo", "-E" ,"env" , f"PATH={os.environ['PATH']}",
            "python3", "-m", "minisweagent.run.extra.sweperf",
            "--subset", "sweperf",
            "--split", "test",
            "--slice", "0:5",
            "--config", config_file,
            "--workers", "30"
        ]
        print(cmd1)
        result = subprocess.run(cmd1, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        stdout = result.stdout

        # Extract with regex
        predictions_path = re.search(r"--predictions_path (\S+)", stdout).group(1)
        run_id = re.search(r"--run_id (\S+)", stdout).group(1)
        log_root = re.search(r"--log_root (\S+)", stdout).group(1)
        log_root = log_root.replace("/ollama/qwen3", "/ollama__qwen3")
        output_path = re.search(r"--output_path (\S+)", stdout).group(1)

        print("Predictions Path:", predictions_path)
        print("Run ID:", run_id)
        print("Log Root:", log_root)
        print("Output Path:", output_path)
        # Command 2: run evaluation
        # Change working directory to ../SWE-Perf
        eval_dir = os.path.abspath("../SWE-Perf")

        cmd2 = [
            "sudo", "-E" ,"env" , f"PATH={os.environ['PATH']}",
            "python3", "-m", "evaluation.run_evaluation",
            "--dataset_name", "SWE-Perf/SWE-Perf",
            "--split", "test",
            "--predictions_path", predictions_path,
            "--max_workers", "30",
            "--run_id", run_id
        ]
    
        subprocess.run(cmd2, cwd=eval_dir, check=True)
    
        cmd3 = [
            "python3", "-m", "evaluation.check_evaluation",
            "--dataset_dir", "SWE-Perf/SWE-Perf",
            "--log_root", log_root,
            "--output_path", output_path
        ]    
        subprocess.run(cmd3, cwd=eval_dir, check=True)

        # Read the CSV
        df = pd.read_csv(output_path)
        
        # Select only model_improved and performance
        result = df[["model_improved", "performance","success"]]
        return result



        
# def main():

#     config_dict = {
#         "step_limit": 40,
#         "cost_limit": 3.0,
#         "environment": {
#             "timeout": 60,
#             "env": {
#                 "PAGER": "cat",
#                 "MANPAGER": "cat",
#                 "LESS": "-R",
#                 "PIP_PROGRESS_BAR": "off",
#                 "TQDM_DISABLE": "1",
#             },
#             "environment_class": "local",
#         },
#         "model": {
#             "model_name": "ollama/qwen3",
#             "model_kwargs": {
#                 "temperature": 0.0,
#                 "top_p": 1.0,
#                 "top_k": -1,
#                 "max_tokens": 4096,
#                 "stop": None,  # translates to YAML "null"
#                 "frequency_penalty": 0.0,
#                 "presence_penalty": 0.0,
#                 "timeout": 60,
#                 "drop_params": True,
#             },
#         },
#     }
#     config_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/template.yaml"
#     baseline_file="/home/ubuntu/dataAvner/ssbseAgents/GreenerAgent/baseline.yaml"
#     agent=AgentRunner(config_file,config_dict)
#     final_config= agent.render(baseline_file)
#     result = agent.runner(final_config)
#     print(result)
    
# main()
