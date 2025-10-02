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
        # Read baseline file with comprehensive encoding handling
        baseline_content = None
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(baseline_file, "r", encoding=encoding) as f:
                    baseline_content = f.read()
                print(f"Successfully read baseline file with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if baseline_content is None:
            raise ValueError(f"Could not read baseline file {baseline_file} with any encoding")
        
        # Clean up common encoding issues
        baseline_content = baseline_content.replace('"', '"').replace('"', '"')
        baseline_content = baseline_content.replace(''', "'").replace(''', "'")
        baseline_content = baseline_content.replace('–', '-').replace('—', '--')
        baseline_content = baseline_content.replace('…', '...')
        baseline_content = baseline_content.replace('→', '->')  # Fix Unicode arrow
        baseline_content = baseline_content.replace('←', '<-')  # Fix Unicode arrow
        baseline_content = baseline_content.replace('↑', '^')   # Fix Unicode arrow
        baseline_content = baseline_content.replace('↓', 'v')   # Fix Unicode arrow
        baseline_content = baseline_content.replace('🎯', '[TARGET]')  # Fix emoji
        baseline_content = baseline_content.replace('🔍', '[SEARCH]')  # Fix emoji
        baseline_content = baseline_content.replace('✅', '[OK]')      # Fix emoji
        baseline_content = baseline_content.replace('❌', '[ERROR]')   # Fix emoji
        
        # Concatenate baseline + rendered yaml
        combined_content = baseline_content.strip() + "\n\n" + yaml_text.strip()
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build new file path (e.g., baseline.yaml -> baseline_rendered.yaml)
        base, ext = os.path.splitext(baseline_file)
        new_file = f"{base}_rendered_{timestamp}{ext or '.yaml'}"
        print(new_file)
        # Write out combined file with UTF-8 BOM to ensure proper encoding
        with open(new_file, "w", encoding="utf-8-sig") as f:
            f.write(combined_content)

        return new_file

    def runner(self, config_file):
        # Use the local SWE-Perf files instead of the external project
        sweperf_script = os.path.abspath("../mini-swe-agent/src/minisweagent/run/extra/sweperf.py")
        
        cmd1 = [
            "python", sweperf_script,
            "--subset", "sweperf",
            "--split", "test",
            "--slice", "0:9",
            "--config", config_file,
            "--model", "gemini-v25-pro",
            # "--model", "claude-3-5-sonnet-20241022",
            "--workers", "30"
        ]
        print(cmd1)
        
        # Set environment variables to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run(cmd1, capture_output=True, text=True, env=env, errors='replace')
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print("="*80)
            print("ERROR SUMMARY:")
            print("="*80)
            
            # Show last 500 characters of stderr to see the actual error
            stderr_short = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            print("STDERR (last 500 chars):", stderr_short)
            
            # Look for specific error patterns
            if "UnicodeDecodeError" in result.stderr:
                print("\n🔍 DIAGNOSIS: Unicode encoding error detected!")
                print("The generated YAML file contains special characters that Windows can't read.")
            elif "ValidationError" in result.stderr:
                print("\n🔍 DIAGNOSIS: Model configuration error detected!")
                print("Missing API keys or model configuration issues.")
            elif "ImportError" in result.stderr:
                print("\n🔍 DIAGNOSIS: Import error detected!")
                print("Missing dependencies or module issues.")
            
            print("="*80)
            raise subprocess.CalledProcessError(result.returncode, cmd1, result.stdout, result.stderr)
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
        eval_dir = os.path.abspath("./")

        cmd2 = [
            "python", "-m", "evaluation.run_evaluation",
            "--dataset_name", "SWE-Perf/SWE-Perf",
            "--split", "test",
            "--predictions_path", predictions_path,
            "--max_workers", "30",
            "--run_id", run_id
        ]
    
        print(f"Running evaluation command: {' '.join(cmd2)}")
        print(f"Working directory: {eval_dir}")
        result2 = subprocess.run(cmd2, cwd=eval_dir, capture_output=True, text=True, env=env)
        print("EVALUATION STDOUT:", result2.stdout)
        if result2.stderr:
            print("EVALUATION STDERR:", result2.stderr)
        if result2.returncode != 0:
            print(f"Evaluation failed with return code {result2.returncode}")
            raise subprocess.CalledProcessError(result2.returncode, cmd2, result2.stdout, result2.stderr)
    
        cmd3 = [
            "python", "-m", "evaluation.check_evaluation",
            "--dataset_dir", "SWE-Perf/SWE-Perf",
            "--log_root", log_root,
            "--output_path", output_path
        ]    
        print(f"Running check evaluation command: {' '.join(cmd3)}")
        result3 = subprocess.run(cmd3, cwd=eval_dir, capture_output=True, text=True, env=env)
        print("CHECK EVALUATION STDOUT:", result3.stdout)
        if result3.stderr:
            print("CHECK EVALUATION STDERR:", result3.stderr)
        if result3.returncode != 0:
            print(f"Check evaluation failed with return code {result3.returncode}")
            raise subprocess.CalledProcessError(result3.returncode, cmd3, result3.stdout, result3.stderr)

        # Read the CSV
        df = pd.read_csv(output_path)
        
        # Select only model_improved and performance
        result = df[["model_improved", "performance","success"]]
        
        # Return results along with run_id
        return result, run_id



        
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
