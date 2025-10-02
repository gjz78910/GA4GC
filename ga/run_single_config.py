#!/usr/bin/env python3
"""
Run a single configuration using the EXACT same workflow as ga.py
This replicates the exact same timing and evaluation process used in the genetic algorithm.
Uses the same three commands (cmd1, cmd2, cmd3) as agentsRunner.py
"""

import os
import sys
import time
import re
import subprocess
import pandas as pd
from datetime import datetime

def run_single_configuration(config_file, slice_range="0:9"):
    """
    Run a single configuration using the EXACT same workflow as ga.py
    
    Args:
        config_file: Path to the configuration file (e.g., "default.yaml")
        slice_range: Slice of instances to process (e.g., "0:9" or "9:12")
    
    Returns:
        tuple: (result_dataframe, run_id, total_time)
    """
    
    print(f"\n{'='*80}")
    print(f"RUNNING SINGLE CONFIGURATION")
    print(f"Config file: {config_file}")
    print(f"Slice range: {slice_range}")
    print(f"{'='*80}")
    
    # Start timing (same as in problem.py)
    print("Starting agent execution...")
    start_time = time.time()
    
    try:
        # Use the local SWE-Perf files instead of the external project
        sweperf_script = os.path.abspath("../mini-swe-agent/src/minisweagent/run/extra/sweperf.py")
        
        # Command 1: Run sweperf.py (EXACT same as agentsRunner.py)
        cmd1 = [
            "python", sweperf_script,
            "--subset", "sweperf",
            "--split", "test",
            "--slice", slice_range,
            "--config", config_file,
            "--model", "gemini-v25-pro",
            "--workers", "30"
        ]
        print("CMD1:", " ".join(cmd1))
        
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

        # Extract with regex (EXACT same as agentsRunner.py)
        predictions_path = re.search(r"--predictions_path (\S+)", stdout).group(1)
        run_id = re.search(r"--run_id (\S+)", stdout).group(1)
        log_root = re.search(r"--log_root (\S+)", stdout).group(1)
        log_root = log_root.replace("/ollama/qwen3", "/ollama__qwen3")
        output_path = re.search(r"--output_path (\S+)", stdout).group(1)

        print("Predictions Path:", predictions_path)
        print("Run ID:", run_id)
        print("Log Root:", log_root)
        print("Output Path:", output_path)
        
        # Command 2: run evaluation (EXACT same as agentsRunner.py)
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
    
        # Command 3: check evaluation (EXACT same as agentsRunner.py)
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

        # Read the CSV (EXACT same as agentsRunner.py)
        df = pd.read_csv(output_path)
        
        # Select only model_improved and performance
        result = df[["model_improved", "performance","success"]]
        
        # Calculate total time
        elapsed_time = time.time() - start_time
        
        print(f"Agent execution completed in {elapsed_time:.2f} seconds")
        print("Result:")
        print(result)
        
        # Extract objectives (same as in problem.py)
        f1 = elapsed_time  # execution time
        f2 = result.iloc[1]["model_improved"]  # model improvement
        f3 = result.iloc[1]["success"]  # success rate
        
        print(f"Objectives: f1(time)={f1:.2f}, f2(model_improved)={f2}, f3(success)={f3}")
        
        return result, run_id, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR during agent execution after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e

def main():
    """Main function to run a single configuration"""
    
    if len(sys.argv) < 2:
        print("Usage: python run_single_config.py <config_file> [slice_range]")
        print("Examples:")
        print("  python run_single_config.py ../configurations/default.yaml")
        print("  python run_single_config.py ../configurations/pareto_eval4_config.yaml 9:12")
        sys.exit(1)
    
    config_file = sys.argv[1]
    slice_range = sys.argv[2] if len(sys.argv) > 2 else "0:9"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found!")
        sys.exit(1)
    
    try:
        # Run the configuration
        result, run_id, total_time = run_single_configuration(
            config_file=config_file,
            slice_range=slice_range
        )
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION COMPLETED SUCCESSFULLY")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Run ID: {run_id}")
        print(f"Results shape: {result.shape}")
        print(f"{'='*80}")
        
        # Save results to CSV (same format as GA results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/single_config_{timestamp}.csv"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        result.to_csv(output_file, index=False)
        print(f"Detailed results saved to: {output_file}")
        
        # Save summary results (same format as GA CSV)
        summary_data = {
            'evaluation_id': 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'agent_run_id': run_id,
            'config_file': config_file,
            'slice_range': slice_range,
            'objective1_execution_time': total_time,
            'objective2_model_improved': result.iloc[1]["model_improved"],
            'objective3_success': result.iloc[1]["success"],
            'status': 'success',
            'error_message': ''
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_file = f"results/single_config_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CONFIGURATION FAILED")
        print(f"Error: {str(e)}")
        print(f"{'='*80}")
        sys.exit(1)

if __name__ == "__main__":
    main()
