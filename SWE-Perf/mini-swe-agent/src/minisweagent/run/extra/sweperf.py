#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-Perf instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWE-Perf instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "sweperf": "SWE-Perf/SWE-Perf",  # Added SWE-Perf dataset
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        step_num = self.model.n_calls + 1
        cost = self.model.cost
        
        # Update progress manager
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {step_num:3d} (${cost:.2f})"
        )
        
        # Log step details to console so user can see what's happening
        logger.info(f"Instance {self.instance_id}: Starting Step {step_num} (${cost:.2f})")
        
        # Execute the step
        result = super().step()
        
        # Log step completion
        logger.info(f"Instance {self.instance_id}: Completed Step {step_num}")
        
        return result


# Removed Docker function - not needed for SWE-Perf


def get_or_clone_repository(instance: dict) -> str:
    """Get existing repository or clone if needed for a SWE-Perf instance."""
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    
    # Create persistent directory for this specific instance in current SWE-Perf directory
    base_dir = Path.cwd() / "cloned_repos"
    base_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = base_dir / instance_id
    
    try:
        # Check if repository already exists and is at the correct commit
        if repo_dir.exists() and (repo_dir / ".git").exists():
            logger.info(f"Found existing repository for {instance_id}, checking commit...")
            
            # Check current commit
            result = subprocess.run([
                "git", "rev-parse", "HEAD"
            ], cwd=repo_dir, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip() == base_commit:
                logger.info(f"Repository {instance_id} already at correct commit {base_commit}")
                return str(repo_dir)
            else:
                logger.info(f"Repository {instance_id} exists but needs commit update")
                # Try to checkout the correct commit
                result = subprocess.run([
                    "git", "checkout", base_commit
                ], cwd=repo_dir, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Updated {instance_id} to commit {base_commit}")
                    return str(repo_dir)
                else:
                    logger.info(f"Failed to checkout commit, will re-clone {instance_id}")
                    import shutil
                    shutil.rmtree(repo_dir, ignore_errors=True)
        
        # Clone the repository fresh
        logger.info(f"Cloning {repo} to {repo_dir}...")
        result = subprocess.run([
            "git", "clone", f"https://github.com/{repo}.git", str(repo_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        
        # Checkout the base commit
        logger.info(f"Checking out commit {base_commit}...")
        result = subprocess.run([
            "git", "checkout", base_commit
        ], cwd=repo_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        
        logger.info(f"Successfully prepared {repo} at commit {base_commit}")
        return str(repo_dir)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preparing repository {repo}: {e}")
        if e.stdout:
            logger.error(f"Git stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Git stderr: {e.stderr}")
        # Clean up failed clone
        if repo_dir.exists():
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)
        raise


def get_sweperf_environment(config: dict, instance: dict) -> Environment:
    """Create environment for SWE-Perf instance (uses local environment)."""
    env_config = config.get("environment", {}).copy()
    
    # Get or clone repository (reuses existing clones)
    repo_dir = get_or_clone_repository(instance)
    
    # Set working directory to the repository
    env_config["cwd"] = repo_dir
    env_config["environment_class"] = "local"  # Force local environment for SWE-Perf
    
    # Ensure we have a working directory
    if not os.path.exists(repo_dir):
        raise RuntimeError(f"Repository directory {repo_dir} does not exist")
    
    logger.info(f"Created SWE-Perf environment with working directory: {repo_dir}")
    return get_environment(env_config, default_type="local")


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSONL file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        # Read existing data if file exists
        existing_data = []
        if output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_data.append(json.loads(line))
        
        # Remove existing instance if it exists
        existing_data = [item for item in existing_data if item.get("instance_id") != instance_id]
        
        # Add new instance data
        new_instance = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        existing_data.append(new_instance)
        
        # Write back to JSONL file
        with open(output_path, 'w') as f:
            for item in existing_data:
                f.write(json.dumps(item) + '\n')





def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        # Read existing data
        existing_data = []
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("instance_id") != instance_id:
                        existing_data.append(data)
        
        # Write back without the removed instance
        with open(output_path, 'w') as f:
            for item in existing_data:
                f.write(json.dumps(item) + '\n')


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    preds_file: Path,
) -> None:
    """Process a single SWE-Perf instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(preds_file, instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    
    # Use problem_statement_oracle for SWE-Perf Oracle mode
    task = instance["problem_statement_oracle"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Setting up repository")

    agent = None
    extra_info = None

    try:
        logger.info(f"Setting up environment for instance {instance_id}")
        env = get_sweperf_environment(config, instance)
        
        logger.info(f"Creating agent for instance {instance_id}")
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        
        logger.info(f"Running agent on task: {task[:100]}...")
        
        # Add progress logging during execution
        logger.info(f"Instance {instance_id}: Starting agent execution...")
        exit_status, result = agent.run(task)
        logger.info(f"Instance {instance_id} completed with status: {exit_status}")
        logger.info(f"Instance {instance_id}: Result length: {len(str(result)) if result else 0}")
        
        # Fix: Extract the actual git diff from the result if it's a submission
        if exit_status == "Submitted" and result:
            # Look for git diff in the result
            if "diff --git" in str(result):
                # Extract clean git diff from submission text
                result_str = str(result)
                
                # Find the start of the git diff
                diff_start = result_str.find("diff --git")
                if diff_start != -1:
                    # Find the end - look for markers or end of text
                    remaining_text = result_str[diff_start:]
                    
                    # Look for end markers
                    end_markers = ["=== PATCH_END ===", "CLEANING REPOSITORY STATE", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
                    diff_end = len(remaining_text)
                    
                    for marker in end_markers:
                        marker_pos = remaining_text.find(marker)
                        if marker_pos != -1:
                            diff_end = min(diff_end, marker_pos)
                    
                    # Extract the clean git diff
                    clean_diff = remaining_text[:diff_end].strip()
                    result = clean_diff
                    logger.info(f"Instance {instance_id}: Extracted clean git diff, length: {len(result)}")
                else:
                    logger.warning(f"Instance {instance_id}: Found 'diff --git' but couldn't locate start position")
                    logger.info(f"Instance {instance_id}: Using full result, length: {len(str(result))}")
            else:
                # Try to extract git diff from the final command output
                logger.warning(f"Instance {instance_id}: No git diff found in result, attempting to extract from git")
                try:
                    # CRITICAL: Only capture changes to target files specified in patch_functions
                    target_files = []
                    if 'patch_functions' in instance:
                        try:
                            patch_functions = json.loads(instance['patch_functions'])
                            if isinstance(patch_functions, dict):
                                # Extract file paths from patch_functions
                                target_files = list(patch_functions.keys())
                                logger.info(f"Instance {instance_id}: Target files from patch_functions: {target_files}")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Instance {instance_id}: Failed to parse patch_functions: {e}")
                    
                    if target_files:
                        # Only stage and diff the target files
                        logger.info(f"Instance {instance_id}: Staging only target files: {target_files}")
                        
                        # Clean up any unwanted files first
                        cleanup_cmd = "git reset --hard HEAD && git clean -fdx"
                        env.run(cleanup_cmd)
                        
                        # Stage only the target files that have changes
                        for target_file in target_files:
                            if os.path.exists(target_file):
                                # Check if file has changes
                                status_result = env.run(f"git status --porcelain {target_file}")
                                if status_result.returncode == 0 and status_result.output.strip():
                                    # File has changes, stage it
                                    add_result = env.run(f"git add {target_file}")
                                    if add_result.returncode == 0:
                                        logger.info(f"Instance {instance_id}: Staged {target_file}")
                                    else:
                                        logger.warning(f"Instance {instance_id}: Failed to stage {target_file}")
                        
                        # Get diff for only staged files
                        git_result = env.run("git diff --cached")
                        if git_result.returncode == 0 and git_result.output:
                            result = git_result.output
                            logger.info(f"Instance {instance_id}: Extracted targeted git diff, length: {len(result)}")
                        else:
                            logger.warning(f"Instance {instance_id}: No changes found in target files")
                    else:
                        # Fallback: use the old method but with better filtering
                        logger.warning(f"Instance {instance_id}: No target files found, using fallback method")
                        
                        # Clean up unwanted files first
                        cleanup_cmd = "git reset --hard HEAD && git clean -fdx"
                        env.run(cleanup_cmd)
                        
                        # Get list of modified files, excluding common unwanted patterns
                        status_cmd = "git status --porcelain | grep -E '^M|^A|^D' | awk '{print $2}' | grep -v -E '\\.(log|tmp|bak|orig|rej|patch|diff|pyc)$' | grep -v -E '(test_.*|.*_test|benchmark_.*|__pycache__)'"
                        status_result = env.run(status_cmd)
                        
                        if status_result.returncode == 0 and status_result.output.strip():
                            # Stage only the filtered files
                            files_to_stage = status_result.output.strip().split('\n')
                            for file_path in files_to_stage:
                                if file_path.strip():
                                    env.run(f"git add {file_path}")
                            
                            # Get diff for staged files
                            git_result = env.run("git diff --cached")
                            if git_result.returncode == 0 and git_result.output:
                                result = git_result.output
                                logger.info(f"Instance {instance_id}: Extracted filtered git diff, length: {len(result)}")
                            else:
                                logger.warning(f"Instance {instance_id}: No git diff found after filtering")
                        else:
                            logger.warning(f"Instance {instance_id}: No modified files found after filtering")
                            
                except Exception as e:
                    logger.error(f"Instance {instance_id}: Failed to extract git diff: {e}")
        
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(preds_file, instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)
        
        # Note: Repository directories are kept for reuse in future runs


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWE-Perf instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("sweperf", "--subset", help="SWE-Perf subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "sweperf_oracle.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are local for SWE-Perf", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    # Always use current working directory's results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped prediction file name (JSONL only)
    preds_file = output_path / f"sweperf_predictions_{timestamp}.jsonl"
    
    logger.info(f"Results will be saved to {output_path}")
    logger.info(f"Predictions will be saved to {preds_file}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}...")
    # SWE-Perf returns a DatasetDict, we need to access the 'test' split
    if subset == "sweperf":
        dataset = load_dataset(dataset_path)
        instances = list(dataset['test'])
    else:
        instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and preds_file.exists():
        # Read existing instances from JSONL file
        existing_instances = []
        with open(preds_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_instances.append(data.get("instance_id"))
        
        if existing_instances:
            logger.info(f"Skipping {len(existing_instances)} existing instances")
            instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")
    
    # Debug: Print first instance to verify data structure
    if instances:
        logger.info(f"First instance ID: {instances[0]['instance_id']}")
        logger.info(f"First instance keys: {list(instances[0].keys())[:5]}")


    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager, preds_file): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    logger.info(f"SWE-Perf predictions saved to {preds_file}")
    
    # Print evaluation commands for user convenience
    print("\n" + "="*80)
    print("🎯 SWE-PERF EVALUATION COMMANDS")
    print("="*80)
    print(f"# Step 1: Run Evaluation")
    print(f"python -m evaluation.run_evaluation --dataset_name SWE-Perf/SWE-Perf --split test --predictions_path {preds_file} --max_workers 1 --run_id mini_swe_agent_{timestamp}")
    print()
    # Get model name from config for the evaluation command
    model_name = config.get("model", {}).get("model_name")
    if not model_name:
        raise ValueError("Model name not found in config file")
    print(f"# Step 2: Check Evaluation")
    print(f"python -m evaluation.check_evaluation --dataset_dir SWE-Perf/SWE-Perf --log_root results/mini_swe_agent_{timestamp}/{model_name} --output_path results/performance_metrics_{timestamp}.csv")
    print("="*80)
    print("📝 Note: Both commands automatically change to the correct directory")
    print("="*80)


if __name__ == "__main__":
    app()