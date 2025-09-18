from __future__ import annotations

import docker
import json
import resource
import traceback

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from functools import partial


from .constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from .docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
    copy_from_container,
)
from .docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from .test_spec import make_test_spec, TestSpec, REPEAT_TIME
from .utils import load_sweperf_dataset, str2bool

import os
from collections import defaultdict


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )


def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
        sweperf_instance: dict = None,
    ):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            # some error, idk why
            pass
    log_file = log_dir / "run_instance.log"

    # Set up report file + logger
    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild, is_large = False, from_remote = False)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        
        # Extract only the git diff portion from model_patch, removing cleanup messages
        from .utils import extract_minimal_patch
        model_patch = pred["model_patch"] or ""
        
        # Look for the clear patch markers first (new approach)
        if "=== PATCH_START ===" in model_patch and "=== PATCH_END ===" in model_patch:
            # Use the clear markers for reliable extraction
            patch_start = model_patch.find("=== PATCH_START ===")
            patch_end = model_patch.find("=== PATCH_END ===")
            
            if patch_start != -1 and patch_end != -1 and patch_end > patch_start:
                # Extract everything between the markers
                clean_patch = model_patch[patch_start + len("=== PATCH_START ==="):patch_end].strip()
                
                # Additional validation: ensure patch starts with "diff --git"
                if not clean_patch.startswith("diff --git"):
                    logger.warning(f"Patch doesn't start with 'diff --git' for {instance_id}, searching for it")
                    diff_start = clean_patch.find("diff --git")
                    if diff_start != -1:
                        clean_patch = clean_patch[diff_start:]
                        logger.info(f"Found 'diff --git' at position {diff_start} for {instance_id}")
                    else:
                        logger.error(f"No 'diff --git' found in extracted patch for {instance_id}")
                        clean_patch = ""
                
                # Validate patch ends with proper @@ boundary
                if clean_patch and not clean_patch.rstrip().endswith("---"):
                    # Find the last complete @@ block
                    last_at_at = clean_patch.rfind("@@")
                    if last_at_at != -1:
                        # Find the end of this @@ block (next @@ or end of string)
                        next_at_at = clean_patch.find("@@", last_at_at + 2)
                        if next_at_at != -1:
                            # Find the end of the current block (look for end of hunk)
                            lines = clean_patch[next_at_at:].split('\n')
                            for i, line in enumerate(lines):
                                if line.startswith('diff --git') or line.startswith('---'):
                                    clean_patch = clean_patch[:next_at_at + sum(len(l) + 1 for l in lines[:i])]
                                    break
                        else:
                            # No more @@ blocks, this is the end
                            clean_patch = clean_patch[:last_at_at + clean_patch[last_at_at:].find('\n', clean_patch[last_at_at:].find('\n') + 1)]
                
                logger.info(f"Extracted patch using clear markers for {instance_id}, length: {len(clean_patch)}")
            else:
                # Fallback to old method
                clean_patch = model_patch
                logger.warning(f"Invalid patch markers for {instance_id}, using full model_patch")
        elif "diff --git" in model_patch:
            # Fallback to old method for backward compatibility
            diff_start = model_patch.find("diff --git")
            clean_patch = model_patch[diff_start:]
            
            # Enhanced cleanup marker detection
            cleanup_markers = [
                "CLEANING REPOSITORY STATE",
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                "CAPTURING CHANGES",
                "Files to be committed:",
                "GENERATING PATCH:",
                "AGGRESSIVE CLEANUP STARTING",
                "VERIFYING CLEAN STATE:",
                "HEAD is now at"
            ]
            
            earliest_cleanup = len(clean_patch)
            for marker in cleanup_markers:
                marker_pos = clean_patch.find(marker)
                if marker_pos != -1 and marker_pos < earliest_cleanup:
                    earliest_cleanup = marker_pos
            
            if earliest_cleanup < len(clean_patch):
                clean_patch = clean_patch[:earliest_cleanup].strip()
            
            logger.info(f"Extracted patch using fallback method for {instance_id}, length: {len(clean_patch)}")
        else:
            clean_patch = model_patch
            logger.warning(f"No git diff or patch markers found in model_patch for {instance_id}")
        
        # Final validation: ensure patch is reasonable size (not too large)
        if len(clean_patch) > 1000000:  # 1MB limit
            logger.error(f"Patch too large ({len(clean_patch)} bytes) for {instance_id}, likely corrupted")
            # Try to extract just the first few hunks
            lines = clean_patch.split('\n')
            limited_patch = []
            hunk_count = 0
            max_hunks = 10  # Limit to 10 hunks
            
            for line in lines:
                if line.startswith('@@'):
                    hunk_count += 1
                    if hunk_count > max_hunks:
                        break
                limited_patch.append(line)
            
            clean_patch = '\n'.join(limited_patch)
            logger.info(f"Limited patch to {len(clean_patch)} bytes ({max_hunks} hunks) for {instance_id}")
        
        # CRITICAL VALIDATION: Reject patches that delete documentation or license files
        if clean_patch:
            # Check for documentation deletion
            if "deleted file mode" in clean_patch:
                deleted_files = []
                lines = clean_patch.split('\n')
                for line in lines:
                    if line.startswith('diff --git') and 'b/' in line:
                        # Extract filename from diff line
                        parts = line.split('b/')
                        if len(parts) > 1:
                            filename = parts[1].strip()
                            deleted_files.append(filename)
                
                # Check for forbidden deletions
                forbidden_patterns = [
                    'docs/', 'licenses/', '*.rst', '*.md', '*.txt', '*.inc',
                    'README', 'CHANGELOG', 'CONTRIBUTING', 'LICENSE'
                ]
                
                for filename in deleted_files:
                    for pattern in forbidden_patterns:
                        if pattern in filename or filename.endswith(tuple(pattern.split(',')[2:])):  # Skip 'docs/' and 'licenses/'
                            logger.error(f"Patch attempts to delete forbidden file: {filename} for {instance_id}")
                            clean_patch = ""
                            break
                    if not clean_patch:
                        break
                
                if not clean_patch:
                    logger.error(f"Rejected patch for {instance_id} due to forbidden file deletions")
        
        # Ensure patch ends with newline
        if clean_patch and not clean_patch.endswith('\n'):
            clean_patch += '\n'
        
        patch_file.write_text(clean_patch)
        logger.info(
            f"Clean patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        # Get base performance from SWE-Perf dataset instead of profiling
        logger.info(f"Reading base performance from SWE-Perf dataset for {instance_id}")
        base_performance_data = {}
        if sweperf_instance and "duration_changes" in sweperf_instance:
            duration_changes = sweperf_instance["duration_changes"]
            efficiency_tests = sweperf_instance.get("efficiency_test", [])
            
            # Extract base performance for each test from the dataset
            for test_name in efficiency_tests:
                base_durations = []
                for duration_change in duration_changes:
                    if test_name in duration_change and "base" in duration_change[test_name]:
                        base_durations.append(duration_change[test_name]["base"])
                
                if base_durations:
                    base_performance_data[test_name] = base_durations
                    logger.info(f"Found base performance for test {test_name}: {len(base_durations)} measurements")
                else:
                    logger.warning(f"No base performance data found for test {test_name}")
        else:
            logger.warning(f"No SWE-Perf dataset instance provided for {instance_id}")

        # Create eval script for model performance profiling
        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))
        
        # DEBUG: Add diagnostic commands to see what's actually in the container
        logger.info("=== DEBUGGING REPOSITORY STRUCTURE ===")
        
        # Check working directory and basic structure
        pwd_result = container.exec_run("pwd", workdir="/testbed", user="root")
        logger.info(f"Working directory: {pwd_result.output.decode('utf-8').strip()}")
        
        # List all files in /testbed
        ls_result = container.exec_run("ls -la", workdir="/testbed", user="root")
        logger.info(f"Files in /testbed:\n{ls_result.output.decode('utf-8')}")
        
        # Check git status
        git_status_result = container.exec_run("git status", workdir="/testbed", user="root")
        logger.info(f"Git status:\n{git_status_result.output.decode('utf-8')}")
        
        # Check git ls-files to see what files git knows about
        git_ls_result = container.exec_run("git ls-files | head -20", workdir="/testbed", user="root")
        logger.info(f"Git tracked files (first 20):\n{git_ls_result.output.decode('utf-8')}")
        
        # Check if the specific file exists
        file_check_result = container.exec_run("find . -name 'diff.py' -type f", workdir="/testbed", user="root")
        logger.info(f"Files named 'diff.py':\n{file_check_result.output.decode('utf-8')}")
        
        # Check if the directory structure exists (generic approach)
        dir_check_result = container.exec_run("find . -type d -maxdepth 2 | head -10", workdir="/testbed", user="root")
        logger.info(f"Repository structure (first 10 dirs):\n{dir_check_result.output.decode('utf-8')}")
        
        # Check the actual file we're trying to patch (generic approach)
        target_file_check = container.exec_run("find . -name '*.py' -type f | head -20", workdir="/testbed", user="root")
        logger.info(f"Python files found (first 20):\n{target_file_check.output.decode('utf-8')}")
        
        logger.info("=== END DEBUGGING ===")
        
        # Attempt to apply patch to container
        val = container.exec_run(
            "git apply --allow-empty -v /tmp/patch.diff",
            workdir="/testbed",
            user="root",
        )
        if val.exit_code != 0:
            logger.info(f"Failed to apply patch to container, trying again...")
            
            # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
            val = container.exec_run(
                "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                workdir="/testbed",
                user="root",
            )
            if val.exit_code != 0:
                logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
                raise EvaluationError(
                    instance_id,
                    f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}",
                    logger,
                )
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        # eval_file = Path(log_dir / "eval.sh")
        # eval_file.write_text(test_spec.eval_script)
        # logger.info(
        #     f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        # )
        # copy_to_container(container, eval_file, Path("/eval.sh"))

        # Run eval script, write output to logs
        test_output_path = log_dir / "test_output.txt"
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout, test_output_path)
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        # with open(test_output_path, "w") as f:
        #     f.write(test_output)
        #     logger.info(f"Test output for {instance_id} written to {test_output_path}")
        if timed_out:
            # f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise EvaluationError(
                instance_id,
                f"Test timed out after {timeout} seconds.",
                logger,
            )

        os.makedirs(log_dir / "report", exist_ok=True)
        for re_idx in range(REPEAT_TIME):
            local_path = log_dir / "report" / f"report{re_idx}.json"
            container_path = f"/testbed/report{re_idx}.json"
            try:
                copy_from_container(container, container_path, local_path)
            except:
                logger.info(f"Failed to copy report from container for {instance_id}, continuing...")
                # continue
        container.exec_run(
            "rm -rf /testbed/report*.json",
            workdir="/testbed",
            user="root",
        )


        # Get git diff after running eval script
        git_diff_output_after = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        report = defaultdict(lambda: defaultdict(dict))
        
        # Add base performance from dataset
        for test_name, durations in base_performance_data.items():
            for idx, duration in enumerate(durations):
                report[test_name]["base"][idx] = {"outcome": "passed", "duration": duration}
        
        # Add model performance from profiling
        missing_reports = []
        available_reports = []
        
        for idx in range(REPEAT_TIME):
            model_path = log_dir / f"report/report{idx}.json"
            if model_path.exists():
                try:
                    with open(model_path, "r") as f:
                        model_report = json.load(f)
                    for t in model_report['tests']:
                        report[t['nodeid']]["model"][idx] = {"outcome": t["outcome"], "duration": t["call"]["duration"]}
                    available_reports.append(idx)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse report {idx} for {instance_id}: {e}")
                    missing_reports.append(idx)
            else:
                missing_reports.append(idx)
        
        # Report on missing reports and continue with available data
        if missing_reports:
            print(f"Warning: {len(missing_reports)}/{REPEAT_TIME} reports missing for {instance_id}: {missing_reports}")
            if len(available_reports) == 0:
                print(f"Error: No valid reports available for {instance_id}, skipping...")
                logger.error(f"No valid reports available for {instance_id}, instance will be skipped")
                return
            else:
                print(f"Continuing with {len(available_reports)} available reports: {available_reports}")
                logger.info(f"Using {len(available_reports)}/{REPEAT_TIME} reports for {instance_id}")
        
        # Write report to report.json if we have any data
        if available_reports:
            logger.info(
                f"report: {report}\n"
            )
            
            # Write report to report.json
            with open(report_path, "w") as f:
                f.write(json.dumps(report, indent=4))
            
            print(f"Successfully processed {instance_id} with {len(available_reports)}/{REPEAT_TIME} reports")
        else:
            print(f"Failed to process {instance_id}: no valid reports available")
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    func = partial(make_test_spec, is_eval=True)
    test_specs = list(map(func, instances))
    
    # Create mapping from instance_id to full instance data for dataset access
    instance_data_map = {instance[KEY_INSTANCE_ID]: instance for instance in instances}

    # Note: instance_image_key is now a read-only property that automatically
    # calculates the correct value based on repo and version for optimization
    # No need to manually set it - the TestSpec class handles this automatically

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                    instance_data_map.get(test_spec.instance_id),
                ): None
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")


def get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        run_id: str,
        exclude_completed: bool = True
    ):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_sweperf_dataset(dataset_name, split)
    print(f"There are {len(dataset)} samples in dataset.")
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
    
    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        # raise ValueError( # TODO
        print(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    # instance_ids = ["sympy__sympy-25982"] # TODO
    # instance_ids = ["sympy__sympy-25852"]
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]
    
    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v["model_patch"] == "" or v["model_patch"] is None}

    # filter dataset to only instances with predictions
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]
    return dataset


def get_gold_predictions(dataset_name: str, split: str):
    """
    Get gold predictions for the given dataset and split.
    """
    dataset = load_sweperf_dataset(dataset_name, split)
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "model_patch": datum["patch"],
            "model_name_or_path": "gold",
        } for datum in dataset
    ]


def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # load predictions as map of instance_id to prediction
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)
    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="SWE-Perf/SWE-Perfs", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default=None, help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
