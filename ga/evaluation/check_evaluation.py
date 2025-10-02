import json
import os
import pandas as pd
from argparse import ArgumentParser
from .utils import filter_outliers, find_max_significant_improvement, load_sweperf_dataset


def calculate_performance_result(sweperf_data, log_root):
    without_prediction = 0
    without_run = 0
    run_failed = 0
    human_improved = []
    model_improved = []
    human_total = []

    for _, data in sweperf_data.iterrows():
        id = data['instance_id']
        duration_changes = data["duration_changes"]
        efficiency_test = data["efficiency_test"]
        # calculate human total
        ht = []
        for test in efficiency_test:
            duration_change_base = filter_outliers([duration_change[test]["base"] for duration_change in duration_changes])
            duration_change_head = filter_outliers([duration_change[test]["head"] for duration_change in duration_changes])
            ht.append(find_max_significant_improvement(duration_change_head, duration_change_base))
        human_total.append(sum(ht)/len(ht))

        # calculate model total
        if not os.path.exists(os.path.join(log_root, id, "run_instance.log")):
            without_prediction += 1
            continue
        if not os.path.exists(os.path.join(log_root, id, "report.json")):
            without_run += 1
            continue

        with open(os.path.join(log_root, id, "report.json"), "r") as file:
            report = json.load(file)
        
        # Parse measured performance data from report file
        durations_model = {}  # Model patch performance (measured)
        results_model = {}    # Model patch test outcomes (measured)
        durations_base = {}   # Baseline performance (measured)
        results_base = {}     # Baseline test outcomes (measured)
        
        for test in report.keys():
            if "base" in report[test]:
                durations_base_ = [rep["duration"] for rep in report[test]["base"].values()]
                durations_base_ = filter_outliers(durations_base_)
                durations_base[test] = durations_base_

                results_base_ = [rep["outcome"] for rep in report[test]["base"].values()]
                results_base[test] = set(results_base_) == {"passed"}
            if "model" in report[test]:
                durations_model_ = [rep["duration"] for rep in report[test]["model"].values()]
                durations_model_ = filter_outliers(durations_model_)
                durations_model[test] = durations_model_
                results_model[test] = set([rep["outcome"] for rep in report[test]["model"].values()]) == {"passed"}
                
        if not durations_model:
            run_failed += 1
            continue

        # Calculate improvements using both dataset (human) and measured (model) data
        human_improvements = []  # Human patch improvement (from dataset)
        model_improvements = []  # Model patch improvement (from measurements)
        
        for test in efficiency_test:
            # Calculate human improvement using dataset values (base vs head) - ALWAYS available
            duration_change_base = filter_outliers([duration_change[test]["base"] for duration_change in duration_changes])
            duration_change_head = filter_outliers([duration_change[test]["head"] for duration_change in duration_changes])
            human_improvement = find_max_significant_improvement(duration_change_head, duration_change_base)
            human_improvements.append(human_improvement)
            
            # Model improvement calculation - only if test exists in evaluation results
            if test in durations_model and test in durations_base and results_base[test] and results_model[test]:
                model_improvement = find_max_significant_improvement(durations_model[test], durations_base[test])
                model_improvements.append(model_improvement)
        
        # Add improvements if we have data
        # Add human improvement (always available from dataset)
        if human_improvements:
            human_improved.append(sum(human_improvements)/len(human_improvements))
        
        # Add model improvement (only if we have model data for at least one test)
        if model_improvements:
            model_improved.append(sum(model_improvements)/len(model_improvements))

    with_prediction = len(sweperf_data) - without_prediction
    total = len(sweperf_data)
    print(f"There are {len(sweperf_data)} data, {without_prediction} without prediction and {with_prediction} with prediction. ")
    print(f"There are {without_run/total} ({without_run}/{total}) failed patch, {(with_prediction - without_run)/total} ({with_prediction - without_run}/{total}) success patch")
    print(f"There are {run_failed/total} ({run_failed}/{total}) failed run, {(with_prediction - without_run - run_failed)/total} ({with_prediction - without_run - run_failed}/{total}) success run")
    print(f"Model efficiency improved: {sum(model_improved)/total}")
    print(f"Human efficiency improved: {sum(human_improved)/total}")
    print(f"Human total efficiency improved: {sum(human_total)/total}")
    return {
        "model": log_root,
        "total": total,
        "with_prediction": with_prediction,
        "with_run": with_prediction - without_run,
        "success": with_prediction - without_run - run_failed,
        "model_improved": sum(model_improved),
        "human_improved": sum(human_improved),
        "human_total_improved": sum(human_total),
        "apply": (with_prediction - without_run)/total,
        "correctness": (with_prediction - without_run - run_failed)/total,
        "performance": sum(model_improved)/total,
        "human_performance": sum(human_improved)/total,
        "human_total_performance": sum(human_total)/total,
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="SWE-Perf/SWE-Perf", required=True, type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--log_root", required=True, type=str, help="log path")
    parser.add_argument("--output_path", required=True, type=str, help="performence output path")
    args = parser.parse_args()

    output_path = args.output_path

    log_root = args.log_root
    sweperf_data = load_sweperf_dataset(args.dataset_dir)
    sweperf_data = pd.DataFrame(sweperf_data)

    results = []
    # performence_paths = []
    # for log_root in log_roots:
    print("================================")
    print(log_root)
    print("[total]")
    result = calculate_performance_result(sweperf_data, log_root)
    result["repo"] = "total"
    predictions=result["with_prediction"]
    results.append(result)
    i=0
    for repo, group in sweperf_data.groupby('repo'):
        print("================================")
        print(log_root)
        print(f"[{repo}]")
        result = calculate_performance_result(group, log_root)
        if result:
            result["repo"] = repo
            results.append(result)
        i += len(group)
        if(i > predictions):
            break
    # save results to csv
    
    df = pd.DataFrame(results)
    log_instance = log_root.split('/')[-1]

    df.to_csv(output_path, index=False)

