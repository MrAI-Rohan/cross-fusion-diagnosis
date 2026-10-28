import os
import gc
import json
import h5py
import torch
import pickle
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils import compute_metrics
import evaluation.pr_auc as pr_auc
from evaluation.instance_eval import InstanceMetrics
from evaluation.eval_utils import load_data, load_model, build_eval_transform, make_predictions_and_count

def run_benchmark(model, h5_root, dataset_dict, patch_size, batch_size, stride, dataset_flags, threshold):
    results = {}
    instance_results = {}

    for enabled, (name, config) in zip(dataset_flags, dataset_dict.items()):
        if enabled != "1":
            continue

        dataset_path = h5_root / config["file"]
        instance_path = h5_root / config["instance_file"]

        transform = build_eval_transform()

        loader = load_data(
            dataset_path,
            patch_size=patch_size,
            batch_size=batch_size,
            transform=transform,
            stride=stride,
            indices=config.get("indices", None)
        )

        counts = make_predictions_and_count(
            loader=loader,
            model=model,
            h5_path=dataset_path,
            instance_h5_path=instance_path,
            patch_size=patch_size,
            gsd=config["gsd"],
            threshold=threshold,
            compute_pr_auc=False
        )

        results[name] = compute_metrics(
            counts["confusion_matrix"]["tp"],
            counts["confusion_matrix"]["fp"],
            counts["confusion_matrix"]["fn"],
            counts["confusion_matrix"]["tn"]
        )
        results[name]["threshold"] = threshold

        instance_results[name] = counts["instance_based"]

        del loader
        torch.cuda.empty_cache()
        gc.collect()

    return results, instance_results


def get_inria_city_indices(h5_path):
    cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    with h5py.File(h5_path, "r") as f:
        filenames = f["filenames"][:]

    city_indices = {city: [] for city in cities}

    for idx, filename in enumerate(filenames):
        filename = filename.decode() if isinstance(filename, bytes) else filename

        for city in cities:
            if filename.startswith(city):
                city_indices[city].append(idx)
                break

    return city_indices


def save_results_to_csv(results, config_name, csv_path="benchmark_results.csv"):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    rows = [
        {
            "config": config_name,
            "dataset": dataset,
            "timestamp": timestamp,
            **metrics
        }
        for dataset, metrics in results.items()
    ]

    df = pd.DataFrame(rows)

    header = (
        not os.path.exists(csv_path)
        or os.path.getsize(csv_path) == 0
    )

    df.to_csv(
        csv_path,
        mode="a",
        header=header,
        index=False
    )

# Line to make commit proper

def update_pkl_dict(pkl_path, key, value):
    """
    Update a dictionary stored in a pickle file.
    """

    if not isinstance(value, dict):
        raise TypeError("value must be a dictionary")

    # Load existing data
    if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0:
        with open(pkl_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception:
                data = {}
    else:
        data = {}

    # Update (or create) key
    data.setdefault(key, {})
    data[key].update(value)

    # Save back
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def aggregate_inria_instance_metrics(city_metrics):
    """
    Aggregate instance metrics across INRIA cities.
    """

    metrics05 = InstanceMetrics()
    metrics03 = InstanceMetrics()

    for city in city_metrics.values():
        metrics05.update(city["iou_0.5"]["size"], city["iou_0.5"]["global"], city["seg_error"])
        metrics03.update(city["iou_0.3"]["size"], city["iou_0.3"]["global"])

    return {
        "iou_0.5": metrics05.get_metrics(),
        "iou_0.3": metrics03.get_metrics(),
        "seg_error": metrics05.get_segmentation_error(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for WHU building segmentation.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for testing.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing.")
    parser.add_argument("--dataset_flags", type=str, default="111", help="Which datasets to evaluate, show with a 3-digit boolean code,"
                        " first digit for WHU Test, second for Massachusetts, third for INRIA. E.g. '110' means evaluate only WHU Test and Massachusetts.")
    parser.add_argument("--stride", type=int, default=None, help="Stride for testing.")
    parser.add_argument("--dest_dir", type=str, default=None, help="Directory to save results CSV.")
    parser.add_argument("--dest_file1", type=str, default="benchmark_results.csv", help="File name to store WHU and Massachusetts benchmarks.")
    parser.add_argument("--dest_file2", type=str, default="inria_benchmark_results.csv", help="File name to store INRIA benchmarks.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for binary classification.")
    parser.add_argument("--pkl_file", type=str, default="instance_metrics.pkl", help="File name to store instance based metrics.")
    # parser.add_argument("--pr_auc_dir", type=str, default=None, help="Save pr auc data and then use optimal threshold to benchmark.")

    # Ensure that all split HDF5 files are present in the same directory,
    # because script will only look for them there or throw an error if not found.

    project_dir = Path(__file__).parent.parent
    
    args = parser.parse_args()

    if args.dest_dir is not None:
        dest_dir = Path(args.dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Default to saving in the results directory relative to this script.
        dest_dir = project_dir / "results"

    h5_path = Path(args.h5_path)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")
    
    config_name = ckpt_path.stem

    # Handle config naming in cross domain pr-auc and any other cases
    # where naming has unnecessary suffix when storing main results.
    if "_last" in config_name or "_best" in config_name:
        config_name = config_name.rsplit("_", 1)[0]

    model = load_model(ckpt_path)

    threshold = args.threshold

    if threshold is None :
        with open(project_dir / "results" / "optimal_thresholds.json", "r") as f:
            data = json.load(f)
        threshold = data["optimal_thresholds"].get(config_name, None)

        # Run PR-AUC if no threshold found.
        if threshold is None:
            # Only WHU training in this paper.
            if "whu" not in config_name:
                raise ValueError(f"Cannot choose a validation file for '{config_name}': the name does not contain 'whu'. "
                                 f"Pass --threshold, or see {project_dir}/config_checks.py for the naming template.")
            
            val_h5_path = h5_path / "whu_val_main.h5"
 

            threshold = pr_auc.main(
                {
                "h5_path": str(val_h5_path),
                "ckpt_path": str(ckpt_path),
                "patch_size": args.patch_size,
                "batch_size": args.batch_size,
                "stride": args.stride,
            })


    results = inria_results = None

    if args.dataset_flags[:2] != "00":
        dataset_dict = {
            "WHU Test": {
                "file": "whu_test_main.h5",
                "instance_file": "whu_test_instances_main.h5",
                "gsd": 0.3
            },
            "Massachusetts": {
                "file": "mas_test.h5",
                "instance_file": "mas_test_instances.h5",
                "gsd": 1
            }
        }

        results, instance_results = run_benchmark(model, h5_path, dataset_dict, args.patch_size,
                                 args.batch_size, args.stride, args.dataset_flags[:2],
                                 threshold=threshold)
    
        save_results_to_csv(results, config_name=config_name, csv_path=dest_dir / args.dest_file1)
        update_pkl_dict(dest_dir / args.pkl_file, config_name, instance_results)
    
    if args.dataset_flags[2] == "1":
        city_indices = get_inria_city_indices(h5_path / "inria_val.h5")

        cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
        inria_datasets = {}

        for city in cities:
            inria_datasets[city] = {
                "file": "inria_val.h5",
                "instance_file": "inria_val_instances.h5",
                "indices": city_indices[city],
                "gsd": 0.3
            }

        inria_results, inria_instance_results = (
            run_benchmark(model, h5_path, inria_datasets, args.patch_size,
                args.batch_size, args.stride, "1"*len(inria_datasets), threshold=threshold)
        )
        
        cf = {i: sum(inria_results[j][i] for j in inria_results) for i in ["tp", "fp", "fn", "tn"]}
        inria_results["overall"] = compute_metrics(**cf)
        inria_results["overall"]["threshold"] = threshold

        inria_instance_results["overall"] = aggregate_inria_instance_metrics(inria_instance_results)

        save_results_to_csv(inria_results, config_name=config_name, csv_path=dest_dir / args.dest_file2)
        update_pkl_dict(dest_dir / args.pkl_file, config_name, inria_instance_results)


if __name__ == "__main__":
    main()

