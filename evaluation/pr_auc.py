import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from evaluation.eval_utils import make_predictions_and_count, load_model, load_data, build_eval_transform

def store_pr_auc(config_name, thresholds, tp, fp, fn, dest_dir):
    # --- Move to CPU and convert to NumPy ---
    thresholds_np = thresholds.cpu().numpy()
    tp_np = tp.cpu().numpy()
    fp_np = fp.cpu().numpy()
    fn_np = fn.cpu().numpy()

    # --- Compute Precision, Recall, F1 ---
    eps = 1e-10
    precision = tp_np / (tp_np + fp_np + eps)
    recall = tp_np / (tp_np + fn_np + eps)
    f1_scores = 2 * (precision * recall) / (precision + recall + eps)

    # --- Compute PR-AUC (sorted by recall) ---
    order = np.argsort(recall)
    recall_sorted, idx = np.unique(recall[order], return_index=True)
    precision_sorted = precision[order][idx]

    pr_auc = np.trapezoid(precision_sorted, recall_sorted)

    # --- Optimal threshold (max F1) ---
    opt_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_np[opt_idx]
    optimal_precision = precision[opt_idx]
    optimal_recall = recall[opt_idx]
    optimal_f1 = f1_scores[opt_idx]

    # --- Prepare dictionary for saving ---
    save_dict = {
        'thresholds': thresholds_np,
        'tp': tp_np,
        'fp': fp_np,
        'fn': fn_np,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1_scores,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_f1': optimal_f1,
    }

    # --- Save to .npz ---
    np.savez(dest_dir/f"pr_auc_{config_name}.npz", **save_dict)
    print(f"✅ PR-AUC data saved to pr_auc_{config_name}.npz")
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.3f} (F1={optimal_f1:.4f})")

    return optimal_threshold


def main(params=None):
    if params is None:
        parser = argparse.ArgumentParser(description="Compute and store PR-AUC data.")
        parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 dataset.")
        parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
        parser.add_argument("--patch_size", type=int, required=True, help="Patch size for testing.")
        parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing.")
        parser.add_argument("--stride", type=int, default=None, help="Stride for testing.")
        parser.add_argument("--dest_dir", type=str, required=True, help="Directory to save results npz file.")

        args = parser.parse_args()

    else:
        # Convert dict into an object with attribute access
        args = SimpleNamespace(**params)

    model = load_model(args.ckpt_path)
    transform = build_eval_transform()
    loader = load_data(
            args.h5_path,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            transform=transform,
            stride=args.stride,
        )
    
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    pr_auc_metrics = make_predictions_and_count(loader=loader,
                                                model=model,
                                                h5_path=args.h5_path, 
                                                patch_size=args.patch_size,
                                                compute_pr_auc=True)
    
    optimal_threshold = store_pr_auc(config_name=Path(args.ckpt_path).stem,
                                     dest_dir=dest_dir,
                                     **pr_auc_metrics,)

    return optimal_threshold


if __name__ == "__main__":
    main()
