# cross_domain_pr_auc.py
import torch
import numpy as np
from pathlib import Path

from evaluation.eval_utils import load_data, load_model, build_eval_transform, make_predictions_and_count
from evaluation.pr_auc import store_pr_auc
from evaluation.benchmark import get_inria_city_indices

def compute_cross_domain_pr_auc_and_cka(ckpt_path, h5_root, patch_size=224, batch_size=256,
                                         stride=112, dest_dir="pr_auc_cross_domain"):
    h5_root = Path(h5_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path)
    transform = build_eval_transform()
    city_indices = get_inria_city_indices(h5_root / "inria_val.h5")
    cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    thresholds = None
    tp_total = fp_total = fn_total = None

    all_cka_keys = []       # (city, img_idx, y, x) per patch
    all_cka_vecs = [[] for _ in range(4)]

    config_name = Path(ckpt_path).stem
    dest_dir_ = dest_dir / config_name
    dest_dir_.mkdir(parents=True, exist_ok=True)

    for city in cities:
        loader = load_data(h5_root / "inria_val.h5", patch_size=patch_size, batch_size=batch_size,
                            transform=transform, stride=stride, indices=city_indices[city])
        counts = make_predictions_and_count(
            loader=loader, model=model, h5_path=h5_root / "inria_val.h5",
            patch_size=patch_size, compute_pr_auc=True, compute_cka=True,
        )

        # PR-AUC: additive, as before
        if thresholds is None:
            thresholds = counts["thresholds"]
            tp_total, fp_total, fn_total = counts["tp"].clone(), counts["fp"].clone(), counts["fn"].clone()
        else:
            tp_total += counts["tp"]; fp_total += counts["fp"]; fn_total += counts["fn"]
        
        store_pr_auc(f"{config_name}_{city}_crossdomain",
                      counts["thresholds"], counts["tp"], counts["fp"], counts["fn"], dest_dir_)

        # CKA: just concatenate — this IS the city aggregation
        for img_id, yy, xx in counts["cka_keys"]:
            all_cka_keys.append((city, img_id, yy, xx))
        for stage_i in range(4):
            all_cka_vecs[stage_i].append(counts["cka_activations"][stage_i])

        del loader
        torch.cuda.empty_cache()

    store_pr_auc(f"{config_name}_overall_crossdomain", thresholds, tp_total, fp_total, fn_total, dest_dir_)

    cka_save = {"keys": np.array(all_cka_keys, dtype=object)}
    for stage_i in range(4):
        cka_save[f"e{stage_i+1}"] = np.concatenate(all_cka_vecs[stage_i], axis=0)
    np.savez(dest_dir_ / f"{config_name}_cka_activations.npz", **cka_save)
    print(f"Saved CKA activations: {len(all_cka_keys)} patches across {len(cities)} cities")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--h5_root", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, default="pr_auc_cross_domain")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    compute_cross_domain_pr_auc_and_cka(args.ckpt_path, args.h5_root,
                                        batch_size=args.batch_size, dest_dir=args.dest_dir)