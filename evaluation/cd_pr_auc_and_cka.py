import torch
import numpy as np
from pathlib import Path

from evaluation.eval_utils import load_data, load_model, build_eval_transform, make_predictions_and_count
from evaluation.pr_auc import store_pr_auc
from evaluation.benchmark import get_inria_city_indices

def compute_cross_domain_pr_auc_and_cka(ckpt_path, h5_root, patch_size=224, batch_size=256,
                                         stride=112, dest_dir=None, compute_cka=False):
    h5_root = Path(h5_root)
    project_dir = (Path(dest_dir) if dest_dir is not None
                   else Path(__file__).parent.parent / "results" / "cross_domain")


    model = load_model(ckpt_path)
    transform = build_eval_transform()
    city_indices = get_inria_city_indices(h5_root / "inria_val.h5")
    cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    thresholds = None
    tp_total = fp_total = fn_total = None

    all_cka_keys = []       # (city, img_idx, y, x) per patch
    all_cka_vecs = [[] for _ in range(4)]

    config_name = Path(ckpt_path).stem
    if "_last" in config_name or "_best" in config_name:
            config_name = config_name.rsplit("_", 1)[0]

    pr_auc_dir = project_dir / "pr_auc" / config_name
    pr_auc_dir.mkdir(parents=True, exist_ok=True)
    cka_dir = project_dir / "cka"
    if compute_cka:
        cka_dir.mkdir(parents=True, exist_ok=True)

    for city in cities:
        loader = load_data(h5_root / "inria_val.h5", patch_size=patch_size, batch_size=batch_size,
                            transform=transform, stride=stride, indices=city_indices[city])
        counts = make_predictions_and_count(
            loader=loader, model=model, h5_path=h5_root / "inria_val.h5",
            patch_size=patch_size, compute_pr_auc=True, compute_cka=compute_cka,
        )

        # PR-AUC: additive, as before
        if thresholds is None:
            thresholds = counts["thresholds"]
            tp_total, fp_total, fn_total = counts["tp"].clone(), counts["fp"].clone(), counts["fn"].clone()
        else:
            tp_total += counts["tp"]; fp_total += counts["fp"]; fn_total += counts["fn"]
        
        store_pr_auc(f"{config_name}_{city}_crossdomain",
                      counts["thresholds"], counts["tp"], counts["fp"], counts["fn"], pr_auc_dir)

        if compute_cka:
            # CKA: just concatenate — this IS the city aggregation
            for img_id, yy, xx in counts["cka_keys"]:
                all_cka_keys.append((city, img_id, yy, xx))
            for stage_i in range(4):
                all_cka_vecs[stage_i].append(counts["cka_activations"][stage_i])

        del loader
        torch.cuda.empty_cache()

    store_pr_auc(f"{config_name}_overall_crossdomain", thresholds, tp_total, fp_total, fn_total, pr_auc_dir)

    if compute_cka:
        cka_save = {"keys": np.array(all_cka_keys, dtype=object)}
        for stage_i in range(4):
            cka_save[f"e{stage_i+1}"] = np.concatenate(all_cka_vecs[stage_i], axis=0)
        np.savez(cka_dir / f"{config_name}_cka_activations.npz", **cka_save)
        print(f"Saved CKA activations: {len(all_cka_keys)} patches across {len(cities)} cities")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--h5_root", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cka", action="store_true", help="also save the CKA activations (about 2 GB per checkpoint)")
    args = parser.parse_args()
    compute_cross_domain_pr_auc_and_cka(args.ckpt_path, args.h5_root,
                                        batch_size=args.batch_size, dest_dir=args.dest_dir, compute_cka=args.cka)