import re
import csv
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from evaluation.eval_utils import load_model

# Comment to push properly

STAGE_MAP = {"layers_0": "e1", "layers_1": "e2", "layers_2": "e3", "layers_3": "e4"}

def get_stage(key):
    """patch_embed is upstream of all four stages — reported separately.
    layers_N.downsample.* (patch-merging at the END of stage N) is grouped
    under stage N's own scope, not attributed to the next stage."""
    if "patch_embed" in key:
        return "patch_embed"
    m = re.search(r"layers_(\d)", key)
    if m:
        return STAGE_MAP.get(f"layers_{m.group(1)}", "unknown")
    return "other"

def compare_encoder_weights(base_ckpt_path, cfe_ckpt_path, key_filter="model.encoder."):
    # device="cpu" explicit — this task was scoped CPU-only from the start,
    # and load_model()'s default is "cuda"
    base_model = load_model(base_ckpt_path, device="cpu")
    cfe_model  = load_model(cfe_ckpt_path,  device="cpu")

    base_sd = base_model.state_dict()
    cfe_sd  = cfe_model.state_dict()

    base_enc = {k: v for k, v in base_sd.items() if key_filter in k}
    cfe_enc  = {k: v for k, v in cfe_sd.items()  if key_filter in k}

    shared = sorted(set(base_enc) & set(cfe_enc))
    missing_base = set(cfe_enc) - set(base_enc)
    missing_cfe  = set(base_enc) - set(cfe_enc)
    if missing_base or missing_cfe:
        print(f"WARNING: {len(missing_base)} keys only in CFE ckpt, "
              f"{len(missing_cfe)} only in base ckpt — skipped, not compared.")

    by_stage = defaultdict(list)
    per_tensor_rows = []

    for k in shared:
        b = base_enc[k].float().flatten()
        c = cfe_enc[k].float().flatten()
        if b.shape != c.shape:
            print(f"SKIP {k}: shape mismatch {b.shape} vs {c.shape}")
            continue

        l2 = torch.norm(c - b).item()
        rel_l2 = l2 / (torch.norm(b).item() + 1e-12)
        cos_sim = torch.nn.functional.cosine_similarity(b.unsqueeze(0), c.unsqueeze(0)).item()

        stage = get_stage(k)
        row = {"key": k, "stage": stage, "rel_l2": rel_l2, "cos_sim": cos_sim, "n_params": b.numel()}
        by_stage[stage].append(row)
        per_tensor_rows.append(row)

    return by_stage, per_tensor_rows


def save_results(by_stage, per_tensor_rows, dest_dir, save_filename):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(save_filename).stem  # strip extension if user included one

    # summary (per-stage aggregated) — the citeable result
    summary_path = dest_dir / f"{stem}_summary.csv"
    stage_order = ["patch_embed", "e1", "e2", "e3", "e4", "other", "unknown"]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n_tensors", "mean_rel_l2", "median_rel_l2", "mean_cos_sim", "min_cos_sim"])
        for stage in stage_order:
            if stage not in by_stage:
                continue
            vals = by_stage[stage]
            rel_l2s = [v["rel_l2"] for v in vals]
            cos_sims = [v["cos_sim"] for v in vals]
            w.writerow([stage, len(vals), f"{np.mean(rel_l2s):.4f}", f"{np.median(rel_l2s):.4f}",
                        f"{np.mean(cos_sims):.4f}", f"{np.min(cos_sims):.4f}"])
    print(f"Saved summary: {summary_path}")

    # full per-tensor detail — for verification / debugging later
    detail_path = dest_dir / f"{stem}_detail.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "stage", "rel_l2", "cos_sim", "n_params"])
        w.writeheader()
        w.writerows(per_tensor_rows)
    print(f"Saved detail: {detail_path}")

    return summary_path, detail_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", type=str, required=True)
    p.add_argument("--cfe_ckpt", type=str, required=True)
    p.add_argument("--key_filter", type=str, default="model.encoder.")
    p.add_argument("--dest_dir", type=str, required=True, help="Directory to save result CSVs.")
    p.add_argument("--save_filename", type=str, required=True,
                   help="Base filename (no extension needed) — saved as "
                        "{filename}_summary.csv and {filename}_detail.csv.")
    args = p.parse_args()

    by_stage, per_tensor_rows = compare_encoder_weights(args.base_ckpt, args.cfe_ckpt, args.key_filter)
    save_results(by_stage, per_tensor_rows, args.dest_dir, args.save_filename)