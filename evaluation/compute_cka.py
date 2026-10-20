import numpy as np

def linear_cka(X, Y):
    """Standard linear CKA (Kornblith et al. 2019), feature-space formulation —
    efficient when #channels << #patches, which is the case here."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    return cross / (norm_x * norm_y)

def subsample_nonoverlapping(keys, patch_size=224):
    """Keep only patches whose (y, x) land exactly on a non-overlapping patch_size grid"""
    mask = []
    for k in keys:
        *_, y, x = k  # works whether key is (img,y,x) or (city,img,y,x)
        mask.append((y % patch_size == 0) and (x % patch_size == 0))
    return mask

def compare_checkpoints(base_npz_path, cfe_npz_path, patch_size=224):
    base = np.load(base_npz_path, allow_pickle=True)
    cfe  = np.load(cfe_npz_path,  allow_pickle=True)

    base_keys = {tuple(k): i for i, k in enumerate(base["keys"])}
    cfe_keys  = {tuple(k): i for i, k in enumerate(cfe["keys"])}
    shared_all = sorted(set(base_keys) & set(cfe_keys))

    nonoverlap_mask = subsample_nonoverlapping(shared_all, patch_size)
    shared = [k for k, keep in zip(shared_all, nonoverlap_mask) if keep]

    print(f"{len(shared)} non-overlapping patches used (of {len(shared_all)} total overlapping)")
    # sanity check: should land near total/4, given 50% overlap in both dims

    base_idx = [base_keys[k] for k in shared]
    cfe_idx  = [cfe_keys[k] for k in shared]

    results = {}
    for stage in ["e1", "e2", "e3", "e4"]:
        X, Y = base[stage][base_idx], cfe[stage][cfe_idx]
        results[stage] = float(linear_cka(X, Y))
        print(f"  {stage}: CKA = {results[stage]:.4f}")
    return results

if __name__ == "__main__":
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_npz", type=str, required=True)
    parser.add_argument("--cfe_npz", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--save_filename", type=str, required=True)

    args = parser.parse_args()

    results = compare_checkpoints(
        args.base_npz,
        args.cfe_npz
    )

    os.makedirs(args.dest_dir, exist_ok=True)

    save_path = os.path.join(args.dest_dir, args.save_filename)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {save_path}")
