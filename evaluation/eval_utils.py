import gc
import cv2
import h5py
import time
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import TiledDataset
from data.transforms import build_transforms
from training_module import SegmentationModule
from evaluation.instance_eval import InstanceEvaluator, InstanceMetrics

def load_data(h5_path, patch_size,  batch_size, indices=None, stride=None, transform=None, num_workers=2):
    dataset = TiledDataset(h5_path, patch_size=patch_size, transform=transform,
                            use_cache=True, stride=stride, indices=indices)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)
    return dataloader

def load_model(ckpt_path, device="cuda"):
    model = SegmentationModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model

def build_eval_transform():
    data_cfg = {"normalization": "imagenet",}
    transform = build_transforms(data_cfg, mode="val")
    return transform

def update_pr_auc_counts(pred_mask, gt_mask, thresholds, tp, fp, fn,):
    # Flatten to 1D (P = H * W)
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten().bool()

    """
    Vectorized addition for WHU and Massachusetts. 

    # thresholds[:, None] -> shape (T, 1)
    # pred_flat[None, :]  -> shape (1, P)
    # Result: (T, P) boolean matrix where row t is (prob >= threshold[t])
    pred_binary = thresholds[:, None] <= pred_flat[None, :]
    
    # gt_flat[None, :] -> shape (1, P) broadcasts to (T, P)
    gt_binary = gt_flat[None, :]
    
    # Sum along axis=1 (across pixels) gives shape (T,)
    tp += torch.sum(pred_binary & gt_binary, axis=1)
    fp += torch.sum(pred_binary & ~gt_binary, axis=1)
    fn += torch.sum(~pred_binary & gt_binary, axis=1)
    """

    # iterative solution for INRIA
    # I should add an if condition later

    for i, t in enumerate(thresholds):
        pred_binary = pred_flat >= t          # shape (P,), not (101, P)
        tp[i] += torch.sum(pred_binary & gt_flat)
        fp[i] += torch.sum(pred_binary & ~gt_flat)
        fn[i] += torch.sum(~pred_binary & gt_flat)

def mask_to_boundary(mask, dilation_ratio = 0.02):
    # Convert a binary torch mask to its boundary mask.

    h, w = mask.shape
    diag = (h * h + w * w) ** 0.5
    d = max(1, int(round(dilation_ratio * diag)))

    mask_np = mask.numpy().astype(np.uint8) * 255

    # Using Chebyshev distance transform to find the boundary
    dist = cv2.distanceTransform(
        mask_np,
        cv2.DIST_C,
        cv2.DIST_MASK_PRECISE,
    )

    eroded = torch.from_numpy(dist > d)

    return (mask > 0) & (~eroded) # Boundary

def update_boundary_iou_counts(pred_mask, gt_mask, intersection, union,):
    pred_boundary = mask_to_boundary(pred_mask)
    gt_boundary = mask_to_boundary(gt_mask)
    
    intersection += torch.logical_and(pred_boundary, gt_boundary).sum()
    union += torch.logical_or(pred_boundary, gt_boundary).sum()

def get_gt_instance_meta(idx, h5_file,):
    mask = h5_file["mask_idx"][:]==idx
    labels = torch.from_numpy(h5_file["labels"][idx])
    area = torch.from_numpy(h5_file["area"][mask])
    bbox = torch.from_numpy(h5_file["bbox"][mask])
    centroid = torch.from_numpy(h5_file["centroid"][mask])
    return labels, bbox, area, centroid

def extract_instances(mask: torch.Tensor):
    mask_np = (mask > 0).cpu().numpy().astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np,
        connectivity=8,
    )

    labels = torch.from_numpy(labels)

    bbox = torch.from_numpy(stats[1:, :4].copy())

    # xywh to xyxy
    bbox[:, 2] += bbox[:, 0]
    bbox[:, 3] += bbox[:, 1]

    area = torch.from_numpy(stats[1:, 4].copy())
    centroid = torch.from_numpy(centroids[1:].copy())
    # num_labels = torch.tensor(num_labels - 1)

    return labels, bbox, area, centroid

def make_predictions_and_count(loader, model, h5_path, patch_size, instance_h5_path=None, 
                               gsd=None, threshold=0.5, compute_pr_auc=False, compute_cka=False,
                               instance_list=None, save_dir=None):
    # I should group counts in dictionaries to improve readability if pipeline grows.

    if not compute_pr_auc and (gsd is None or (instance_h5_path is None and instance_list is None)):
        raise Exception("Not passed: gsd or instance_h5_path or instance_list.")

    # Pixel based counts
    tp = fp = fn = tn = 0

    # Boundary IoU counts
    intersection = torch.tensor(0, dtype=torch.int64)
    union = torch.tensor(0, dtype=torch.int64)

    # Instance based counts
    instance_metrics05 = InstanceMetrics() # IoU threshold=0.5
    instance_metrics03 = InstanceMetrics() # IoU threshold=0.3


    if compute_pr_auc:
        thresholds = torch.linspace(0, 1, 101)
        tp_auc = torch.zeros(101, dtype=torch.int64)
        fp_auc = torch.zeros(101, dtype=torch.int64)
        fn_auc = torch.zeros(101, dtype=torch.int64)

    if compute_cka:
        cka_keys = []
        cka_vecs = [[] for _ in range(4)]  # e1..e4
        _batch_stage_pooled = {}

        def _encoder_hook(module, input, output):
            # output: list of 4 NHWC tensors from Swin encoder
            for stage_i, feat in enumerate(output):
                _batch_stage_pooled[stage_i] = feat.mean(dim=(1, 2)).detach().cpu().numpy()

        hook_handle = model.model.encoder.register_forward_hook(_encoder_hook)

    current_img = None
    full_pred = None
    count_map = None

    loader_time = infer_time = stitch_time = 0
    batch_end = time.time()

    with h5py.File(h5_path, 'r') as f:
        masks = f["masks"]
        pad_h, pad_w = loader.dataset.pad_h, loader.dataset.pad_w

        # Load instances in RAM
        if not compute_pr_auc:
            if instance_list is None:
                with h5py.File(instance_h5_path, 'r') as instance_f:
                    assert len(masks) == instance_f["labels"].shape[0], (
                            f"{len(masks)} doesn't match {instance_f["labels"].shape[0]}"
                    )
                    gt_instances = [get_gt_instance_meta(i, instance_f) for i in range(len(masks))]
            elif instance_list is not None:
                assert len(masks) == len(instance_list), (
                        f"{len(masks)} doesn't match {len(instance_list)}"
                )
                gt_instances = instance_list # To support fast city-wise instance metrics.
            
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Predicting patches", unit="batch", total=len(loader)):
                loader_time += time.time() - batch_end

                images, _, img_idx, y, x = batch
                images = images.cuda(non_blocking=True)

                torch.cuda.synchronize()

                t1 = time.time()

                preds = torch.sigmoid(model(images))

                torch.cuda.synchronize()
                infer_time += time.time() - t1

                preds = preds.cpu()

                t2 = time.time()

                if compute_cka:
                    for i in range(images.shape[0]):
                        cka_keys.append((img_idx[i].item(), y[i].item(), x[i].item()))
                        for stage_i in range(4):
                            cka_vecs[stage_i].append(_batch_stage_pooled[stage_i][i])

                for i in range(preds.shape[0]):
                    img = img_idx[i].item()
                    yi = y[i].item()
                    xi = x[i].item()

                    # NEW IMAGE → finalize previous one
                    if current_img is not None and img != current_img:
                        avg = full_pred / torch.clamp(count_map, min=1)
                        avg = avg[:orig_h, :orig_w]

                        gt = torch.from_numpy(masks[current_img]).float()

                        if save_dir is not None:
                            save_dir = Path(save_dir)
                            save_dir.mkdir(parents=True, exist_ok=True)
                            np.save(save_dir / f"img_{current_img:04d}_probs.npy", avg.numpy())

                        if compute_pr_auc:
                            update_pr_auc_counts(avg, gt, thresholds, tp_auc, fp_auc, fn_auc)
                        else:
                            pred_mask = (avg >= threshold)

                            tp += ((pred_mask == 1) & (gt == 1)).sum().item()
                            fp += ((pred_mask == 1) & (gt == 0)).sum().item()
                            fn += ((pred_mask == 0) & (gt == 1)).sum().item()
                            tn += ((pred_mask == 0) & (gt == 0)).sum().item()
                            
                            # Update boundary IoU counts
                            update_boundary_iou_counts(pred_mask, gt, intersection, union)
                            
                            # Instance counts
                            gt_instance_meta = gt_instances[current_img]
                            pred_instance_meta = extract_instances(pred_mask)

                            ie = InstanceEvaluator(gt_instance_meta, pred_instance_meta, gsd)
                            instance_metrics05.update(
                                *ie.evaluate_instance_metrics(0.5),
                                ie.compute_segmentation_errors(0.1)
                            )
                            
                            instance_metrics03.update(
                                *ie.evaluate_instance_metrics(0.3)
                            )

                        # cleanup
                        del full_pred, count_map

                    # initialize new image
                    if img != current_img:
                        current_img = img

                        orig_h, orig_w = masks[img].shape
                        padded_h = orig_h + pad_h
                        padded_w = orig_w + pad_w

                        full_pred = torch.zeros(padded_h, padded_w, dtype=torch.float16)
                        count_map = torch.zeros(padded_h, padded_w, dtype=torch.float16)

                    patch = preds[i].squeeze()

                    full_pred[yi:yi+patch_size, xi:xi+patch_size] += patch
                    count_map[yi:yi+patch_size, xi:xi+patch_size] += 1

                del images, preds

                stitch_time += time.time() - t2
                batch_end = time.time()


        # finalize last image
        if current_img is not None:
            avg = full_pred / torch.clamp(count_map, min=1)
            avg = avg[:orig_h, :orig_w]

            gt = torch.from_numpy(masks[current_img]).float()

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                np.save(save_dir / f"img_{current_img:04d}_probs.npy", avg.numpy())

            if compute_pr_auc:
                update_pr_auc_counts(avg, gt, thresholds, tp_auc, fp_auc, fn_auc)
            else:
                pred_mask = (avg >= threshold)
                tp += ((pred_mask == 1) & (gt == 1)).sum().item()
                fp += ((pred_mask == 1) & (gt == 0)).sum().item()
                fn += ((pred_mask == 0) & (gt == 1)).sum().item()
                tn += ((pred_mask == 0) & (gt == 0)).sum().item()

                # Update boundary IoU counts
                update_boundary_iou_counts(pred_mask, gt, intersection, union)

                # Instance counts
                gt_instance_meta = gt_instances[current_img]
                pred_instance_meta = extract_instances(pred_mask)

                ie = InstanceEvaluator(gt_instance_meta, pred_instance_meta, gsd)
                instance_metrics05.update(
                    *ie.evaluate_instance_metrics(0.5),
                    ie.compute_segmentation_errors(0.1)
                )

                instance_metrics03.update(
                    *ie.evaluate_instance_metrics(0.3)
                )

    del masks
    gc.collect()
    print(f"Loading time: {loader_time:.2f}s, Inference time: {infer_time:.2f}s, Stitching time: {stitch_time:.2f}s")

    if compute_cka:
        hook_handle.remove()

    if compute_pr_auc:
        result = {"tp": tp_auc, "fp": fp_auc, "fn": fn_auc, "thresholds": thresholds}
        if compute_cka:
            result["cka_keys"] = np.array(cka_keys)
            result["cka_activations"] = [np.stack(v) for v in cka_vecs]  # 4 arrays, (n_patches, channels)
        return result
        

    results =  {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "boundary_counts": {"intersection": intersection.item(), "union": union.item(),
                         "boundary_iou": (intersection/(union+1e-6)).item()},
        "instance_based": {
            "iou_0.5": instance_metrics05.get_metrics(),
            "iou_0.3": instance_metrics03.get_metrics(),
            "seg_error": instance_metrics05.get_segmentation_error()
        }
    }

    if compute_cka:
        results["cka_keys"] = np.array(cka_keys)
        results["cka_activations"] = [np.stack(v) for v in cka_vecs]

    return results
