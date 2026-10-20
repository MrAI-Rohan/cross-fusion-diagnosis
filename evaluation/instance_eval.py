import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class InstanceEvaluator:
    def __init__(self, gt_instance_meta, pred_instance_meta, gsd):
        gt_labels, gt_bbox, gt_area, gt_centroid = gt_instance_meta
        pred_labels, pred_bbox, pred_area, pred_centroid = pred_instance_meta

        gt_area = gt_area * (gsd**2)
        self.pred_area = pred_area * (gsd**2)
        self.pred_labels = pred_labels
        self.gt_labels = gt_labels

        # --- Filter GT by minimum area (remove noise < 2 m²) ---
        keep_gt = gt_area >= 2
        self.filtered_gt_areas = gt_area[keep_gt]
        self.filtered_gt_bbox = gt_bbox[keep_gt]
        self.num_gt = len(self.filtered_gt_areas)
        self.num_pred = len(pred_bbox)  # Keep ALL predictions for matching

        # Original 1-based IDs for the filtered GT (used for mask cropping)
        # Aligned 1:1 with filtered_gt_areas and filtered_gt_bbox
        self.gt_original_ids = torch.arange(1, len(gt_bbox) + 1, dtype=torch.int64)[keep_gt]

        # Compute IoU pairs: returns 0-based filtered GT indices and 0-based prediction indices
        self.gt_indices, self.pred_indices, self.ious = (
            self.compute_instance_pairs(
                gt_labels, pred_labels,
                self.filtered_gt_bbox, pred_bbox,
                self.gt_original_ids
            )
        )

    def compute_instance_pairs(self, gt_labels, pred_labels, gt_bbox, pred_bbox, gt_original_ids):
        """
        Returns:
            - gt_indices: 0-based indices into the filtered GT list
            - pred_indices: 0-based indices into the original prediction list
            - ious: mask IoU for each pair
        """
        if len(gt_bbox) == 0 or len(pred_bbox) == 0:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.float32),
            )

        # Bounding box overlap (prunes candidates before expensive mask IoU)
        x0 = torch.maximum(gt_bbox[:, None, 0], pred_bbox[None, :, 0])
        y0 = torch.maximum(gt_bbox[:, None, 1], pred_bbox[None, :, 1])
        x1 = torch.minimum(gt_bbox[:, None, 2], pred_bbox[None, :, 2])
        y1 = torch.minimum(gt_bbox[:, None, 3], pred_bbox[None, :, 3])

        overlap = (x1 > x0) & (y1 > y0)
        gt_idx, pred_idx = overlap.nonzero(as_tuple=True)

        if len(gt_idx) == 0:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.float32),
            )

        pair_gt = []
        pair_pred = []
        pair_iou = []

        for g, p in zip(gt_idx.tolist(), pred_idx.tolist()):
            # g = 0-based filtered GT index
            # p = 0-based prediction index

            # Get the original 1-based ID for mask cropping
            orig_gt_id = int(gt_original_ids[g])
            pred_id = p + 1  # 1-based for mask cropping

            gt_bbox_ = gt_bbox[g]
            pred_bbox_ = pred_bbox[p]

            # Crop the union region for speed
            xmin = min(gt_bbox_[0], pred_bbox_[0]).item()
            ymin = min(gt_bbox_[1], pred_bbox_[1]).item()
            xmax = max(gt_bbox_[2], pred_bbox_[2]).item()
            ymax = max(gt_bbox_[3], pred_bbox_[3]).item()

            gt_crop = gt_labels[ymin:ymax, xmin:xmax]
            pred_crop = pred_labels[ymin:ymax, xmin:xmax]

            gt_mask = gt_crop == orig_gt_id
            pred_mask = pred_crop == pred_id

            intersection = torch.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue

            union = torch.logical_or(gt_mask, pred_mask).sum()
            iou = intersection.float() / union.float()

            # Store the FILTERED INDEX (g) and PREDICTION INDEX (p)
            pair_gt.append(g)
            pair_pred.append(p)
            pair_iou.append(iou)

        return (
            torch.tensor(pair_gt, dtype=torch.int64),
            torch.tensor(pair_pred, dtype=torch.int64),
            torch.stack(pair_iou) if pair_iou else torch.empty(0, dtype=torch.float32),
        )

    def evaluate_instance_metrics(self, iou_thresh=0.5, return_indices=False):
        """
        Returns:
            - size_metrics (dict): Per-bin TP, FN, GT counts
            - global_metrics (dict): Overall TP, FP, FN, Precision, Recall, F1
            - matched_gt_indices (Tensor): 0-based filtered GT indices of matches
            - matched_pred_indices (Tensor): 0-based prediction indices of matches
        """

        # --- Define size bins (m²) ---
        bin_edges = [2, 75, 300, 1000, 5000, float('inf')]
        bin_names = ['small', 'medium', 'large', 'very_large', 'outlier']

        tp_per_bin = {name: 0 for name in bin_names}
        fn_per_bin = {name: 0 for name in bin_names}
        gt_per_bin = {name: 0 for name in bin_names}

        def get_bin_idx(area):
            idx = np.digitize(area, bin_edges) - 1
            return min(idx, len(bin_names) - 1)

        # --- Count total GT per bin ---
        for area in self.filtered_gt_areas:
            gt_per_bin[bin_names[get_bin_idx(area)]] += 1


        # --- Edge Case ---
        if len(self.ious) == 0:
            for area in self.filtered_gt_areas:
                fn_per_bin[bin_names[get_bin_idx(area)]] += 1

            overall_fp = (self.pred_area >= 2).sum().item()
            discarded_preds = (self.pred_area < 2).sum().item()
            return self._format_metrics(tp_per_bin, fn_per_bin, gt_per_bin,
                                       0, overall_fp, self.num_gt, self.num_gt, discarded_preds,
                                       torch.empty(0), torch.empty(0), return_indices=return_indices)

        # --- Filter pairs by IoU threshold ---
        valid = self.ious >= iou_thresh

        if valid.sum() == 0:
            for area in self.filtered_gt_areas:
                fn_per_bin[bin_names[get_bin_idx(area)]] += 1

            overall_fp = (self.pred_area >= 2).sum().item()
            discarded_preds = (self.pred_area < 2).sum().item()
            return self._format_metrics(tp_per_bin, fn_per_bin, gt_per_bin,
                                       0, overall_fp, self.num_gt, self.num_gt, discarded_preds,
                                       torch.empty(0), torch.empty(0), return_indices=return_indices)

        # --- Valid pairs exist ---
        gt_indices = self.gt_indices[valid]     # 0-based filtered indices
        pred_indices = self.pred_indices[valid] # 0-based prediction indices
        ious = self.ious[valid]

        # --- Hungarian Matching ---
        # Cost matrix dimensions: [num_filtered_GT] x [num_predictions]
        cost = np.zeros((self.num_gt, self.num_pred), dtype=np.float32)

        # DIRECT indexing using 0-based indices (NO -1 conversion!)
        cost[gt_indices.numpy(), pred_indices.numpy()] = ious.numpy()

        rows, cols = linear_sum_assignment(cost, maximize=True)
        valid_match = cost[rows, cols] > 0.0

        matched_gt_idx = rows[valid_match]    # 0-based filtered GT indices
        matched_pred_idx = cols[valid_match]  # 0-based prediction indices

        # --- Count TP per bin ---
        matched_gt_areas = self.filtered_gt_areas[matched_gt_idx]
        for area in matched_gt_areas:
            tp_per_bin[bin_names[get_bin_idx(area)]] += 1

        # --- Count FN per bin (unmatched GTs) ---
        unmatched_gt_mask = torch.ones(self.num_gt, dtype=torch.bool)
        unmatched_gt_mask[matched_gt_idx] = False
        unmatched_gt_areas = self.filtered_gt_areas[unmatched_gt_mask]
        for area in unmatched_gt_areas:
            fn_per_bin[bin_names[get_bin_idx(area)]] += 1

        # --- Global Metrics ---
        overall_tp = len(matched_gt_idx)
        overall_fn = self.num_gt - overall_tp

        # FP: Unmatched predictions that are >= 2 m²
        matched_mask = torch.zeros(self.num_pred, dtype=torch.bool)
        matched_mask[matched_pred_idx] = True
        unmatched_mask = ~matched_mask
        discarded_preds = (unmatched_mask & (self.pred_area < 2)).sum().item()
        overall_fp = (unmatched_mask & (self.pred_area >= 2)).sum().item()

        return self._format_metrics(tp_per_bin, fn_per_bin, gt_per_bin,
                                   overall_tp, overall_fp, overall_fn, self.num_gt, discarded_preds,
                                   torch.from_numpy(matched_gt_idx),
                                   torch.from_numpy(matched_pred_idx),
                                   return_indices=return_indices)

    def _format_metrics(self, tp_per_bin, fn_per_bin, gt_per_bin,
                        overall_tp, overall_fp, overall_fn, overall_gt, discarded_preds,
                        matched_gt_indices, matched_pred_indices, return_indices=False):
        """
        Helper to format metrics.
        """
        # Size-based metrics
        size_metrics = {}
        for name in tp_per_bin.keys():
            tp = tp_per_bin[name]
            fn = fn_per_bin[name]
            gt = gt_per_bin[name]

            size_metrics[name] = {
                'tp': tp,
                'fn': fn,
                'gt': gt,
            }

        # Global metrics
        global_metrics = {
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn,
            'gt': overall_gt,
            'discarded_preds': discarded_preds,
        }
        
        if return_indices:
            return size_metrics, global_metrics, matched_gt_indices, matched_pred_indices
        else:
            return size_metrics, global_metrics

    def compute_segmentation_errors(self, iou_thresh=0.1):
        """
        Compute under- and over-segmentation errors.

        Under-segmentation:
            A prediction overlaps more than one GT.

        Over-segmentation:
            A GT overlaps more than one prediction.
        """

        valid = self.ious >= iou_thresh

        if valid.sum() == 0:
            return {
                "under_seg_count": 0,
                "over_seg_count": 0,
                "gt_involved": 0,
                "pred_involved": 0,
                "under_severity": 0,
                "over_severity": 0,
            }

        gt_indices = self.gt_indices[valid]
        pred_indices = self.pred_indices[valid]

        gt_counts = torch.bincount(gt_indices, minlength=self.num_gt,)
        pred_counts = torch.bincount(pred_indices, minlength=self.num_pred,)

        over_severity = torch.clamp(gt_counts - 1, min=0).sum().item()
        under_severity = torch.clamp(pred_counts - 1, min=0).sum().item()

        return {
            "under_seg_count": (pred_counts > 1).sum().item(),
            "over_seg_count": (gt_counts > 1).sum().item(),
            "pred_involved": (pred_counts > 0).sum().item(),
            "gt_involved": (gt_counts > 0).sum().item(),
            "under_severity": under_severity,
            "over_severity": over_severity,
        }
    
# Comment to push properly
# One more

class InstanceMetrics:
    def __init__(self,):
        bin_names = ['small', 'medium', 'large', 'very_large', 'outlier']
        self.size_metrics = {k: {i: 0 for i in ["tp", "fn", "gt"]} for k in bin_names}
        self.global_metrics = {k: 0 for k in ["tp", "fp", "fn", "gt", "discarded_preds"]}

        self.segmentation_errors = {
                "under_seg_count": 0,
                "over_seg_count": 0,
                "gt_involved": 0,
                "pred_involved": 0,
                "under_severity": 0,
                "over_severity": 0,
            }

    def update(self, size_metrics, global_metrics, segmentation_error=None):
        for size_bin in self.size_metrics:
            for k in self.size_metrics[size_bin]:
                self.size_metrics[size_bin][k] += size_metrics[size_bin][k]

        for k in self.global_metrics:
            self.global_metrics[k] += global_metrics[k]

        if segmentation_error is not None:
            for k in self.segmentation_errors:
                self.segmentation_errors[k] += segmentation_error[k]
    
    def get_metrics(self):
        """
        Returns final metrics with derived values (Precision, Recall, F1, per-bin Recall).
        """
        # --- Global metrics ---
        g = self.global_metrics
        tp, fp, fn, gt = g['tp'], g['fp'], g['fn'], g['gt']
        discarded_preds = g['discarded_preds']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        den = precision + recall
        f1 = 2 * precision * recall / den if den > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        global_final = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt': gt,
            'discarded_preds': discarded_preds,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }

        # --- Per-size metrics ---
        size_final = {}
        for bin_name, counts in self.size_metrics.items():
            tp_bin = counts['tp']
            gt_bin = counts['gt']
            recall_bin = tp_bin / gt_bin if gt_bin > 0 else 0.0
            size_final[bin_name] = {
                'tp': tp_bin,
                'fn': counts['fn'],
                'gt': gt_bin,
                'recall': recall_bin,
            }

        return {
            'global': global_final,
            'size': size_final,
        }

    def get_segmentation_error(self):
        """
        Returns final segmentation error metrics with derived ratios and severity.
        """
        seg = self.segmentation_errors
        under_count = seg['under_seg_count']
        over_count = seg['over_seg_count']
        pred_involved = seg['pred_involved']
        gt_involved = seg['gt_involved']
        under_severity = seg['under_severity']
        over_severity = seg['over_severity']

        under_ratio = under_count / pred_involved if pred_involved > 0 else 0.0
        over_ratio = over_count / gt_involved if gt_involved > 0 else 0.0

        avg_under_severity = under_severity / under_count if under_count > 0 else 0.0
        avg_over_severity = over_severity / over_count if over_count > 0 else 0.0

        return {
            'under_seg_count': under_count,
            'over_seg_count': over_count,
            'gt_involved': gt_involved,
            'pred_involved': pred_involved,
            'under_seg_ratio': under_ratio,
            'over_seg_ratio': over_ratio,
            'under_severity': under_severity,
            'over_severity': over_severity,
            'average_under_severity': avg_under_severity,
            'average_over_severity': avg_over_severity
        }
