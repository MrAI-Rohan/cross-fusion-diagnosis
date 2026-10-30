"""PR curve and PR-AUC from threshold-wise pixel counts. numpy only, so analysis scripts can import it without torch.
This file could later be merged into evaluation/pr_auc.py if not used indepedently.

Conventions (the same as sklearn.metrics.precision_recall_curve):
  * recall is tp / (tp + fn); at threshold 0 every pixel is predicted positive, so recall is 1.
  * where nothing is predicted positive (tp + fp == 0) precision is undefined; it is taken as 1.
  * the curve must start at recall 0. If no threshold reaches recall 0 (probabilities that saturate to exactly 1.0
    still pass `>= 1.0`), the point (recall 0, precision 1) is added.
"""
import numpy as np

EPS = 1e-10
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def precision_recall(tp, fp, fn):
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    predicted = tp + fp
    precision = np.where(predicted > 0, tp / np.maximum(predicted, 1.0), 1.0)
    recall = tp / (tp + fn + EPS)
    return precision, recall


def pr_auc_from_counts(thresholds, tp, fp, fn):
    """Area under the precision-recall curve, walking the thresholds from high to low so recall never decreases."""
    t = np.asarray(thresholds, dtype=np.float64)
    order = np.argsort(-t, kind="stable")
    precision, recall = precision_recall(np.asarray(tp)[order], np.asarray(fp)[order], np.asarray(fn)[order])
    if recall[0] > 0:
        recall = np.r_[0.0, recall]
        precision = np.r_[1.0, precision]
    return float(_trapezoid(precision, recall))