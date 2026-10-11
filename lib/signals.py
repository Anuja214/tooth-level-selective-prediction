"""Subset of selective_prediction_lib.py needed for live demo + val τ."""
import os
import numpy as np

SEEDS = [42, 43, 44, 45, 46]
EPS = 1e-7


def load_seed_ensemble(predictions_dir, split_suffix):
    probs_by_seed, labels, q = [], None, None
    for s in SEEDS:
        d = np.load(os.path.join(predictions_dir, f"swin_t_seed{s}_{split_suffix}_predictions.npz"))
        probs_by_seed.append(d["probs"])
        if labels is None:
            labels = d["labels"]
            q = d["yolo_box_conf"] if "yolo_box_conf" in d else None
        else:
            assert np.array_equal(labels, d["labels"])
    return np.stack(probs_by_seed, axis=0), labels, q


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def entropy_decomposition(probs_stack):
    mean_p = probs_stack.mean(axis=0)
    h_total = binary_entropy(mean_p)
    h_aleatoric = binary_entropy(probs_stack).mean(axis=0)
    u_epistemic = np.clip(h_total - h_aleatoric, 0.0, None)
    return h_total, h_aleatoric, u_epistemic


def apply_temperature(probs, T):
    p = np.clip(probs, EPS, 1 - EPS)
    logits = np.log(p / (1 - p))
    z = logits / T
    return 1 / (1 + np.exp(-z))


def risk_coverage_curve(risk_score, correct, n_points=None):
    order = np.argsort(-risk_score, kind="stable")
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    selective_risk = 1.0 - (cum_correct / k)
    return coverage, selective_risk


def coverage_at_risk_target(coverage, selective_risk, target_risk):
    order = np.argsort(coverage)
    cov_sorted = coverage[order]
    risk_sorted = selective_risk[order]
    ok = risk_sorted <= target_risk
    if not ok.any():
        return 0.0
    return float(cov_sorted[ok].max())
