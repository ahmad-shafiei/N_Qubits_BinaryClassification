import numpy as np
import pandas as pd
import torch

from .metrics import compute_metrics, fidelity, l1_error, kl_divergence

NOISE_MODES = ["synthetic", "experimental_single", "experimental_correlated"]

TRAINING_NOISE_TO_MODE = {
    "Synthetic": "synthetic",
    "Experimental Single": "experimental_single",
    "Experimental Correlated": "experimental_correlated",
}

MODE_TO_TRAINING_NOISE = {v: k for k, v in TRAINING_NOISE_TO_MODE.items()}


def make_test_sets(synthetic=None, experimental_single=None, experimental_correlated=None):
    """Build a test_sets dict from optional (X, Y) pairs."""
    out = {}
    if synthetic is not None:
        out["synthetic"] = synthetic
    if experimental_single is not None:
        out["experimental_single"] = experimental_single
    if experimental_correlated is not None:
        out["experimental_correlated"] = experimental_correlated
    return out


def matched_noise_modes(training_noise):
    """Noise mode(s) to evaluate for a given training condition."""
    if training_noise == "No Training":
        return NOISE_MODES
    return [TRAINING_NOISE_TO_MODE[training_noise]]


def evaluate_phase(model, X_test, Y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    return compute_metrics(preds, Y_test)


def run_matched_tests(model, experiments, circuit, training_noise, test_sets):
    """Register metrics on test_sets restricted to training_noise policy.

    - No Training  -> all keys present in test_sets (should be 3 noises)
    - Trained phase -> only the noise matching training_noise
    """
    allowed = set(matched_noise_modes(training_noise))
    for noise_mode, (X_test, Y_test) in test_sets.items():
        if noise_mode not in allowed:
            continue
        metrics = evaluate_phase(model, X_test, Y_test)
        experiments[(circuit, training_noise)].append({
            "Dataset": noise_mode,
            **metrics,
        })


run_all_tests = run_matched_tests


def evaluate_model(model, X, Y):
    model.eval()
    fids, l1s, kls = [], [], []
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    for pred, true in zip(preds, Y):
        fids.append(fidelity(true, pred))
        l1s.append(l1_error(true, pred))
        kls.append(kl_divergence(true, pred))
    return {
        "fidelity_mean": np.mean(fids),
        "l1_mean": np.mean(l1s),
        "kl_mean": np.mean(kls),
        "all_fidelities": np.array(fids),
        "all_l1": np.array(l1s),
        "all_kl": np.array(kls),
    }


def evaluate_matched_phase(model, training_noise, test_sets):
    """Evaluate model only on noise matched to training_noise."""
    results = {}
    for mode in matched_noise_modes(training_noise):
        X_test, Y_test = test_sets[mode]
        results[mode] = evaluate_model(model, X_test, Y_test)
    return results


def results_to_dataframe(results):
    rows = []
    for name, res in results.items():
        rows.append({
            "Dataset": name,
            "FidelityMean": round(res["fidelity_mean"], 3),
            "L1 Mean": round(res["l1_mean"], 3),
            "KL Mean": round(res["kl_mean"], 3),
        })
    return pd.DataFrame(rows)
