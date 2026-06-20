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


def matched_noise_modes(training_noise):
    """Return test noise mode(s) aligned with training condition."""
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
    """Register metrics on test noise matched to training noise.

    Baseline (No Training) is evaluated on all three noise types.
    Trained models are evaluated only on the same noise used in training.
    """
    for noise_mode in matched_noise_modes(training_noise):
        X_test, Y_test = test_sets[noise_mode]
        metrics = evaluate_phase(model, X_test, Y_test)
        experiments[(circuit, training_noise)].append({
            "Dataset": noise_mode,
            **metrics,
        })


# Backward-compatible alias
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
    """Per-phase helper: evaluate only on noise matched to training."""
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
