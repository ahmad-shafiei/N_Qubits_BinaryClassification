import torch
from .metrics import compute_metrics

def evaluate_phase(model, X_test, Y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        preds = model(X_test).numpy()

    return compute_metrics(preds, Y_test)


def run_all_tests(model, experiments,
                  circuit, training_noise, test_sets):

    for noise_mode, (X_test, Y_test) in test_sets.items():
        metrics = evaluate_phase(model, X_test, Y_test)

        experiments[(circuit, training_noise)].append({
            "Dataset": noise_mode,
            **metrics
        })