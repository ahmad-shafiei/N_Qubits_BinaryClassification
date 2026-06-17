import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10)/(q + 1e-10)))
def fidelity(p, q):
    return (np.sum(np.sqrt(p*q)))**2
def l1_error(p, q):
    return np.sum(np.abs(p-q))

def compute_metrics(preds, Y):
    fidelities = []
    l1s = []
    kls = []

    for i in range(len(preds)):
        p = preds[i]
        y = Y[i]

        fidelities.append((np.sum(np.sqrt(p * y))) ** 2)
        l1s.append(np.sum(np.abs(p - y)))
        kls.append(np.sum(y * np.log((y + 1e-10) / (p + 1e-10))))

    return {
        "FidelityMean": np.mean(fidelities),
        "L1Mean": np.mean(l1s),
        "KLMean": np.mean(kls),
    }