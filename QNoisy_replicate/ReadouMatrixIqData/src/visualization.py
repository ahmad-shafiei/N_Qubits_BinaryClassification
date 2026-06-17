import numpy as np
import torch
import matplotlib.pyplot as plt
from .metrics import fidelity
def visualize_aligned_samples(
    model, X_syn, Y_syn, X_single, Y_single,
    X_corr, Y_corr,n_samples=3,seed=42):

    np.random.seed(seed)
    model.eval()
    # 🎯 shared indices across ALL datasets
    base_len = len(X_syn)
    indices = np.random.choice(base_len, n_samples, replace=False)
    print(f"Shared indices: {indices}")
    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        datasets = [("Synthetic", X_syn, Y_syn, axes[0]),
            ("Experimental Single", X_single, Y_single, axes[1]),
            ("Experimental Correlated", X_corr, Y_corr, axes[2]),   ]
        for title, X, Y, ax in datasets:
            x = torch.tensor( X[idx], dtype=torch.float32 ).unsqueeze(0)
            with torch.no_grad():
                pred = model(x).numpy().flatten()
            true = Y[idx]
            noisy = X[idx]
            f = fidelity(true, pred)
            x_axis = np.arange(len(true))
            ax.bar(x_axis - 0.25, true, width=0.25, label="Ideal")
            ax.bar(x_axis, noisy, width=0.25, label="Noisy", alpha=0.6)
            ax.bar(x_axis + 0.25, pred, width=0.25, label="Predicted")
            ax.set_title(f"{title}\nFidelity = {f:.4f}")
            ax.set_xlabel("State Index")
            ax.set_ylabel("Probability")
            ax.grid()
            ax.legend()
        plt.suptitle(f"Aligned Sample Index = {idx}", fontsize=14)
        plt.tight_layout()
        plt.show()