from collections import defaultdict

EXPERIMENTS = defaultdict(list)

def register_results(experiments_dict,
                     circuit,
                     training_noise,
                     noise_mode,
                     metrics):
    key = (circuit, training_noise)
    experiments_dict[key].append({
        "Dataset": noise_mode,
        **metrics
    })