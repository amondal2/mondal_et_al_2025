"""
Configuration object for hyperparameter tuning using ray.
"""
from ray import tune

config = {
    "l1": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l2": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l3": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l4": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l5": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l6": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l7": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l8": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l9": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l10": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l11": tune.choice([2 ** (i + 1) for i in range(9)]),
    "l12": tune.choice([2 ** (i + 1) for i in range(9)]),
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "dropout_prob": tune.uniform(0, 1),
}
