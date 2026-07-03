import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aerial.model import train
from aerial.rule_extraction import generate_rules

# ── config ─────────────────────────────────────────────────────────────────────

NUM_ROWS = 10000
COLUMN_COUNTS = list(range(5, 55, 5))
NUM_CLASSES = 5
EPOCHS = 10
MIN_RULE_FREQUENCY = 0.5
MIN_RULE_STRENGTH = 0.8
SEEDS = [42, 123, 7]


def _synthetic_data(num_columns, seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, NUM_CLASSES, NUM_ROWS)
    columns = {}
    for i in range(num_columns):
        noise = rng.integers(0, NUM_CLASSES, NUM_ROWS)
        keep_base = rng.random(NUM_ROWS) < 0.7
        columns[f"col{i}"] = np.where(keep_base, base, noise).astype(str)
    return pd.DataFrame(columns)


def _run_once(num_columns, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    df = _synthetic_data(num_columns, seed)

    t0 = time.perf_counter()
    model = train(df, epochs=EPOCHS, show_progress=False)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    generate_rules(model, min_rule_frequency=MIN_RULE_FREQUENCY, min_rule_strength=MIN_RULE_STRENGTH,
                   max_antecedents=None)
    extract_time = time.perf_counter() - t0

    return train_time, extract_time


if __name__ == "__main__":
    results = {}
    for num_columns in COLUMN_COUNTS:
        train_times, extract_times = [], []
        for seed in SEEDS:
            train_time, extract_time = _run_once(num_columns, seed)
            train_times.append(train_time)
            extract_times.append(extract_time)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] columns={num_columns} | seed={seed} | "
                  f"train={train_time:.1f}s extract={extract_time:.3f}s", flush=True)
        results[num_columns] = {
            "train_time": round(sum(train_times) / len(train_times), 3),
            "extract_time": round(sum(extract_times) / len(extract_times), 3),
        }
        print(f"  → columns={num_columns}  avg train={results[num_columns]['train_time']}s  "
              f"avg extract={results[num_columns]['extract_time']}s", flush=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")