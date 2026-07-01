import json
import logging
import multiprocessing
import random
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from aerial.model import train
from aerial.rule_extraction import generate_rules
from aerial.discretization import equal_frequency_discretization


def _git_version():
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip().lstrip("v")
        return tag
    except subprocess.CalledProcessError:
        return "unknown"


# ── config ─────────────────────────────────────────────────────────────────────

DATASETS_DIR = Path(__file__).parent / "datasets"
EPOCHS = 10
MIN_SUPPORT = 0.1
MIN_CONFIDENCE = 0.8
SEEDS = [42, 123, 7, 55, 121231, 5345, 613131, 123125, 234, 6745]
N_PARALLEL = 10


def _prepare(path):
    df = pd.read_csv(path).dropna()
    if df.select_dtypes(include="number").shape[1] > 0:
        df = equal_frequency_discretization(df)
    return df


def _run_once(df, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t0 = time.perf_counter()
    model = train(df, epochs=EPOCHS, show_progress=False)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = generate_rules(model, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE)
    extract_time = time.perf_counter() - t0

    stats = result["statistics"]
    return {
        "train_time": train_time,
        "extract_time": extract_time,
        "rule_count": stats.get("rule_count", 0),
        "avg_support": stats.get("average_support", 0.0),
        "avg_confidence": stats.get("average_confidence", 0.0),
        "avg_zhangs_metric": stats.get("average_zhangs_metric", 0.0),
        "data_coverage": stats.get("data_coverage", 0.0),
    }


def _parallel_task(args):
    torch.set_num_threads(1)
    logging.getLogger("aerial").setLevel(logging.CRITICAL)
    dataset_name, df, seed = args
    return dataset_name, seed, _run_once(df, seed)


def _log(dataset_name, seed, r):
    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{ts}] {dataset_name} | seed={seed} | rules={r['rule_count']} "
        f"support={r['avg_support']:.3f} confidence={r['avg_confidence']:.3f} "
        f"zhang={r['avg_zhangs_metric']:.3f} train={r['train_time']:.1f}s",
        flush=True,
    )


def _run(path):
    df = _prepare(path)
    dataset_name = path.stem
    completed = {}

    if N_PARALLEL > 1:
        tasks = [(dataset_name, df, seed) for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=N_PARALLEL,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_parallel_task, t): t[2] for t in tasks}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    _, _, result = future.result()
                    completed[seed] = result
                    _log(dataset_name, seed, result)
                except Exception as e:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] {dataset_name} | seed={seed} | ERROR: {e}\n{traceback.format_exc()}",
                          flush=True)
    else:
        for seed in SEEDS:
            try:
                result = _run_once(df, seed)
                completed[seed] = result
                _log(dataset_name, seed, result)
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] {dataset_name} | seed={seed} | ERROR: {e}\n{traceback.format_exc()}",
                      flush=True)

    if not completed:
        return {"error": "all seeds failed"}
    runs = list(completed.values())
    keys = runs[0].keys()
    avg = {k: round(sum(run[k] for run in runs) / len(runs), 3) for k in keys}
    avg["sd"] = {
        k: round(float(np.std([run[k] for run in runs], ddof=1)), 3) if len(runs) > 1 else 0.0
        for k in keys
    }
    return avg


PERF_KEYS = ["rule_count", "avg_support", "avg_confidence", "avg_zhangs_metric", "data_coverage"]
TIME_KEYS = ["train_time", "extract_time"]


def _overall_avg(results):
    valid = [v for v in results.values() if "error" not in v]
    if not valid:
        return {}
    avg = {k: round(sum(r[k] for r in valid) / len(valid), 3) for k in PERF_KEYS}
    avg["sd"] = {k: round(sum(r["sd"][k] for r in valid) / len(valid), 3) for k in PERF_KEYS}
    return avg


def _diff(current_avg, prev_path):
    try:
        with open(prev_path) as f:
            prev = json.load(f)
        prev_avg = prev.get("overall_avg", {})
        return {k: round(current_avg[k] - prev_avg[k], 3) for k in PERF_KEYS if k in prev_avg}
    except Exception:
        return {}


if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    stem = f"pyaerial_{_git_version()}"
    out = results_dir / f"{stem}.json"
    if out.exists():
        out = results_dir / f"{stem}_updated.json"

    existing = sorted(results_dir.glob("pyaerial_*.json"), key=lambda p: p.stat().st_mtime)
    prev_path = existing[-1] if existing else None

    results = {}
    timings = {}
    for csv in sorted(DATASETS_DIR.glob("*.csv")):
        name = csv.stem
        print(f"\n{name} ({len(SEEDS)} seeds):", flush=True)
        try:
            r = _run(csv)
            timings[name] = {k: r[k] for k in TIME_KEYS if k in r}
            results[name] = {k: v for k, v in r.items() if k not in TIME_KEYS}
            if "error" not in r:
                print(f"  → rules={r['rule_count']}  train={r['train_time']:.1f}s  extract={r['extract_time']:.3f}s")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ERROR: {e}\n{traceback.format_exc()}")

    if timings:
        valid_t = [t for t in timings.values() if TIME_KEYS[0] in t]
        avg_train = sum(t["train_time"] for t in valid_t) / len(valid_t)
        avg_extract = sum(t["extract_time"] for t in valid_t) / len(valid_t)
        print(f"\nAvg train={avg_train:.1f}s  avg extract={avg_extract:.3f}s  (not saved)")

    avg = _overall_avg(results)
    results["overall_avg"] = avg
    if avg:
        print("\nOverall averages across datasets (mean ± avg within-dataset SD):")
        for k in PERF_KEYS:
            print(f"  {k}: {avg[k]} ± {avg['sd'][k]}")
    if prev_path:
        results["diff"] = _diff(avg, prev_path)
        print(f"Diff vs {prev_path.name}: {results['diff']}")

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
