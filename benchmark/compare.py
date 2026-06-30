import json
import sys

METRICS = ["rule_count", "avg_support", "avg_confidence", "avg_zhangs_metric", "data_coverage", "train_time", "extract_time"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _delta(before, after):
    if before == 0:
        return "N/A"
    pct = (after - before) / before * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _print_table(label, before_row, after_row):
    print(f"\n{label}")
    print(f"  {'metric':<22} {'before':>10} {'after':>10} {'delta':>10}")
    print(f"  {'-' * 54}")
    for m in METRICS:
        bv = before_row.get(m, "-")
        av = after_row.get(m, "-")
        delta = _delta(bv, av) if isinstance(bv, (int, float)) and isinstance(av, (int, float)) else "-"
        print(f"  {m:<22} {str(bv):>10} {str(av):>10} {delta:>10}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py before.json after.json")
        sys.exit(1)

    before = _load(sys.argv[1])
    after = _load(sys.argv[2])

    valid = [(d, before[d], after[d])
             for d in sorted(set(before) | set(after))
             if "error" not in before.get(d, {"error": True}) and "error" not in after.get(d, {"error": True})]

    for dataset, b, a in valid:
        _print_table(dataset, b, a)

    if len(valid) > 1:
        def _avg(rows, m):
            vals = [row[m] for row in rows if isinstance(row.get(m), (int, float))]
            return round(sum(vals) / len(vals), 3) if vals else "-"

        before_rows = [b for _, b, _ in valid]
        after_rows = [a for _, _, a in valid]
        avg_b = {m: _avg(before_rows, m) for m in METRICS}
        avg_a = {m: _avg(after_rows, m) for m in METRICS}
        _print_table("OVERALL (avg across datasets)", avg_b, avg_a)