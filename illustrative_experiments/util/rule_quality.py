import pandas as pd


def mlxtend_calculate_average_rule_quality(rules):
    stats = []
    for rule in rules:
        stats.append([
            rule["support"], rule["confidence"],
        ])

    stats = pd.DataFrame(stats).mean()
    stats = {
        "rules": len(rules),
        "support": stats[0],
        "confidence": stats[1],
    }
    return stats


def mlxtend_calculate_dataset_coverage(rules, dataset: pd.DataFrame) -> float:
    coverage_mask = pd.Series(False, index=dataset.index)

    for _, rule in rules.iterrows():
        indices = rule["antecedents"]
        if not indices:
            continue
        cols = [dataset.columns[i] for i in indices]
        rule_mask = dataset[cols].all(axis=1)
        coverage_mask |= rule_mask

    coverage_percentage = 100.0 * coverage_mask.sum() / len(dataset)
    return coverage_percentage
