# Runs DE and PSO algorithms for ARM using NiaARM: https://github.com/firefly-cpp/NiaARM

import numpy as np
from niaarm import Dataset, get_rules
from niapy.algorithms.basic import DifferentialEvolution, ParticleSwarmOptimization


def run_niaarm_de(transactions, population_size=200, max_evals=50000):
    algo = DifferentialEvolution(population_size=population_size, differential_weight=0.5, crossover_probability=0.9)
    metrics = ('support', 'confidence')

    dataset = Dataset(transactions)
    rules, exec_time = get_rules(dataset, algorithm=algo, metrics=metrics, max_evals=max_evals, logging=False)
    if len(rules) == 0:
        return {
            "rule_count": 0,
            "average_support": 0,
            "average_confidence": 0,
            "data_coverage": 0,
            "exec_time": exec_time
        }

    coverage = calculate_coverage(rules, transactions)
    return {
        "rule_count": len(rules),
        "average_support": rules.mean("support"),
        "average_confidence": rules.mean("confidence"),
        "data_coverage": coverage,
        "exec_time": exec_time
    }


def run_niaarm_pso(transactions, population_size=200, max_evals=50000):
    algo = ParticleSwarmOptimization(population_size=population_size)
    metrics = ('support', 'confidence')

    dataset = Dataset(transactions)
    rules, exec_time = get_rules(dataset, algorithm=algo, metrics=metrics, max_evals=max_evals, logging=False)
    if len(rules) == 0:
        return {
            "rule_count": 0,
            "average_support": 0,
            "average_confidence": 0,
            "data_coverage": 0,
            "exec_time": exec_time
        }

    coverage = calculate_coverage(rules, transactions)
    return {
        "rule_count": len(rules),
        "average_support": rules.mean("support"),
        "average_confidence": rules.mean("confidence"),
        "data_coverage": coverage,
        "exec_time": exec_time
    }


def calculate_coverage(rules, dataset):
    rule_coverage = np.zeros(len(dataset))

    for index, row in dataset.iterrows():
        for rule in rules:
            covered = True
            for item in rule.antecedent:
                if item.categories:
                    if item.categories[0] not in list(row):
                        covered = False
                        break
                else:
                    covered = False
                    for key, value in row.items():
                        if item.name == key:
                            if item.min_val <= value <= item.max_val:
                                covered = True
                                break
            if covered:
                rule_coverage[index] = 1
                break

    return sum(rule_coverage) / len(dataset)
