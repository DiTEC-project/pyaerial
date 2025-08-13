# Runs FP-Growth algorithm from SPMF library: https://www.philippe-fournier-viger.com/spmf/

import os
import re
import subprocess
import time

import pandas as pd


def run_spmf_fpgrowth(dataset, min_support=0.5, min_confidence=0.8, antecedents=2, return_stats=False):
    one_hot_df = pd.get_dummies(dataset, prefix_sep='__')

    unique_items = pd.Series(one_hot_df.columns)
    item_to_id = {item: idx + 1 for idx, item in enumerate(unique_items)}

    # SPMF accepts input in the form of numbers where each number represents an item
    transactions_ids = one_hot_df.apply(
        lambda row: ' '.join(str(item_to_id[item]) for item in row.index[row == 1]),
        axis=1
    )

    input_file = './illustrative_experiments/transactions.txt'
    output_file = './illustrative_experiments/spmf_rules.txt'

    transactions_ids.to_csv(input_file, index=False, header=False)

    start = time.time()
    subprocess.run([
        "java", "-jar", "./illustrative_experiments/algorithms/spmf_source/spmf.jar",
        "run", "FPGrowth_association_rules",
        input_file, output_file,
        str(min_support), str(min_confidence),
        # 1 refers to maximum number of consequents
        str(antecedents), "1"
    ], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    exec_time = time.time() - start

    if not return_stats:
        exec_results = {
            "exec_time": exec_time,
            "rule_count": count_rules(output_file)
        }
    else:
        stats = parse_spmf_rules(output_file, transactions_ids)
        exec_results = {
            "rule_count": stats["rule_count"],
            "average_support": stats["average_support"],
            "average_confidence": stats["average_confidence"],
            "data_coverage": stats["data_coverage"],
            "exec_time": exec_time
        }

    os.remove(input_file)
    os.remove(output_file)

    return exec_results


def parse_spmf_rules(file_path, transactions):
    """
    file_path: path to SPMF output file
    transactions: list of sets, each set = transaction items (IDs or names)
    """
    total_transactions = len(transactions)
    supports = []
    confidences = []
    covered_transaction_ids = set()
    rule_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            # Extract antecedent and support/confidence values
            match = re.search(r'^(.*?)==>.*?#SUP:\s*([\d\.]+)\s+#CONF:\s*([\d\.]+)', line)
            if match:
                rule_count += 1
                antecedent_str = match.group(1).strip()
                sup_count = float(match.group(2))
                conf = float(match.group(3))

                # Convert antecedent string to set of ints
                antecedent_items = set(map(int, antecedent_str.split()))

                # Store metrics
                supports.append((sup_count / total_transactions))
                confidences.append(conf)

                # Find all transactions covered by this antecedent
                for idx, t in enumerate(transactions):
                    if antecedent_items.issubset(set(map(int, t.split()))):
                        covered_transaction_ids.add(idx)

    average_support = sum(supports) / len(supports) if supports else 0
    average_conf = sum(confidences) / len(confidences) if confidences else 0
    coverage = (len(covered_transaction_ids) / total_transactions)

    return {
        "average_support": average_support,
        "average_confidence": average_conf,
        "data_coverage": coverage,
        "rule_count": rule_count
    }


def count_rules(file_path):
    with open(file_path) as f:
        return sum(1 for line in f if '==>' in line)
