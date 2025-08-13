import warnings

from illustrative_experiments.algorithms.spmf import *
from illustrative_experiments.algorithms.arm_ae import *
from illustrative_experiments.algorithms.aerial import *
from illustrative_experiments.algorithms.niaarm import *
from illustrative_experiments.algorithms.mlxtend import *
from illustrative_experiments.algorithms.arules import *
from illustrative_experiments.algorithms.pyeclat import *
from illustrative_experiments.util.preprocessing import *

# todo: resolve the warnings
warnings.filterwarnings("ignore")

# Generic hyper-parameters
ANTECEDENTS = 2

# Aerial hyper-parameters
NUMBER_OF_RUNS = 1
ANT_SIMILARITY = 0.5
CONS_SIMILARITY = 0.8
LAYER_DIMS = [5]
EPOCHS = 5

# Exhaustive hyper-parameters
MIN_CONFIDENCE = 0.8
# min support is to be decided based on Aerial+'s average support for fairness


# Optimization-based ARM hyperparameters
POPULATION_SIZE = 200
MAX_EVALS = 500

# load the dataset
file_name = "./illustrative_experiments/data/cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
gene_expression = pd.read_csv(file_name)


def print_stats(stats, algorithm):
    if stats is not None and len(stats) > 0:
        stats_keys = ["rule_count", "average_support", "average_confidence", "data_coverage", "exec_time"]
        print(algorithm, " --> ", " | ".join(f"{key}: {stats[key]:.2f}" for key in stats_keys))


# gene_count = column count (not one-hot encoded yet)
gene_count_list = [50]
aerial_support_90pct_per_column = {}

print("INFO: Running Aerial+ with PyAerial for", NUMBER_OF_RUNS, "times per column for robustness ...")

aerial_total_stats = []
for gene_count in gene_count_list:
    print("Column count:", gene_count)
    gene_expression_subset = gene_expression.iloc[:, :gene_count]
    # for each Aerial+ execution per different column count, find a support value that covers 90% of the rules
    ninety_pct_support_values = []
    # Run Aerial+ NUMBER_OF_RUNS times and get average results for robustness since Aerial+ is not exhaustive
    for i in range(NUMBER_OF_RUNS):
        print((i + 1), "/", NUMBER_OF_RUNS)
        aerial_stats, aerial_rules = run_aerial(gene_expression_subset, antecedents=ANTECEDENTS, ant_sim=ANT_SIMILARITY,
                                                cons_sim=CONS_SIMILARITY, epochs=EPOCHS, layer_dims=LAYER_DIMS)
        print_stats(aerial_stats, "Aerial+")
        if aerial_stats:
            aerial_total_stats.append(aerial_stats)
            ninety_pct_support_values.append(sorted((r["support"] for r in aerial_rules), reverse=True)[
                                                 max(int(len(aerial_rules) * 0.9) - 1, 0)])
    aerial_avg_stats = pd.DataFrame(aerial_total_stats).mean()
    print_stats(aerial_avg_stats, "Aerial+ Average Stats")
    aerial_support_90pct_per_column[gene_count] = {"support": np.average(ninety_pct_support_values),
                                                   "rules": aerial_avg_stats["rule_count"]}

print("\nINFO: Running other libraries/algorithms ...\n")
for gene_count, metrics in aerial_support_90pct_per_column.items():
    print("Column count:", gene_count)
    support, rules = metrics["support"], metrics["rules"]

    # Load the data
    gene_expression_subset = gene_expression.iloc[:, :gene_count]
    one_hot_encoded_data = one_hot_encode_dataframe(gene_expression_subset)

    # Run MLxtend - HMine
    print("INFO: Running MLxtend HMine ...")
    mlxtend_hmine_stats = run_mlxtend_hmine(one_hot_encoded_data, antecedents=ANTECEDENTS,
                                            min_support=support,
                                            min_confidence=MIN_CONFIDENCE)
    print_stats(mlxtend_hmine_stats, "MLxtend - HMine")

    # Run MLxtend - FP-Growth
    print("INFO: Running MLxtend - FP-Growth ...")
    mlxtend_fpg_stats = run_mlxtend_fpgrowth(one_hot_encoded_data, antecedents=ANTECEDENTS,
                                             min_support=support,
                                             min_confidence=MIN_CONFIDENCE)
    print_stats(mlxtend_fpg_stats, "MLxtend - FP-Growth")

    # Run pyECLAT - ECLAT
    # print("INFO: Running pyECLAT - ECLAT ...")
    # pyeclat_stats = run_pyeclat(gene_expression_subset, antecedents=ANTECEDENTS,
    #                             min_support=support,
    #                             min_confidence=MIN_CONFIDENCE)
    # print_stats(pyeclat_stats, "pyECLAT - ECLAT")

    # Run arules - ECLAT
    print("INFO: Running arules - ECLAT ...")
    arules_eclat_stats = run_arules_eclat(gene_expression_subset, antecedents=ANTECEDENTS,
                                          min_support=support, min_confidence=MIN_CONFIDENCE)
    print_stats(arules_eclat_stats, "arules - ECLAT")

    # Run arules - Apriori
    print("INFO: Running arules - Apriori ...")
    arules_apriori_stats = run_arules_apriori(gene_expression_subset, antecedents=ANTECEDENTS,
                                              min_support=support, min_confidence=MIN_CONFIDENCE)
    print_stats(arules_apriori_stats, "arules - Apriori")

    # Run NiaARM - DE
    print("INFO: Running NiaARM - DE ...")
    niaarm_de_stats = run_niaarm_de(gene_expression_subset, population_size=POPULATION_SIZE, max_evals=MAX_EVALS)
    print_stats(niaarm_de_stats, "NiaARM - DE")

    # Run NiaARM - PSO
    print("INFO: Running NiaARM - PSO ...")
    niaarm_pso_stats = run_niaarm_pso(gene_expression_subset, population_size=POPULATION_SIZE, max_evals=MAX_EVALS)
    print_stats(niaarm_pso_stats, "NiaARM - PSO")

    # Run SPMF - FP-Growth
    print("INFO: Running SPMF - FP-Growth ...")
    spmf_fpgrowth_stats = run_spmf_fpgrowth(gene_expression_subset, antecedents=ANTECEDENTS, return_stats=True,
                                            min_support=support, min_confidence=MIN_CONFIDENCE)
    print_stats(spmf_fpgrowth_stats, "SPMF - FP-Growth")

    # Run ARM-AE
    print("INFO: Running ARM-AE ...")
    # numberOfRules per consequent is adjusted to approximate Aerial+ for a fair comparison
    arm_ae_stats = run_arm_ae(gene_expression_subset, likeness=0.5,
                              rules_per_consequent=max(int(rules / one_hot_encoded_data.shape[1]), 2),
                              antecedents=ANTECEDENTS)
    print_stats(arm_ae_stats, "ARM-AE")
