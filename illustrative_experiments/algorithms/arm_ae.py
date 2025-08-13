# Run ARM-AE using its original source code which is copied under "arm_ae_source" folder
#
# Berteloot, Théophile, Richard Khoury, and Audrey Durand. "Association rules mining with auto-encoders." International
# Conference on Intelligent Data Engineering and Automated Learning. Cham: Springer Nature Switzerland, 2024.

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from illustrative_experiments.algorithms.arm_ae_source.armae import ARMAE


def run_arm_ae(dataset, likeness=0.5, rules_per_consequent=2, antecedents=2, epochs=2):
    transactions = dataset.apply(
        lambda row: [str(f"{col}_{row[col]}") for col in dataset.columns],
        axis=1
    )
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    one_hot_encoded_data = pd.DataFrame(te_ary, columns=te.columns_)
    arm_ae = ARMAE(len(one_hot_encoded_data.loc[0]), maxEpoch=epochs, batchSize=2, learningRate=5e-3, likeness=likeness)
    dataLoader = arm_ae.dataPreprocessing(one_hot_encoded_data)

    arm_ae.train(dataLoader)
    arm_ae.generateRules(one_hot_encoded_data,
                         numberOfRules=rules_per_consequent,
                         nbAntecedent=antecedents)
    arm_ae_stats = arm_ae.reformat_rules(transactions, list(one_hot_encoded_data.columns))
    return arm_ae_stats
