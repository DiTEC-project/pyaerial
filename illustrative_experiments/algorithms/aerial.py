import time

from aerial import model, rule_extraction, rule_quality


def run_aerial(dataset, antecedents, ant_sim, cons_sim, layer_dims=[10], epochs=2):
    start_time = time.time()
    # train an autoencoder on the given table
    trained_autoencoder = model.train(dataset, epochs=epochs, layer_dims=layer_dims)

    # extract association rules from the autoencoder
    association_rules = rule_extraction.generate_rules(trained_autoencoder,
                                                       max_antecedents=antecedents,
                                                       ant_similarity=ant_sim,
                                                       cons_similarity=cons_sim)
    exec_time = time.time() - start_time
    # calculate rule quality statistics (support, confidence, zhangs metric) for each rule
    if association_rules and len(association_rules) > 0:
        stats, rules, = rule_quality.calculate_rule_stats(association_rules,
                                                          trained_autoencoder.input_vectors,
                                                          max_workers=8)
        stats["exec_time"] = exec_time
        return stats, rules

    return None, None
