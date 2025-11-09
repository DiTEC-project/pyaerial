import time

from aerial import model, rule_extraction


def run_aerial(dataset, antecedents, ant_sim, cons_sim, layer_dims=[10], epochs=2):
    start_time = time.time()
    # train an autoencoder on the given table
    trained_autoencoder = model.train(dataset, epochs=epochs, layer_dims=layer_dims)

    # extract association rules from the autoencoder with quality metrics calculated automatically
    result = rule_extraction.generate_rules(trained_autoencoder,
                                           max_antecedents=antecedents,
                                           ant_similarity=ant_sim,
                                           cons_similarity=cons_sim,
                                           num_workers=8)
    exec_time = time.time() - start_time

    # Return statistics and rules in the new format
    if result and len(result['rules']) > 0:
        stats = result['statistics']
        stats["exec_time"] = exec_time
        return stats, result['rules']

    return None, None
