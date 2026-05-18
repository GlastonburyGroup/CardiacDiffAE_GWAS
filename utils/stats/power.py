import numpy as np
from scipy import stats

def simulate_power_wilcoxon(n, effect_size, n_sim=1000, alpha=0.05):
    successes = 0
    for _ in range(n_sim):
        data1 = np.random.normal(0, 1, n)  # Control group
        data2 = data1 + np.random.normal(effect_size, 1, n)  # Treatment group with effect size

        stat, p_value = stats.wilcoxon(data1, data2)
        if p_value < alpha:
            successes += 1

    power = successes / n_sim
    return power