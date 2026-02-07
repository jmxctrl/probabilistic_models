"""Adaptive Binomial distribution utilities.

This module experiments with adaptivity of binomial probability
where p varies on context. n remains independent 


Inputs: 
   Array of probabilities for each user 
   Target number of successes k 

Enumerate over possible sequences 
   for all n trials, comb of k successes and n-k failures 

Calculate probability of each sequence 
   binomial probability dependent on pi

Sum probabilities of all sequences with k successes 

Return total probability 

"""

import math 
import itertools
import numpy as np 


def adaptive_binomial_pmf(prob_list: Iterable[float], k: int) -> float:
    
    ## convert list into float 
    probs = np.asarray(prob_list, dtype=float)
    n = len(probs)

    # Error checks 
    if not (0 <= k <= n):
        raise ValueError("k must be between 0 and n(inclusive)")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("All probabilities must be in 0 and 1")

    # generate all combinations of k successes 
    success_indices_combos = itertools.combinations(range(n), k)

    total_prob = 0.0

    for success_indices in success_indices_combos: 
        all_indices = np.arange(n)
        failure_indices = np.setdiff1d(all_indices, success_indices)
        prob_success = np.prod(probs[list(success_indices)])
        prob_failure = np.prod(1 - probs[list(failure_indices)])
        seq_prob = prob_success * prob_failure
        total_prob += seq_prob

    return total_prob



if __name__ == "__main__":
    # Example: 5 users with different probabilities, want exactly 3 opens
    prob_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    k = 3
    result = adaptive_binomial_pmf(prob_list, k)
    print(f"Probability of exactly {k} successes: {result}")
    