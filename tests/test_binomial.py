import math
from math import comb, log
import numpy as np

from models.random_sampling import binomial


def test_fit_mle_simple():
    successes = [4]
    trials = [10]
    result = binomial.fit_mle(successes, trials)
    assert result == 0.4 

def test_logpmf():
    k = 3
    n = 10
    p = 0.3
    result = binomial.logpmf(k, n, p)

    expected = log(comb(n, k)) + 3 * log(0.3) + 7 * log(0.7)
    assert abs(result - expected) < 1e-10