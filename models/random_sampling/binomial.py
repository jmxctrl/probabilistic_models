"""Binomial distribution utilities.

This module provides a small, dependency-light implementation of the
Binomial distribution useful for ML/quant pipelines: `pmf`, `logpmf`,
`cdf`, sampling and an `fit_mle` helper.

The functions accept scalar or array-like `k` inputs and use `numpy`
for vectorized operations. Only the Python standard library and
`numpy` are required.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np

Number = Union[int, float]

## takes a list of numbers (in floats) and return nd array 
def _gammaln(x: np.ndarray) -> np.ndarray:
    """Compute elementwise log-gamma using math.lgamma as a fallback.

    Uses `np.vectorize` over `math.lgamma` to avoid adding scipy as a
    dependency while still supporting array inputs.
    """

    # function vec applies log-gamma of single number 
    vec = np.vectorize(math.lgamma, otypes=[float]) # output will be floats 
    return vec(x)

## calculates binomial coefficient (n choose k) 
def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    return _gammaln(n + 1) - _gammaln(k + 1) - _gammaln(n - k + 1)

def logpmf(k: Union[int, Iterable[int], np.ndarray], n: int, p: float) -> Union[float, np.ndarray]:
    """Log of the probability Mass Function P(X = k) for Binomial(n, p).

    Parameters
    - k: integer or array-like of integers (number of successes)
    - n: number of trials (non-negative integer)
    - p: success probability in [0, 1]

    Returns scalar or ndarray matching the shape of `k`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    k_arr = np.asarray(k)
    
    # handle scalar input 
    scalar = k_arr.shape == () # check scalar is single number 
    k_int = k_arr.astype(int) # convert k_int into int 
    invalid = (k_arr < 0) | (k_arr > n) | (k_arr != k_int) # no negative success, no more success than trial, non-integer 

    # compute logpmf where valid
    with np.errstate(divide="ignore"):
        lcomb = _log_comb(n, k_int) # stores log of binomial coefficients 
        # careful when p is 0 or 1 to avoid log(0) * 0 producing nan
        if p == 0.0:
            lp = np.where(k_int == 0, 0.0, -np.inf) # log prob of success
            lq = np.where(k_int == n, 0.0, -np.inf) # log prob of failure 
        elif p == 1.0:
            lp = np.where(k_int == n, 0.0, -np.inf)
            lq = np.where(k_int == 0, 0.0, -np.inf)
        else:
            lp = k_int * math.log(p)
            lq = (n - k_int) * math.log1p(-p)

        logpmf_vals = lcomb + lp + lq

    
    #for each value, 
        #if invalid (negative, too large, not integer),
            #set log-prob to -inf
        #otherwise keep value in logpmf_vals 
     
    logpmf_vals = np.where(invalid, -np.inf, logpmf_vals)
    return float(logpmf_vals) if scalar else logpmf_vals


def pmf(k: Union[int, Iterable[int], np.ndarray], n: int, p: float) -> Union[float, np.ndarray]:
    """Probability mass function P(X = k) for Binomial(n, p)."""
    lp = logpmf(k, n, p)
    if isinstance(lp, float):
        return math.exp(lp) if lp > -math.inf else 0.0 # converts log probability back to regular by exponentiating log
    return np.exp(lp)


def cdf(k: int, n: int, p: float) -> float:
    """Cumulative distribution function P(X <= k) for integer k."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    ks = np.arange(0, int(k) + 1)
    return float(np.sum(pmf(ks, n, p)))


def mean(n: int, p: float) -> float:
    return n * p


def var(n: int, p: float) -> float:
    return n * p * (1.0 - p)


def sample(size: Optional[Union[int, tuple]] = None, n: int = 1, p: float = 0.5, random_state: Optional[int] = None) -> np.ndarray:
    """Draw samples from Binomial(n, p) using numpy's Generator.

    - `size` follows numpy convention (None, int, or tuple).
    - returns ndarray of shape `size` (or scalar if size is None).
    """
    rng = np.random.default_rng(random_state)
    return rng.binomial(n, p, size=size)


def fit_mle(successes: Iterable[int], trials: Optional[Iterable[int]] = None) -> float:
    """Fit p by MLE from observed successes.

    - If `trials` is provided, both must be array-like of same shape and
      `p_hat = sum(successes) / sum(trials)`.
    - If `trials` is None, `successes` is assumed to be Bernoulli outcomes
      (0 or 1) and `p_hat` is the mean.
    """

    ## 0. convert input successes into array of floats 
    success = np.asarray(list(successes), dtype=float)

    # 1. if user provided no trial arg 
    if trials is None:
        if success.size == 0: # checks if success arrays empty, if so, output error 
            raise ValueError("no data provided")
        return float(np.mean(success)) # if no trials, return avg of success array 
    
    ## 2. converts input trial into array of floats 
    tr = np.asarray(list(trials), dtype=float)
    if success.shape != tr.shape: # checks if success / trial have same length
        raise ValueError("`successes` and `trials` must have the same shape") # raise error if so
   
    ## sum all successes
    total_success = float(np.sum(success))
    total_trial = float(np.sum(tr))

    ## if trial less than 0, raise error 
    if total_trial <= 0:
        raise ValueError("total trials must be positive")

    ## return estimated prob: success (Good Value) / total trial (n)
    return total_success / total_trial


__all__ = ["pmf", "logpmf", "cdf", "sample", "fit_mle", "mean", "var"]
