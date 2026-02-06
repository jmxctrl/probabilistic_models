## Sampling with Replacement 

N balls. Some are good (G), some bad (B), and you put the ball back after every draw.

## Setup
- Population size: **N**
- Good elements: **G**
- Bad elements: **B**, where `G + B = N`
- Probability of drawing a good element on any draw:

$$
p = \frac{G}{N}, \quad q = 1 - p
$$


Each draw is 
1. independent
2. identically distributed 
3. probability never changes 

Binomial Distribution 
Bin(n, p)

$$
X \sim \text{Binomial}(n, p)
$$

## Normal Approximation (Large n)

### Mean
$$
\mu = np
$$

### Standard deviation
$$
\sigma = \sqrt{npq}
$$

### Exactly \( k \) good results

The probability of getting **exactly \( k \) good results** in \( n \) draws is:

$$
P(X = k) = \binom{n}{k} p^k q^{\,n-k}
$$


### Exactly \( k \) good and \( b \) bad results

This is the same event, written explicitly:

$$
P(\text{exactly } k \text{ good and } n-k \text{ bad}) 
= \binom{n}{k} \left(\frac{G}{N}\right)^k \left(\frac{B}{N}\right)^{n-k}
$$


