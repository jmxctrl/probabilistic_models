## Poisson Distribution

### Definition
Let \( X \sim \text{Poisson}(\lambda) \) with probability mass function (pmf):

$$
P(X = k) = e^{-\lambda} \frac{\lambda^k}{k!}, \quad k = 0,1,2,\dots
$$

### Properties
- **Mean:** \( \lambda \)
- **Variance:** \( \lambda \)
- **Additivity:**  
$$
X_1 \sim \text{Poisson}(\lambda_1), \quad
X_2 \sim \text{Poisson}(\lambda_2)
$$

$$
X_1 + X_2 \sim \text{Poisson}(\lambda_1 + \lambda_2)
$$


### Interpretation
Models the number of events occurring in a fixed interval of time or space,
assuming events occur independently and at a constant average rate.

### Connections
- Limit of the Binomial distribution as \( n \to \infty \), \( p \to 0 \), with \( np = \lambda \)
- Increments of a Poisson process
