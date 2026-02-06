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

\[
X \sim \text{Binomial}(n, p)
\]

Normal Approximation 
- **Mean**:
  \[
  \mu = np
  \]

- **Standard deviation**:
  \[
  \sigma = \sqrt{npq}
  \]
