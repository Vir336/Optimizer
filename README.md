# 🚀 StatSGD: Statistical Stochastic Gradient Descent Optimizer

`statsgd` is a custom optimization algorithm based on **Stochastic Gradient Descent (SGD)** with Momentum, featuring a unique **statistical boosting mechanism**. It identifies and accelerates the updates for parameters whose gradient components are statistically significantly larger than the mean gradient magnitude, aiming to speed up convergence by prioritizing updates for salient dimensions.

---

## ✨ Key Features

* **Momentum:** Uses the standard momentum technique to accelerate convergence in relevant directions and dampen oscillations.
* **Statistical Boosting:** Calculates the **mean** ($\mu$) and **standard deviation** ($\sigma$) of the absolute gradient magnitudes ($\left|g\right|$) and uses these statistics to compute **z-scores**.
* **Outlier Acceleration:** If a gradient component's z-score exceeds a threshold (default $\mathbf{1.5}$), its effective learning rate is multiplied by a $\mathbf{boost\_factor}$ (default $\mathbf{100}$). This selectively accelerates the learning along the most significant parameter dimensions.
* **Scalability:** For high-dimensional problems (where the number of parameters, `numel`, is greater than `stats_batch_size`), the statistics are estimated from a random sample of the gradients to maintain performance.

---

## ⚙️ Function Signature

The optimizer is implemented as a single Python function:

```python
def statsgd(f, grad_f, x0, lr=0.01, beta=0.9, boost_factor=100,
            stats_batch_size=2000, eps=1e-8, tol=1e-6, max_iter=25000):

```

## Example Usage

```python
# --- Define the function and its gradient ---
# f(x) = x[0]^2 + 10 * x[1]^2 + 5
def objective_function(x):
    return x[0]**2 + 10 * x[1]**2 + 5

# grad_f(x) = [2 * x[0], 20 * x[1]]
def gradient_function(x):
    return np.array([2 * x[0], 20 * x[1]])

# --- Run the Optimizer ---
x_start = np.array([5.0, 5.0]) # Starting point

x_opt, num_iter, history = statsgd(
    f=objective_function, 
    grad_f=gradient_function, 
    x0=x_start, 
    lr=0.005, 
    boost_factor=10, 
    max_iter=500,
    tol=1e-8
)

# --- Results ---
print(f"Initial Loss: {objective_function(x_start):.4f}")
print(f"Final parameters (x): {x_opt}")
print(f"Iterations run: {num_iter}")
print(f"Final loss: {objective_function(x_opt):.4f}")
