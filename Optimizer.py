import numpy as np

def statsgd(f, grad_f, x0, lr=0.01, beta=0.9, boost_factor=100,
                stats_batch_size=2000, eps=1e-8,
                tol=1e-6, max_iter=25000):

    x = x0.copy()
    v = np.zeros_like(x)
    losses = []

    for i in range(max_iter):
        loss = f(x)
        losses.append(loss)

        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        abs_grad = np.abs(g)
        numel = abs_grad.size

        # mean/std estimation
        if numel > stats_batch_size:
            flat_view = abs_grad.ravel()
            sample_batch = np.random.choice(flat_view, stats_batch_size, replace=False)
            mean_g = np.mean(sample_batch)
            std_g = np.std(sample_batch)
        else:
            mean_g = np.mean(abs_grad)
            std_g = np.std(abs_grad)

        # z-scores and boosting mask
        z_scores = (abs_grad - mean_g) / (std_g + eps)
        boost_mask = 1.0 + (z_scores > 1.5) * boost_factor

        # momentum update
        v = beta * v + (1 - beta) * g

        # boosted update
        x -= (lr * boost_mask) * v

    return x, i + 1, losses