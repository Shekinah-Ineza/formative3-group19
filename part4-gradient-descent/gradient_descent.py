"""
Part 4 - Gradient descent for y = m1*x1 + m2*x2 + b, written with matrices.
Same data, starting values and learning rate as the manual Part 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

# ---------------------------------------------------------------- problem data (Part 3)
X = np.array([[1.0, 3.0],
              [4.0, 10.0]])      # each row is one data point (x1, x2)
Y = np.array([5.0, 6.0])
M_START = np.array([-1.0, 2.0])
B_START = np.array([1.0, 1.0])   # one bias per data point (see notebook)
ALPHA = 0.01                     # learning rate agreed for Part 3


# ---------------------------------------------------------------- model and cost
def predict(X, m, b):
    """y_hat = X m + b"""
    return X @ m + b


def mse(X, y, m, b):
    """J(m, b) = (1/n) * e^T e, with e = y_hat - y"""
    e = predict(X, m, b) - y
    return float(e @ e) / len(y)


def analytic_gradients(X, y, m, b):
    """Chain rule results: dJ/dm = (2/n) X^T e,  dJ/db = (2/n) e"""
    n = len(y)
    e = predict(X, m, b) - y
    return (2 / n) * X.T @ e, (2 / n) * e


# ---------------------------------------------------------------- SciPy derivative
def derivative(equation, point, step=1e-6):
    """Accepts ANY equation f(point) -> number and returns its derivative
    (gradient) at `point`, computed numerically with SciPy."""
    return approx_fprime(np.atleast_1d(np.asarray(point, dtype=float)), equation, step)


def scipy_gradients(X, y, m, b):
    """Numerical dJ/dm1, dJ/dm2, dJ/db1, dJ/db2 using `derivative`."""
    k = len(m)
    cost = lambda params: mse(X, y, params[:k], params[k:])
    grad = derivative(cost, np.concatenate([m, b]))
    return grad[:k], grad[k:]


def check_derivatives(X, y, m, b, tol=1e-3):
    """Compare the chain-rule gradients with SciPy's for all four parameters."""
    dm, db = analytic_gradients(X, y, m, b)
    sm, sb = scipy_gradients(X, y, m, b)
    names = [f"m{i + 1}" for i in range(len(m))] + [f"b{i + 1}" for i in range(len(b))]
    rows = []
    for name, a, s in zip(names, np.concatenate([dm, db]), np.concatenate([sm, sb])):
        rows.append((name, a, s, abs(a - s) < tol))
    print(f"{'param':<6}{'chain rule':>12}{'SciPy':>12}  match")
    for name, a, s, ok in rows:
        print(f"{name:<6}{a:>12.4f}{s:>12.4f}  {'yes' if ok else 'NO'}")
    return all(ok for *_, ok in rows)


# ---------------------------------------------------------------- gradient descent
def gradient_descent_step(X, y, m, b, alpha, show=True, label=""):
    """One update with every intermediate value visible."""
    n = len(y)
    y_hat = X @ m + b                  # 1. predictions
    error = y_hat - y                  # 2. errors
    cost = float(error @ error) / n    # 3. MSE
    dJ_dm = (2 / n) * X.T @ error      # 4. gradient for m (chain rule)
    dJ_db = (2 / n) * error            #    gradient for b
    m_new = m - alpha * dJ_dm          # 5. update
    b_new = b - alpha * dJ_db
    if show:
        print(f"--- Iteration {label} ---")
        print(f"y_hat = X m + b       = {np.round(y_hat, 4)}")
        print(f"error = y_hat - y     = {np.round(error, 4)}")
        print(f"J (MSE)               = {cost:.4f}")
        print(f"dJ/dm = (2/n) X^T e   = {np.round(dJ_dm, 4)}")
        print(f"dJ/db = (2/n) e       = {np.round(dJ_db, 4)}")
        print(f"m_new = m - a*dJ/dm   = {np.round(m_new, 4)}")
        print(f"b_new = b - a*dJ/db   = {np.round(b_new, 4)}\n")
    return m_new, b_new, cost


def run_gradient_descent(X, y, m, b, alpha, iterations, show_first=3):
    """Repeat the step; print the first `show_first` in full. Returns history."""
    history = {"m": [m.copy()], "b": [b.copy()], "cost": [mse(X, y, m, b)]}
    for i in range(1, iterations + 1):
        m, b, _ = gradient_descent_step(X, y, m, b, alpha, show=i <= show_first, label=i)
        history["m"].append(m.copy())
        history["b"].append(b.copy())
        history["cost"].append(mse(X, y, m, b))
    return m, b, history


# ---------------------------------------------------------------- plots
def _finish(title, xlabel, ylabel, save_path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_parameters(history, save_path=None):
    """How m1, m2, b1, b2 change over iterations."""
    m, b = np.array(history["m"]), np.array(history["b"])
    plt.figure(figsize=(8, 4.5))
    for i in range(m.shape[1]):
        plt.plot(m[:, i], marker=".", label=f"m{i + 1}")
    for i in range(b.shape[1]):
        plt.plot(b[:, i], ls="--", marker=".", label=f"b{i + 1}")
    _finish("Parameters over iterations", "Iteration", "Value", save_path)


def plot_error(history, save_path=None):
    """How the MSE changes over iterations."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(history["cost"], marker=".", color="crimson", label="MSE")
    plt.yscale("log")
    _finish("Error over iterations", "Iteration", "MSE (log scale)", save_path)
