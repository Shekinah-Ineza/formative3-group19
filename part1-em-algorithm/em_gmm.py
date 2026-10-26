
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUP_NAMES = ("Component 1 (shorter)", "Component 2 (taller)")


# ---------------------------------------------------------------- data
def _find_column(df, candidates):
    """Return the first column of df whose lowercase name is in candidates."""
    lookup = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")


def load_heights(csv_path, parent="father"):
    """
    Load the Galton CSV and return one unlabeled array of heights,
    plus the true labels (0 = child, 1 = parent) used ONLY to check results.
    Each parent appears once per child in the file, so we keep one row per family.
    """
    df = pd.read_csv(csv_path)
    parent_col = _find_column(df, [parent])
    child_col = _find_column(df, ["childHeight", "child_height", "childheight", "child", "height"])

    if "family" in [c.lower() for c in df.columns]:
        family_col = _find_column(df, ["family"])
        parents = df.drop_duplicates(subset=family_col)[parent_col].to_numpy(dtype=float)
    else:
        parents = df[parent_col].to_numpy(dtype=float)
    children = df[child_col].to_numpy(dtype=float)

    heights = np.concatenate([children, parents])
    true_labels = np.concatenate([np.zeros(len(children)), np.ones(len(parents))]).astype(int)
    return heights, true_labels


# ---------------------------------------------------------------- maths
def gaussian_pdf(x, mean, variance):
    """Normal density N(x | mean, variance)."""
    return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def weighted_densities(x, params):
    """pi_k * N(x | mu_k, var_k) for both groups -> shape (n, 2)."""
    x = np.atleast_1d(x).astype(float)
    return np.column_stack([
        params["pi"][k] * gaussian_pdf(x, params["mu"][k], params["var"][k])
        for k in range(2)
    ])


def log_likelihood(x, params):
    """Sum over all points of log( pi_1 N1 + pi_2 N2 )."""
    return float(np.sum(np.log(weighted_densities(x, params).sum(axis=1))))


# ---------------------------------------------------------------- EM steps
def initialize(x):
    """Start: means at the 25th and 75th percentiles, overall variance, equal weights."""
    overall_var = np.var(x)
    return {
        "mu": np.array([np.percentile(x, 25), np.percentile(x, 75)]),
        "var": np.array([overall_var, overall_var]),
        "pi": np.array([0.5, 0.5]),
    }


def e_step(x, params):
    """Responsibilities: probability each point belongs to each group (soft assignment)."""
    dens = weighted_densities(x, params)
    return dens / dens.sum(axis=1, keepdims=True)


def m_step(x, resp):
    """Update mu, var, pi using the responsibilities as weights."""
    n_k = resp.sum(axis=0)                                   # effective group sizes
    mu = (resp * x[:, None]).sum(axis=0) / n_k               # weighted means
    var = (resp * (x[:, None] - mu) ** 2).sum(axis=0) / n_k  # weighted variances
    pi = n_k / len(x)                                        # mixing weights
    return {"mu": mu, "var": var, "pi": pi}


def _snapshot(iteration, params, x):
    return {
        "Iteration": iteration,
        "mu1 (Children)": params["mu"][0], "mu2 (Parents)": params["mu"][1],
        "sigma1^2": params["var"][0], "sigma2^2": params["var"][1],
        "pi1": params["pi"][0], "pi2": params["pi"][1],
        "Log-Likelihood": log_likelihood(x, params),
    }


def fit_em(x, tol=1e-6, max_iter=2000):
    """
    Run EM until the log-likelihood changes by less than tol.
    Returns final params and the history of every iteration.
    """
    x = np.asarray(x, dtype=float)
    params = initialize(x)
    history = [_snapshot(0, params, x)]

    for i in range(1, max_iter + 1):
        resp = e_step(x, params)
        params = m_step(x, resp)
        history.append(_snapshot(i, params, x))
        change = history[-1]["Log-Likelihood"] - history[-2]["Log-Likelihood"]
        if abs(change) < tol:
            print(f"Converged after {i} iterations (log-likelihood change {change:.2e} < {tol})")
            break
    return params, history


# ---------------------------------------------------------------- outputs
def tracking_table(history, rows=3):
    """The required table: initialization + first iterations."""
    table = pd.DataFrame(history[:rows]).set_index("Iteration").round(4)
    return table.rename(columns={"mu1 (Children)": "mu1 (shorter)",
                                 "mu2 (Parents)": "mu2 (taller)"})


def classify_height(height, params, names=GROUP_NAMES):
    """Exact posterior probabilities for one test height."""
    probs = e_step(np.array([float(height)]), params)[0]
    print(f"Test height: {height}")
    for name, p in zip(names, probs):
        print(f"  P({name} | height) = {p:.6f}  ({p * 100:.2f}%)")
    print(f"  -> Most likely: {names[int(np.argmax(probs))]}")
    return dict(zip(names, probs))


def global_mean_split(x):
    """The naive method: cut at the global mean and average each pile."""
    cut = np.mean(x)
    low, high = x[x < cut], x[x >= cut]
    return {"cut": cut, "mean_low": low.mean(), "mean_high": high.mean(),
            "labels": (x >= cut).astype(int)}


def compare_methods(x, true_labels, params):
    """Accuracy and group means of the naive split vs EM, against the true labels."""
    naive = global_mean_split(x)
    em_labels = np.argmax(e_step(x, params), axis=1)
    true_means = [x[true_labels == k].mean() for k in range(2)]
    return pd.DataFrame({
        "True labels": [true_means[0], true_means[1], 1.0],
        "Global-mean split": [naive["mean_low"], naive["mean_high"],
                              np.mean(naive["labels"] == true_labels)],
        "EM": [params["mu"][0], params["mu"][1], np.mean(em_labels == true_labels)],
    }, index=["Mean of Children", "Mean of Parents", "Accuracy"]).round(3)


def plot_fit(x, params, names=GROUP_NAMES, unit="inches", save_path=None):
    """Histogram of all heights with the two fitted Gaussians."""
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    dens = weighted_densities(grid, params)
    plt.figure(figsize=(9, 5))
    plt.hist(x, bins=40, density=True, alpha=0.4, color="gray", label="All heights")
    for k in range(2):
        plt.plot(grid, dens[:, k], lw=2, label=f"{names[k]}: mu={params['mu'][k]:.2f}")
    plt.plot(grid, dens.sum(axis=1), "k--", lw=1.5, label="Mixture")
    plt.axvline(np.mean(x), color="red", ls=":", label="Global mean (naive cut)")
    plt.xlabel(f"Height ({unit})")
    plt.ylabel("Density")
    plt.title("EM fit: mixture of two Gaussians")
    plt.legend()
    _save_and_show(save_path)


def plot_log_likelihood(history, save_path=None):
    """Log-likelihood per iteration, to show convergence."""
    ll = [h["Log-Likelihood"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(ll)), ll, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("EM convergence")
    plt.grid(alpha=0.3)
    _save_and_show(save_path)


def _save_and_show(save_path):
    """Save the current figure if a path is given, then display it."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()