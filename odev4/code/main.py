import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from pathlib import Path

# True values used only to generate synthetic data
TRUE_MU = 150.0
TRUE_SIGMA = 10.0
N_OBS = 50
SEED = 42

# MCMC settings
N_WALKERS = 32
N_STEPS = 2000
BURN_IN = 500
THIN = 15

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_data():
    np.random.seed(SEED)
    return TRUE_MU + TRUE_SIGMA * np.random.randn(N_OBS)


def log_likelihood(theta, data):
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_prior(theta, mu_min=0, mu_max=300, sigma_min=0, sigma_max=50):
    mu, sigma = theta
    if mu_min < mu < mu_max and sigma_min < sigma < sigma_max:
        return 0.0
    return -np.inf


def log_probability(theta, data, mu_min=0, mu_max=300, sigma_min=0, sigma_max=50):
    lp = log_prior(theta, mu_min, mu_max, sigma_min, sigma_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)


def run_sampler(data, initial, mu_min=0, mu_max=300, sigma_min=0, sigma_max=50):
    pos = np.array(initial) + 1e-4 * np.random.randn(N_WALKERS, 2)
    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        2,
        log_probability,
        args=(data, mu_min, mu_max, sigma_min, sigma_max),
    )
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    samples = sampler.get_chain(discard=BURN_IN, thin=THIN, flat=True)
    return sampler, samples


def summarize(samples):
    median = np.median(samples, axis=0)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    return {
        "mu_median": median[0],
        "sigma_median": median[1],
        "mu_p16": p16[0],
        "mu_p84": p84[0],
        "sigma_p16": p16[1],
        "sigma_p84": p84[1],
        "mu_abs_error": abs(median[0] - TRUE_MU),
        "sigma_abs_error": abs(median[1] - TRUE_SIGMA),
        "corr": np.corrcoef(samples.T)[0, 1],
    }


def save_trace_plot(sampler, filename):
    chain = sampler.get_chain()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels = [r"$\mu$", r"$\sigma$"]
    for i in range(2):
        axes[i].plot(chain[:, :, i], alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_corner_plot(samples, filename):
    fig = corner.corner(
        samples,
        labels=[r"$\mu$ (Brightness)", r"$\sigma$ (Noise)"],
        truths=[TRUE_MU, TRUE_SIGMA],
        show_titles=True,
        title_fmt=".2f",
    )
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(title, result):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in result.items():
        print(f"{key}: {value:.4f}")


def main():
    data = generate_data()
    np.savetxt(OUTPUT_DIR / "observations.csv", data, delimiter=",", header="brightness", comments="")

    sampler, samples = run_sampler(data, initial=[140, 5])
    result = summarize(samples)
    print_summary("BASELINE", result)
    save_trace_plot(sampler, "trace_baseline.png")
    save_corner_plot(samples, "corner_baseline.png")

    sampler_small, samples_small = run_sampler(data[:5], initial=[140, 5])
    result_small = summarize(samples_small)
    print_summary("N_OBS = 5", result_small)
    save_corner_plot(samples_small, "corner_n_obs_5.png")

    sampler_prior, samples_prior = run_sampler(data, initial=[105, 8], mu_min=100, mu_max=110)
    result_prior = summarize(samples_prior)
    print_summary("NARROW PRIOR (100-110)", result_prior)
    save_corner_plot(samples_prior, "corner_narrow_prior.png")


if __name__ == "__main__":
    main()
