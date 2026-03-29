#!/usr/bin/env python
"""
Plot continuous step-density curves (KDE in log-sigma space) for the
Linear and Sigmoid beta schedules defined in src/utils/diffusion.py.

Output: results/visualizations/schedule_densities.pdf
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from src.utils.diffusion import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmas_from_betas,
)

def edm_sigma_schedule(sigma_min=0.002, sigma_max=80.0, T=20, rho=7):
    """EDM noise schedule (Karras et al. 2022), sorted low → high."""
    i = torch.arange(T, dtype=torch.float64)
    sigmas = (sigma_max ** (1/rho) + i / (T - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return torch.flip(sigmas, dims=[0])  # low → high

# ── parameters ────────────────────────────────────────────────────────────────
T      = 20
N_GRID = 1000
OUTPUT = "results/visualizations/schedule_densities.pdf"

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":        11,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

ADAPTIVE_SCHEDULE_PATH = (
    "/mnt/SSD2/constantin/diffusion-multisteps/checkpoints/"
    "KuramotoSivashinsky/exploration/run_4/greedy_schedule/schedule.json"
)

COLORS = {
    "linear":   "#e07b39",
    "cosine":   "#9b59b6",
    "edm":      "#5aab61",
    "adaptive": "#e63946",
}

# ── build sigma schedules ─────────────────────────────────────────────────────
s_linear   = sigmas_from_betas(linear_beta_schedule(T)).numpy()
s_cosine   = sigmas_from_betas(cosine_beta_schedule(T)).numpy()
s_edm      = edm_sigma_schedule(sigma_min=0.002, sigma_max=10**-0.0001, T=T).numpy()
with open(ADAPTIVE_SCHEDULE_PATH) as f:
    s_adaptive = np.array(json.load(f)["schedule"])

all_sigmas = np.concatenate([s_linear, s_cosine, s_edm, s_adaptive])
sigma_min  = all_sigmas.min() * 0.5
sigma_max  = all_sigmas.max() * 1.5

# ── KDE in log-sigma space, peak-normalised to 1 ─────────────────────────────
from scipy.stats import gaussian_kde

log_grid   = np.linspace(np.log(sigma_min), np.log(sigma_max), N_GRID)
sigma_grid = np.exp(log_grid)

def kde_cdf(sigmas, grid, bw=0.25):
    """KDE fitted in log space, integrated to give a smooth CDF from 0 to 1."""
    kde = gaussian_kde(np.log(sigmas), bw_method=bw)
    density = kde(grid)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    return cdf

c_linear   = kde_cdf(s_linear,   log_grid)
c_cosine   = kde_cdf(s_cosine,   log_grid)
c_edm      = kde_cdf(s_edm,      log_grid)
c_adaptive = kde_cdf(s_adaptive, log_grid)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.8))

def plot_cdf(ax, sigma_grid, cdf, sigmas, color, label):
    mask = (sigma_grid >= sigmas.min()) & (sigma_grid <= sigmas.max())
    # re-normalise the clipped segment so it runs 0 → 1
    c = cdf[mask]
    c = (c - c[0]) / (c[-1] - c[0])
    ax.plot(sigma_grid[mask], c, color=color, lw=2.0, label=label)

plot_cdf(ax, sigma_grid, c_linear,   s_linear,   COLORS["linear"],   "Linear schedule")
plot_cdf(ax, sigma_grid, c_cosine,   s_cosine,   COLORS["cosine"],   "Cosine schedule")
plot_cdf(ax, sigma_grid, c_edm,      s_edm,      COLORS["edm"],      r"EDM schedule ($\rho{=}7$)")
plot_cdf(ax, sigma_grid, c_adaptive, s_adaptive, COLORS["adaptive"], "Adaptive schedule (KS)")

ax.set_xscale("log")
ax.set_xlabel(r"Noise level $\sigma$")
ax.set_ylabel("Cumulative step density")
ax.set_xlim(sigma_min, sigma_max)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Distribution of diffusion steps across noise levels")

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
