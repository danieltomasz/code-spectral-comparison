"""Aperiodic model comparison: fit single models, diff any two, bootstrap CIs.

A *model* here is one ``(aperiodic_mode, peak-fit settings)`` configuration.
:func:`fit_model` fits exactly one model and returns a per-channel BIC/R²/MSE
table; the caller fits as many models as it likes (e.g. fixed-optimal,
knee-optimal, knee-looser) and tags each with a ``model`` label.
:func:`paired_delta` then differences *any two* named models, so the same
machinery covers knee-vs-fixed and knee-vs-knee (different peak settings)
comparisons without changing the functions.

Each fit is scored with the Ameen-style BIC [@ameen2025TemporallyResolvedAnalyses]

    BIC = N * log(MSE) + n_p * log(N),

where ``N`` is the number of fitted frequency bins, ``MSE`` the mean squared
error of the full log-power model fit, and ``n_p = 3 * n_peaks + n_ap``. For a
comparison of models ``a`` and ``b`` the per-channel difference is
``delta_bic = BIC_a - BIC_b`` (negative favours ``a``), summarised with a naive
channel-level and an honest subject-clustered percentile bootstrap. The subject
mapping is dataset-specific and stays in the calling notebook; the bootstrap
just expects a ``subject`` column to be present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from specparam import SpectralGroupModel

# Bootstrap defaults; override per call.
B_DEFAULT = 10_000
CI_DEFAULT = 95
SEED_DEFAULT = 3


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def bic(mse: float, n: int, n_p: int) -> float:
    """Ameen-style BIC of a single spectrum fit in log-power space."""
    return n * np.log(max(float(mse), np.finfo(float).tiny)) + n_p * np.log(n)


def load_psd_csv(path):
    """Split a saved normalized-PSD table into (psd_matrix, freqs, channels)."""
    df = pd.read_csv(path, index_col=0)
    num_cols = [c for c in df.columns if str(c).replace(".", "", 1).isdigit()]
    freqs = np.array([float(c) for c in num_cols])
    psd = df[num_cols].to_numpy(dtype=float)
    return psd, freqs, df.index.astype(str).to_numpy()


def fit_model(psd, freqs, channels, dataset, aperiodic_mode, settings, freq_range):
    """Fit ONE aperiodic mode with given peak settings over ``freq_range``.

    Returns a per-channel table (dataset, channel, aperiodic_mode, bic,
    gof_rsquared, error_mae, mse_log10, n_peaks). The MSE is the mean squared
    residual of the full model (aperiodic + all peaks) in log-power space,
    recomputed from the fitted parameters so the BIC matches the formula above.
    The caller adds a ``model`` label to distinguish configurations that share a
    mode (e.g. two knees with different peak settings).
    """
    fg = SpectralGroupModel(**settings, aperiodic_mode=aperiodic_mode, verbose=False)
    fg.fit(freqs, psd, freq_range=list(freq_range), n_jobs=1)
    ff = np.asarray(fg.data.freqs, dtype=float)
    n = ff.size
    rows = []
    for i, (channel, r) in enumerate(zip(channels, fg.results)):
        ap = fg.modes.aperiodic.func(ff, *np.asarray(r.aperiodic_fit, dtype=float))
        pp = np.asarray(r.peak_fit, dtype=float).ravel()
        peak = fg.modes.periodic.func(ff, *pp) if pp.size else 0.0
        residual = np.asarray(fg.data.power_spectra[i], dtype=float) - (ap + peak)
        mse = float(np.mean(residual**2))
        n_p = int(len(r.aperiodic_fit) + np.asarray(r.peak_fit).size)
        rows.append(
            {
                "dataset": dataset,
                "channel": channel,
                "aperiodic_mode": aperiodic_mode,
                "bic": bic(mse, n, n_p),
                "gof_rsquared": r.metrics.get("gof_rsquared", np.nan),
                "error_mae": r.metrics.get("error_mae", np.nan),
                "mse_log10": mse,
                "n_peaks": int(len(r.peak_fit)),
            }
        )
    return pd.DataFrame(rows)


def paired_delta(model_values, model_a, model_b, label_col="model"):
    """Per-channel difference between two named models in ``model_values``.

    ``delta_bic = BIC_a - BIC_b`` (negative favours ``model_a``); also ΔR² and
    the MSE ratio. One row per (dataset, channel) where both models were fit.
    Carries ``model_a``/``model_b`` so downstream summaries know the contrast.
    """
    idx = ["dataset", "channel"]
    a = model_values[model_values[label_col] == model_a].set_index(idx)
    b = model_values[model_values[label_col] == model_b].set_index(idx)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    out = pd.DataFrame(
        {
            "delta_bic": a["bic"] - b["bic"],
            "delta_r2": a["gof_rsquared"] - b["gof_rsquared"],
            "mse_ratio": a["mse_log10"] / b["mse_log10"],
        }
    ).reset_index()
    out = out.dropna(subset=["delta_bic"])
    out["model_a"] = model_a
    out["model_b"] = model_b
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _percentiles(boot, ci):
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def bootstrap_channel(values, fn, B=B_DEFAULT, ci=CI_DEFAULT, seed=SEED_DEFAULT):
    """Naive channel-level percentile bootstrap of statistic ``fn``."""
    v = np.asarray(values, dtype=float)
    n = v.size
    rng = np.random.default_rng(seed)
    boot = np.array([fn(v[rng.integers(0, n, n)]) for _ in range(B)])
    lo, hi = _percentiles(boot, ci)
    return float(fn(v)), lo, hi


def bootstrap_cluster(values, subjects, fn, B=B_DEFAULT, ci=CI_DEFAULT, seed=SEED_DEFAULT):
    """Subject-clustered percentile bootstrap: resample whole subjects."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(subjects)
    uniq = np.unique(s)
    groups = {u: v[s == u] for u in uniq}
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            fn(np.concatenate([groups[u] for u in rng.choice(uniq, uniq.size, True)]))
            for _ in range(B)
        ]
    )
    lo, hi = _percentiles(boot, ci)
    return float(fn(v)), lo, hi


# Statistics summarised with both bootstraps. mse_reduction is the median
# percentage by which model_a lowers the per-channel MSE relative to model_b.
STAT_DEFS = [
    ("median_delta_bic", "delta_bic", np.median),
    ("mean_delta_bic", "delta_bic", np.mean),
    ("median_delta_r2", "delta_r2", np.median),
    ("median_mse_reduction_pct", "mse_ratio", lambda x: (1.0 - np.median(x)) * 100.0),
]


def bootstrap_summary(paired, stat_defs=STAT_DEFS, B=B_DEFAULT, ci=CI_DEFAULT, seed=SEED_DEFAULT):
    """Point estimate + channel CI + subject-clustered CI for each statistic.

    Operates on a ``paired_delta`` frame (delta columns + ``subject``), grouped
    by ``dataset`` so each modality is summarised separately.
    """
    rows = []
    for dataset, g in paired.groupby("dataset", observed=True):
        for name, col, fn in stat_defs:
            point, c_lo, c_hi = bootstrap_channel(g[col], fn, B=B, ci=ci, seed=seed)
            _, s_lo, s_hi = bootstrap_cluster(g[col], g["subject"], fn, B=B, ci=ci, seed=seed)
            rows.append(
                {
                    "dataset": dataset,
                    "statistic": name,
                    "n_channels": int(len(g)),
                    "n_subjects": int(g["subject"].nunique()),
                    "point": point,
                    "channel_ci_lo": c_lo,
                    "channel_ci_hi": c_hi,
                    "subject_ci_lo": s_lo,
                    "subject_ci_hi": s_hi,
                }
            )
    return pd.DataFrame(rows)


def fit_quality_table(model_values, label_col="model"):
    """Descriptive median fit metrics per dataset and model."""
    return (
        model_values.groupby(["dataset", label_col], observed=True)
        .agg(
            median_r2=("gof_rsquared", "median"),
            median_mse_log10=("mse_log10", "median"),
            median_mae=("error_mae", "median"),
            median_n_peaks=("n_peaks", "median"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Raftery evidence categories (signed toward the lower-BIC model)
# --------------------------------------------------------------------------- #
def signed_category(delta_bic, label_a, label_b):
    """Signed Raftery category for one channel's ΔBIC (model_a − model_b).

    Negative ΔBIC favours ``label_a``, positive favours ``label_b``.
    """
    if delta_bic < -6:
        return f"strong {label_a}"
    if delta_bic < -2:
        return f"positive {label_a}"
    if delta_bic <= 2:
        return "weak"
    if delta_bic <= 6:
        return f"positive {label_b}"
    return f"strong {label_b}"


def category_order(label_a, label_b):
    """Ordered Raftery categories for a given (model_a, model_b) contrast."""
    return [
        f"strong {label_a}",
        f"positive {label_a}",
        "weak",
        f"positive {label_b}",
        f"strong {label_b}",
    ]


def bootstrap_category_fractions(paired, B=B_DEFAULT, ci=CI_DEFAULT, seed=SEED_DEFAULT):
    """Fraction of channels per Raftery category, with both bootstrap CIs.

    Category labels are derived from the ``model_a``/``model_b`` columns, so the
    output reads e.g. "strong knee" / "positive fixed" for a knee-vs-fixed
    contrast and "strong knee (looser)" for a knee-vs-knee one. Each channel is
    graded on its own ΔBIC; the bootstrap varies which channels (channel-level)
    or subjects (clustered) enter the resample.
    """
    label_a = paired["model_a"].iloc[0]
    label_b = paired["model_b"].iloc[0]
    categories = category_order(label_a, label_b)
    code_of = {c: i for i, c in enumerate(categories)}
    rows = []
    for dataset, g in paired.groupby("dataset", observed=True):
        codes = (
            g["delta_bic"]
            .map(lambda d: signed_category(d, label_a, label_b))
            .map(code_of)
            .to_numpy()
        )
        subjects = g["subject"].to_numpy()
        for i, name in enumerate(categories):
            indicator = (codes == i).astype(float)
            frac, c_lo, c_hi = bootstrap_channel(indicator, np.mean, B=B, ci=ci, seed=seed)
            _, s_lo, s_hi = bootstrap_cluster(indicator, subjects, np.mean, B=B, ci=ci, seed=seed)
            rows.append(
                {
                    "dataset": dataset,
                    "evidence_category": name,
                    "n_channels": int(indicator.sum()),
                    "fraction": frac,
                    "channel_ci_lo": c_lo,
                    "channel_ci_hi": c_hi,
                    "subject_ci_lo": s_lo,
                    "subject_ci_hi": s_hi,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def build_forest_plot(summary, row, stats=("median_delta_bic",), title=None, x_label=None):
    """Forest plot of ΔBIC point estimates with channel vs subject CIs.

    ``row`` is the facet-row column (e.g. ``"comparison"`` to stack several
    contrasts, or ``"statistic"`` for median/mean of one contrast). ``summary``
    is a (possibly concatenated) :func:`bootstrap_summary` frame carrying that
    column.
    """
    from plotnine import (
        aes,
        element_text,
        facet_grid,
        geom_errorbarh,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        theme,
        theme_bw,
    )

    d = summary[summary["statistic"].isin(stats)]
    long = pd.concat(
        [
            d.assign(ci_type="channel", ci_lo=d["channel_ci_lo"], ci_hi=d["channel_ci_hi"]),
            d.assign(ci_type="subject", ci_lo=d["subject_ci_lo"], ci_hi=d["subject_ci_hi"]),
        ],
        ignore_index=True,
    )
    if row == "statistic":
        long["statistic"] = long["statistic"].map(
            {"median_delta_bic": "median", "mean_delta_bic": "mean"}
        )
    return (
        ggplot(long, aes("point", "dataset"))
        + geom_vline(xintercept=0, linetype="dashed", color="grey")
        + geom_errorbarh(aes(xmin="ci_lo", xmax="ci_hi"), height=0.2)
        + geom_point(size=2.5)
        + facet_grid(f"{row} ~ ci_type", scales="free_x")
        + labs(
            x=x_label or "ΔBIC (model A − model B),  negative favours A",
            y="",
            title=title or "Bootstrap CI of per-channel ΔBIC",
        )
        + theme_bw()
        + theme(plot_title=element_text(size=10))
    )
