"""Per-lobe oscillatory power: cross-modal comparison of the aperiodic-removed spectrum.

Reads the cached per-channel spectra from the analysis store (not the peak
backbone), derives four oscillatory extractors — the detrended residual and the
modelled (specparam-peak) power, each in a log and a linear form — bins them to
2 Hz, and compares the two modalities per lobe with a subject-level frequency
cluster-permutation test (Maris-Oostenveld). Builds the
``lobe_oscillatory_diff_bootstrap`` figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient2,
    theme,
    theme_bw,
)

from pesco import config, crossmodal, store

# extractor column in the features table -> display label (plotting order)
EXTRACTORS: dict[str, str] = {
    "resid_log": "residual · log",
    "resid_lin": "residual · linear",
    "mod_log": "modelled · log",
    "mod_lin": "modelled · linear",
}
LOBE_ORDER_FULL = ["Occipital", "Parietal", "Frontal", "Temporal", "Insula"]
BIN_WIDTH = 2.0


def default_params_hash() -> str:
    """The store key for the canonical specparam settings / fit range / mode selection."""
    return store.params_hash(config.SPECPARAM_SETTINGS, config.FREQ_RANGE, config.SELECTED_MODE)


def load_lobe_features(
    db_path: str | Path,
    params_hash: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached per-channel spectra and derive the four oscillatory extractors.

    Returns ``(feat, channels_lobe)``: ``feat`` is the long features table with the
    extractor columns ``resid_log``, ``resid_lin``, ``mod_log``, ``mod_lin`` added,
    and ``channels_lobe`` maps each channel to ``subject`` and ``Lobe``. The linear
    extractors are normalised to total power per channel, so they are comparable to
    relative band power.
    """
    params_hash = default_params_hash() if params_hash is None else params_hash
    con = store.open_db(db_path)
    features = store.load(con, "features", params_hash)
    channels_lobe = store.load(con, "channels", params_hash)[
        ["channel", "dataset", "subject", "Lobe"]
    ]
    con.close()

    feat = features.merge(channels_lobe, on=["channel", "dataset"], how="left")
    psd = 10.0 ** feat["log_psd"]
    psd_ap = 10.0 ** feat["ap_log"]
    total = feat.assign(psd=psd).groupby(["dataset", "channel"])["psd"].transform("sum")
    feat["resid_log"] = feat["log_psd"] - feat["ap_log"]
    feat["resid_lin"] = (psd - psd_ap) / total
    feat["mod_log"] = feat["peak_log"]
    feat["mod_lin"] = (psd_ap * (10.0 ** feat["peak_log"] - 1.0)) / total
    return feat, channels_lobe


def _binned_matrix(feat, channels_lobe, dataset, col, bin_width):
    """channel x bin matrix for one modality/extractor, with per-row subject and lobe."""
    sub = feat.loc[feat["dataset"] == dataset, ["channel", "freq", col]]
    wide = sub.pivot(index="channel", columns="freq", values=col).sort_index(axis=1)
    centers, binned = crossmodal.bin_spectrum(
        wide.columns.to_numpy(float), wide.to_numpy(float), bin_width
    )
    meta = channels_lobe[channels_lobe["dataset"] == dataset].set_index("channel")
    return {
        "centers": centers,
        "M": binned,
        "subject": meta.loc[wide.index, "subject"].to_numpy(),
        "lobe": meta.loc[wide.index, "Lobe"].to_numpy(),
    }


def lobe_oscillatory_diff(
    db_path: str | Path,
    params_hash: str | None = None,
    extractors: Mapping[str, str] = EXTRACTORS,
    datasets: Sequence[str] = config.DATASETS,
    lobes: Sequence[str] = LOBE_ORDER_FULL,
    bin_width: float = BIN_WIDTH,
    n_perm: int = 2000,
    seed: int = 3,
) -> pd.DataFrame:
    """Cross-modal (iEEG - HD) Welch t per lobe x frequency bin, with cluster significance.

    For each extractor and lobe, the per-bin Welch t between the iEEG-patient and
    HD-subject means is thresholded into frequency clusters whose mass is tested
    against a subject-relabelling null (``crossmodal.cluster_permutation_freq``),
    controlling the family-wise error over frequency. Returns one row per
    (extractor, lobe, freq) with ``t`` and a boolean ``significant``.
    """
    feat, channels_lobe = load_lobe_features(db_path, params_hash)
    mats = {
        (col, ds): _binned_matrix(feat, channels_lobe, ds, col, bin_width)
        for col in extractors
        for ds in datasets
    }
    rows = []
    for col, label in extractors.items():
        a, b = mats[(col, datasets[0])], mats[(col, datasets[1])]
        for lobe in lobes:
            ra, rb = a["lobe"] == lobe, b["lobe"] == lobe
            _, A = crossmodal.subject_means(a["M"][ra], a["subject"][ra])
            _, Bm = crossmodal.subject_means(b["M"][rb], b["subject"][rb])
            if A.shape[0] < 2 or Bm.shape[0] < 2:
                continue
            res = crossmodal.cluster_permutation_freq(A, Bm, n_perm=n_perm, seed=seed)
            rows.append(
                pd.DataFrame(
                    {
                        "extractor": label,
                        "Lobe": lobe,
                        "freq": a["centers"],
                        "t": res["t_obs"],
                        "significant": res["sig"],
                    }
                )
            )
    diff = pd.concat(rows, ignore_index=True)
    diff["extractor"] = pd.Categorical(diff["extractor"], list(extractors.values()), ordered=True)
    diff["Lobe"] = pd.Categorical(diff["Lobe"], list(lobes), ordered=True)
    return diff


def plot_lobe_oscillatory_diff(
    diff: pd.DataFrame,
    title: str = "Cross-modal difference (dots: frequency clusters, permutation p < 0.05)",
) -> ggplot:
    """Welch-t heatmap (lobe x 2 Hz bin) per extractor, dots on significant clusters.

    Takes :func:`lobe_oscillatory_diff` output. The colour limit is the 99th
    percentile of |t| so a few extreme bins do not wash out the scale. Per the
    project convention the figure is returned only; the caller saves it.
    """
    lim = float(np.nanpercentile(np.abs(diff["t"]), 99))
    return (
        ggplot(diff, aes("freq", "Lobe", fill="t"))
        + geom_tile()
        + geom_point(
            diff[diff["significant"]], aes("freq", "Lobe"), size=0.5, color="black"
        )
        + facet_wrap("extractor", ncol=1)
        + scale_fill_gradient2(
            low="#2166AC", mid="white", high="#B2182B",
            midpoint=0, limits=(-lim, lim), name="Welch t\n(iEEG − HD)",
        )
        + labs(
            x="frequency (Hz, 2-Hz bins)", y="", title=title,
        )
        + theme_bw()
        + theme(figure_size=(9, 11), plot_title=element_text(size=11))
    )
