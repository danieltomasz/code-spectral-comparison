# build_figures

Figure-generation for the thesis chapters. **One build doc per compute group.**
Each build doc loads data, calls `pesco` plot functions, and saves figures to a
**staging mirror** of the chapter figure tree. You then eyeball staging and copy
into `chapters/figures/` by hand.

## Layers

```
data/  ──►  pesco plot fns  ──►  build doc (here)  ──►  build_figures/figures/  ──►  (manual copy)  ──►  chapters/figures/
            pure, no I/O        one per compute      staging mirror             verify              consumed by prose
```

- Build docs use **absolute** data paths (`PROJECT_DIR`), so location-independent.
- Each writes relative `figures/chapter2/...`, i.e. into `build_figures/figures/chapter2/`.
- Chapter prose embeds `figures/chapter2/...` relative to `chapters/`. Copy staging → there after review.

## Manifest — one producer per figure

| Build doc | Compute group | Chapter figures produced |
|-----------|---------------|--------------------------|
| `clusters.qmd` | cluster pipeline (specparam fit + clustering) | `combined_clusters_psd.svg`, `combined_regional_differences.svg` |
| `afnan_overlap.qmd` | Afnan band-overlap / regions-per-lobe | `afnan_overlap_raw_heatmap.svg` (+ overlap/regions variants) |
| `bandpower.qmd` | relative band-power correlation grid | `*_relative_band_power_correlations_*.svg` (not yet wired into prose) |
| `bic_knee_vs_fixed.qmd` | Ameen-style fixed-vs-knee aperiodic BIC selection | none (tables only → `data/interim/specparam_model_selection/`) |
| `5_peak_prevalence_modelled_power.qmd` | specparam peaks from the BIC-selected mode (iEEG knee / HD fixed) | `peak_prevalence_heatmap.svg`, `peak_prevalence_difference_heatmap.svg`, `peak_prevalence_agreement_scatter.svg`, `modelled_power_heatmap.svg`, `peak_centre_frequency_distribution.svg` |
| _relative_power_ — **TBD** | Afnan fig2 relative PSD | `relative_power/fig2a_relative_psd.svg` |

### Not produced here
- `pipeline.png` — diagram, built from `diagrams/*.mmd` via `make render-diagrams`.
- `frauscher_channels_and_centroids.png` — **no producer found**; source unknown / possibly hand-made. Reproducibility gap.

## Adjust a figure
1. Edit the `plot_*` function in `pesco/` (e.g. `pesco/bandpower.py`).
2. Re-render the relevant build doc here → writes to `build_figures/figures/`.
3. Eyeball, then copy the changed file into `chapters/figures/`.

## _archive/
Duplicate producers retired during dedupe (kept, not deleted):
- `02_cluster_pipeline_ctx.qmd/.ipynb` — older dupe of `clusters.qmd` (wrote to `images/`).
- `aperiodic_paper_analiza_dupe.qmd` — byte-identical dupe of `afnan_overlap.qmd`.
