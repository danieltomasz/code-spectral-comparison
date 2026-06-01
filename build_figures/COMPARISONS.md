# Cross-modal comparisons

How the many iEEG-vs-HD-EEG peak comparisons are organised. The rule: **fit
once, then every comparison is `(feature matrix, subjects) → stats → plot`**.
Comparisons differ only in the *feature extractor*; everything else is shared.

## Layers

```
pesco.config            ── canonical settings (single source of truth)
        │  params_hash
pesco.store (DuckDB)    ── runs registry + cached tables, keyed by hash
        │
00_specparam_features   ── PRODUCER: fits once, writes channels/peaks/quality/features
        │  load by hash (asserts match → no drift)
comparison notebooks    ── thin: load → extractor → stats → plot
        │
pesco.crossmodal (TODO) ── add_subject, cluster bootstrap, cluster permutation
```

- **Settings live in `pesco.config`** (`SPECPARAM_SETTINGS`, `FREQ_RANGE`,
  `SELECTED_MODE`). Never re-declare them in a notebook.
- **`00_specparam_features.qmd` is the only place that fits specparam.** It
  writes to `data/interim/analysis.duckdb`, tagging every row with
  `run_id = params_hash(settings)`.
- Consumers call `store.load(con, table, params_hash)` which **raises** if the
  current settings hash isn't in the store — so results can't silently drift.
  Change settings in `pesco.config` → new hash → re-render the producer.

## Store tables (per `run_id`)

| table | grain | columns |
|---|---|---|
| `channels` | channel | `channel, subject, dataset, region, Lobe, mni_y` |
| `peaks` | fitted peak (selected mode) | `+ CF, PW, BW, band` |
| `quality` | channel × mode | `channel, dataset, mode, gof_rsquared, mse_log10, bic, n_peaks` |
| `features` | channel × frequency | `channel, dataset, freq, log_psd, ap_log` |

`features` (log-PSD + log-aperiodic) is enough to derive any residual or
modelled spectrum later **without refitting**.

## Feature extractors (the only per-comparison code)

Each returns a `channels × bins` matrix per modality (+ region/Lobe + subjects),
then identical stats/plot run on all. Per
@castanheira2025QuantifyingRhythmicArrhythmic, prefer the **modelled** rhythmic
quantities; the detrended residual is for visualisation/overlap only.

| extractor | from | Castanheira |
|---|---|---|
| modelled rhythmic-power spectrum | sum of fitted Gaussians on a CF grid (`peaks`) | ✅ recommended |
| modelled power per band | highest-peak `PW` (`peaks`) | ✅ |
| peak prevalence per band | has-peak indicator (`peaks` + `channels`) | — |
| detrended residual (log / linear) | `features` (`log_psd − ap_log`) | ⚠️ conflation demo |

## Statistic (chosen): fine bins + cluster permutation

Per (region, fine-frequency bin), compare modalities with a **subject-clustered,
two-sample** difference (resample iEEG patients and HD subjects independently),
then **Maris–Oostenveld cluster permutation across frequency** (subject-level)
to control the per-frequency multiplicity. Channels are nested in subjects, so
channel-level resampling understates uncertainty — always cluster by subject.
(To be implemented in `pesco.crossmodal`.)

## Manifest — one row per comparison

| id | quantity | resolution | statistic | notebook | output |
|---|---|---|---|---|---|
| residual_spectra | detrended residual (log+linear) | per freq, per lobe (mean) | descriptive | `aperiodic_residual_spectra.qmd` | `figures/chapter2/aperiodic_residual_spectra.svg` |
| _modelled_power_spectrum_ — TODO | modelled rhythmic power | fine freq bins × region | cluster-perm diff | — | — |
| _prevalence_ — TODO | peak prevalence | band × region | clustered bootstrap CI | (in `5_peak_prevalence_modelled_power.qmd`) | heatmaps |

## Sweeping settings / frequency ranges

The store holds **many runs at once** — each `(settings, freq_range)` combo has
its own `params_hash`, so they never collide. To sweep, add entries to
`RUN_SPECS` in `00_specparam_features.qmd` (e.g. a narrower `range_1-40`, or the
looser peak-fit) and re-render; each writes a separate run with a `label`.

A comparison notebook then picks a run by name:

```python
h = store.find_run(con, label="canonical")     # or label="range_1-40"
peaks = store.load(con, "peaks", h)
```

To compare *across* runs (e.g. how peak count changes with frequency range),
load two hashes and diff — or query the store directly:

```sql
SELECT r.label, q.dataset, median(q.n_peaks)
FROM quality q JOIN runs r USING (run_id) GROUP BY 1, 2;
```

Mind that different `freq_range` runs have different `features.freq` grids —
interpolate before a per-frequency comparison across runs.

## Add a new comparison

1. Render `00_specparam_features.qmd` once (or after changing `pesco.config`).
2. New notebook: `store.load(...)` → write one extractor → call the
   `pesco.crossmodal` stat → plot. ~30 lines.
3. Add a manifest row above.
