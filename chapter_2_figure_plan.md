# Chapter 2 — Figure Plan

## Guiding principle (Wolfe/Siuda)
Each figure = one argument. Every analysis should be designed as 
a contribution to one figure. The figures below are ordered to 
follow the Results narrative.

---

## Overview: 3 questions → 7-9 figures

| RQ | Question | Figures |
|----|----------|---------|
| — | Methods/pipeline | Fig 1 |
| Q1 | Regional spectral features within each modality | Fig 2, 3, 4 |
| Q2 | Cross-modal preservation | Fig 5, 6 |
| Q3 | Effect of aperiodic correction | Fig 7, 8 |
| — | Supplementary: EO/EC, quality control | Fig S1, S2 |

---

## Fig 1 — Analysis pipeline overview
**Section:** Methods (or start of Results)
**Shows:** Schematic of the study design: three datasets in, 
spectral parameterisation (specparam + IRASA), region-level 
aggregation, cross-modal comparison, aperiodic correction loop.
**Format:** Flowchart/diagram
**Precedent:** Afnan 2023 Fig 1 (pipeline from iEEG atlas + MEG 
to ViEEG comparison)
**Status:** To create
**Why:** Orients the reader; all subsequent figures are outputs 
of steps shown here.

---

## Fig 2 — Spectral clustering: iEEG and HD-EEG
**Section:** Results §"Reproducing the Frauscher oscillatory atlas"
**Shows:** K-means cluster mean spectra, side by side: 
(A) iEEG 8 clusters, (B) HD-EEG (Mantini) 6 clusters. 
Highlight the no-peak set in each.
**Format:** Panel of line plots (mean ± SEM per cluster)
**Precedent:** Frauscher 2018 Fig 3 (cluster spectra)
**Status:** DONE (two separate SVGs exist). Combine into one 
two-panel figure.
**Data needed:** Already have both.

---

## Fig 3 — Regional broadband maps (exponent and offset)
**Section:** Results §"Regional broadband profiles"
**Shows:** 
(A) Exponent per MICCAI region, iEEG — bar/dot plot ordered 
by lobe, or brain surface map
(B) Same for HD-EEG (Mantini)
(C) Offset per region, iEEG
(D) Offset per region, HD-EEG
Optionally: specparam vs IRASA exponent agreement as inset 
scatter or separate panel (E).
**Format:** Bar/dot plots grouped by lobe (like Frauscher's 
region-level results) OR cortical surface maps (like 
Kalamangalam 2020, Shafiei 2020). Bar/dot plots are simpler 
and more honest given MICCAI atlas granularity.
**Precedent:** 
- Shafiei 2020: cortical surface maps of exponent gradients
- Kalamangalam 2020: brain surface color-coded by spectral 
  parameters
- Janiukstyte 2023: bar plots per region/band
**Status:** specparam computed but not summarized → need to 
aggregate per region, compute means, and plot
**Data needed:** Aggregate specparam output per MICCAI region 
for iEEG and Mantini. Compute IRASA exponent (line fit in 
log-log on IRASA fractal component) if you want the 
agreement panel.

---

## Fig 4 — Regional oscillatory peak detection heatmap
**Section:** Results §"Regional oscillatory profiles"  
**Shows:** Heatmap: regions (rows) × frequency bins (columns), 
colored by significance of peak (KS test). 
(A) iEEG, (B) HD-EEG (Mantini).
Mark the frequency binning difference between datasets 
(different bin boundaries).
**Format:** Heatmap with binary or graded coloring (p-value or 
yes/no significant)
**Precedent:** Frauscher 2018 Fig 4 (the core result figure 
of the atlas — region × frequency significant peaks)
**Status:** To compute and plot
**Data needed:** KS test results per region × frequency bin, 
both datasets. This is the Frauscher procedure applied to 
your data.

---

## Fig 5 — Cross-modal correspondence: broadband parameters
**Section:** Results §"Cross-modal correspondence"
**Shows:** Scatter plots:
(A) iEEG exponent vs HD-EEG exponent, per region 
    (each dot = one MICCAI region). Spearman rho + permutation p.
(B) iEEG offset vs HD-EEG offset, per region.
(C) Optionally: specparam exponent vs IRASA exponent 
    within each dataset (methodological consistency check).
**Format:** Scatter plots with regression line, rho, p-value
**Precedent:** 
- Janiukstyte 2023 Fig 3 (cross-modality correlation per band, 
  scalp EEG vs iEEG vs MEG)
- Afnan 2023 (ViEEG vs iEEG spectral comparison per ROI)
**Status:** To compute — requires aggregated regional parameters 
from both datasets
**Data needed:** Region-level mean exponent and offset for iEEG 
and Mantini. Atlas mapping (DK → MICCAI) already described 
in Methods.

---

## Fig 6 — Cross-modal correspondence: oscillatory peaks
**Section:** Results §"Cross-modal correspondence"
**Shows:** 
(A) Peak prevalence correlation: for each frequency band, 
    scatter of "proportion of regions with significant peak" 
    in iEEG vs HD-EEG. Or: region × band agreement matrix 
    (how often do iEEG and HD-EEG agree on peak presence).
(B) Dominant peak frequency per region: iEEG vs HD-EEG scatter.
**Format:** Scatter plots or agreement heatmap
**Precedent:** Afnan 2023 (comparison of oscillatory peaks 
per ROI between ViEEG and iEEG, showing which bands/regions 
agree)
**Status:** To compute
**Data needed:** Peak detection results from Fig 4 for both 
datasets.

---

## Fig 7 — Effect of aperiodic correction on peak detection
**Section:** Results §"Effect of aperiodic correction"
**Shows:** The key "before vs after" comparison.
(A) Heatmap: region × frequency bin, BEFORE correction (= Fig 4A 
    repeated or referenced)
(B) Same heatmap AFTER specparam correction
(C) Same heatmap AFTER IRASA correction
(D) Difference/summary: which peaks survived both corrections 
    (robust), which disappeared (potentially confounded), 
    which appeared (newly revealed).
**Format:** Three aligned heatmaps + summary panel. 
Color code: green = robust, red = lost after correction, 
blue = new after correction.
**Precedent:** Afnan 2023 finding that "MEG-estimated spectra 
were more comparable to iEEG after aperiodic components were 
removed" — but they showed this with overlaid spectra rather 
than heatmaps. Your heatmap approach is more systematic.
**Status:** To compute — needs aperiodic correction + re-running 
Frauscher procedure on corrected spectra
**Data needed:** 
- specparam aperiodic fit subtracted from each PSD → flattened 
  spectra → re-run KS peak detection
- IRASA fractal subtracted → oscillatory residual → same
- For iEEG first (the cleaner dataset), then for HD-EEG

---

## Fig 8 — Does aperiodic correction improve cross-modal agreement?
**Section:** Results §"Effect of aperiodic correction" 
(or bridge to Discussion)
**Shows:** Cross-modal peak agreement (from Fig 6) recomputed 
after aperiodic correction. 
(A) Agreement BEFORE correction (= Fig 6 summary)
(B) Agreement AFTER specparam correction
(C) Agreement AFTER IRASA correction
This directly tests the Afnan 2023 finding (MEG-iEEG agreement 
improves after aperiodic removal) in your HD-EEG context.
**Format:** Paired bar plot or heatmap comparison
**Status:** To compute (depends on Fig 6 + Fig 7)
**Data needed:** Corrected peak detection for both datasets.

---

## Supplementary figures

### Fig S1 — Eyes-open vs eyes-closed (Nencki)
**Section:** Results §"EO/EC comparison"
**Shows:** Within-dataset comparison of spectral parameters 
across conditions. 
(A) Exponent: EO vs EC per region
(B) Peak prevalence: EO vs EC
**Purpose:** Quantify the EO/EC confound between iEEG (EC) 
and Mantini (EO)
**Status:** Blocked — needs Nencki data processing

### Fig S2 — Specparam fit quality
**Section:** Methods or Supplementary
**Shows:** 
(A) Distribution of R² values across channels/sources
(B) Example spectra: good fit vs poor fit (near-flat slope, 
    low R²)
(C) Sensitivity check: regional exponent with vs without 
    R² < 0.90 exclusion
**Purpose:** Supports the methodological caveats paragraph 
(lines 338 of the draft)
**Status:** To compute from existing specparam output

---

## Priority order for figure production

Given what you have now (iEEG + Mantini, specparam computed 
but not summarized, two clustering figures done):

### Phase 1 — Quick wins (existing data, just needs aggregation)
1. **Fig 2** — Combine existing cluster SVGs into one panel
2. **Fig 3** — Aggregate specparam output → regional bar/dot plots
3. **Fig S2** — R² distribution from specparam output

### Phase 2 — Core analyses (needs new computation)
4. **Fig 4** — Run Frauscher peak detection (KS tests) per region
5. **Fig 5** — Cross-modal scatter (exponent, offset) once Fig 3 
   is done for both datasets
6. **Fig 6** — Cross-modal peak agreement once Fig 4 is done

### Phase 3 — Aperiodic correction (the novel contribution)
7. **Fig 7** — Subtract aperiodic fits, re-run peak detection
8. **Fig 8** — Recompute cross-modal agreement after correction

### Phase 4 — Replication and supplementary
9. **Fig S1** — Process Nencki dataset, run EO/EC comparison
10. **Fig 1** — Pipeline diagram (can be made last)

---

## Mapping figures to the narrative (Wolfe's principle)

Each figure drives one paragraph of results:

- Fig 2 → "Clustering reproduces Frauscher; HD-EEG shows fewer 
  clusters due to spatial smoothing"
- Fig 3 → "Regional broadband parameters show [gradient/pattern]; 
  specparam and IRASA [agree/disagree]"
- Fig 4 → "Peak detection reveals [which regions, which bands]; 
  frequency binning differs between datasets"
- Fig 5 → "Cross-modal broadband correspondence is [moderate/weak]; 
  exponent [better/worse] than offset"
- Fig 6 → "Oscillatory peak agreement is [band-dependent]; alpha 
  most preserved, gamma least"
- Fig 7 → "After correction, [N] peaks disappear, [M] survive; 
  most affected: [bands/regions]"
- Fig 8 → "Cross-modal agreement [improves/unchanged] after 
  correction, consistent with / departing from Afnan 2023"o