# Chapter 2 — Consolidated Outline

**Framing:** Option A (ground-clearing for later thesis chapters)
**Format:** Thesis chapter, reviewed by neuroimaging/methods-literate examiners
**Other chapters:** Standalone (Ch. 3 published, Ch. 4 to be released as preprint); cross-references should be real but not forced
**Figures:** 7 main + 2 supplementary (see separate figure plan)
**Budget:** ~12 hours writing over the next week

---

## 1. Introduction

### §1.1 Opening — what this chapter does for the thesis

- Later chapters of this thesis use aperiodic exponent, aperiodic offset, and oscillatory peak parameters as regional descriptors of brain activity, measured in source-reconstructed EEG (Ch. 3) and MEG (Ch. 4).
- Whether these measures can be interpreted physiologically depends partly on whether the regional patterns they produce are stable across recording modalities, or whether they mainly reflect method-specific artefacts.
- Two questions for this chapter:
  1. Do these spectral features show consistent regional organisation in both invasive and non-invasive recordings?
  2. Where cross-modal agreement is partial, what constrains it?
- The purpose is not to produce a new normative atlas but to establish what can reasonably be claimed about regional spectral features before using them elsewhere in the thesis.
- > **Tone:** direct, no literature citations in opening paragraph.

### §1.2 Regional spectral organisation in invasive recordings

- iEEG documents region-specific oscillatory and broadband patterns — the reference picture for this chapter.
- Cite [@frauscher2018AtlasNormalIntracranial] (the atlas, k-means + no-peak reference set approach, 1785 channels, 38 regions) and [@kalamangalam2020NeurophysiologicalBrainMap] (parameterisation re-analysis of the same atlas).
- Include Frauscher's finding that the 1/f exponent also varies regionally (steeper posterior, flatter frontal) — this is the direct precedent for the broadband analysis in this chapter.
- > **Context depth:** two or three sentences on *why* these are the reference, not just that they exist. Frauscher is the canonical normative iEEG atlas; Kalamangalam is the parameterised version. Together they define what "regional organisation in iEEG" means empirically.

### §1.3 Periodic and aperiodic decomposition

- Keep your existing paragraph, minor typo fixes only ("certain assumption" → "certain assumptions"; "two components types" → "two component types").
- **Backward link to Ch. 1:** the paragraph already references Ch. 1's treatment of the operational-vs-physiological distinction. Keep that.
- Include your existing closing logic: failing to model the broadband background can produce apparent oscillatory differences that reflect spectral variation in the background; the converse can also happen.

### §1.4 Non-invasive recordings and source reconstruction

- iEEG has direct neural access but opportunistic sampling restricted to non-epileptogenic channels in epilepsy patients.
- HD-EEG with source reconstruction has whole-brain coverage but depends on inverse modelling assumptions that introduce spatial blurring and spectral distortion [@tait2021SystematicEvaluationSource; @piastra2020ComprehensiveStudyElectroencephalography].
- The SNR gap is substantial: [@ball2009SignalQualitySimultaneously] reported 20–100× higher SNR in invasive vs simultaneously recorded scalp EEG.
- Choice of inverse method, electrode density, and head model all affect spectral properties [@liu2018DetectingLargeScaleBrain; @lai2018ComparisonScalpSourcereconstructed].
- If a regional pattern seen in iEEG is recoverable in HD-EEG despite these differences, this is evidence the pattern reflects organised brain activity rather than a property of one recording method.
- > **Restored context:** the source-reconstruction limitations deserve more than a passing mention — they are the reason the cross-modal question matters in the first place.

### §1.5 Key precedent: the Afnan 2023 MEG–iEEG comparison

- [@afnan2023ValidatingMEGSource] validated MEG source imaging against the Frauscher iEEG atlas in 45 healthy participants, using the same 38 regions.
- Two key findings: MEG-estimated spectra were more comparable to iEEG after aperiodic components were removed; lateral regions were recovered more reliably than deep or medial ones.
- This chapter asks the analogous question for HD-EEG and extends it to aperiodic features alongside oscillatory ones.
- > **Context depth:** Afnan is not just "the precedent" — it is the template for the analytic strategy used here (Spearman on regional parameters, permutation inference, aperiodic correction as a robustness check). This deserves explicit statement.
- > **Honest positioning:** MEG source reconstruction has better spatial properties than HD-EEG, so cross-modal agreement with iEEG should in principle be lower here. This should be flagged now, not only in the Discussion.

### §1.6 Eyes-open / eyes-closed mismatch

- One constraint needs stating at the outset.
- The iEEG atlas was recorded eyes closed; the Mantini HD-EEG dataset used for comparison was recorded eyes open.
- Eye state affects oscillatory activity — particularly alpha power, with strong posterior effects — and, more modestly, the aperiodic exponent and offset [@geller2014EyeClosureCauses].
- Where cross-modal agreement is observed, it is observed despite this difference, not controlling for it.
- Where agreement is weak, condition and modality cannot be fully separated with the data used here.
- Returned to in the Discussion.
- > **Why this paragraph is in the Intro, not hidden in caveats:** the EO/EC mismatch shapes what the chapter can and cannot claim about cross-modal preservation. Promoting it up-front reframes the reader's expectations before they encounter the results.

### §1.7 Aims and predictions

Three questions, addressed in parallel for oscillatory and aperiodic features:

1. How do oscillatory and broadband slope spectral features vary across cortical regions within each modality?
2. To what extent is the regional organisation of these features preserved across iEEG and HD-EEG source-reconstructed recordings?
3. How does removing the broadband slope component change the interpretation of regional oscillatory patterns?

**Weak a priori predictions, stated plainly:**

- Regional organisation is present in both modalities (i.e., spectral features vary systematically across regions, not randomly).
- Cross-modal agreement is partial, stronger for alpha-band features than for other oscillatory features, following [@afnan2023ValidatingMEGSource].
- Aperiodic correction changes the regional oscillatory pattern in both modalities, with direction of change dependent on whether the raw-spectrum peak was driven by narrowband activity or by broadband shape.

**Robustness strategy:** specparam and IRASA are used in parallel as complementary decomposition methods. The two methods rest on different assumptions; convergence across them is evidence of robustness to decomposition choice. Chapter 1 (§X) discusses the range-dependence of the exponent [@boncompte2026AperiodicExponentBrain; @gerster2022SeparatingNeuralOscillations]; these caveats apply throughout.

> **Restored structure:** your original aims paragraph closed with the specparam/IRASA robustness point. Keep it, but explicitly label it as a robustness strategy rather than a methodological aside.

### §1.8 Operationalisation

One short paragraph defining the key constructs used in this chapter:

- "Regional organisation" = systematic variation of spectral parameters across the 38 MICCAI atlas regions.
- "Cross-modal preservation" = Spearman rank correlation of regional parameter values between iEEG and HD-EEG, assessed against a permutation null.
- "Oscillatory peak" = frequency bin where the per-region spectral distribution differs significantly from the no-peak reference set (KS test, Dunnett-corrected), following [@frauscher2018AtlasNormalIntracranial].
- "Aperiodic correction" = subtraction of the aperiodic component (specparam fit or IRASA fractal) from the raw PSD before peak detection.

> **Why include this:** examiners read for it. It also forces you to be explicit about where the constructs are defined and where they are measured. Short paragraph, but it pulls several methodological choices into focus before Methods.

### §1.9 Forward links to the thesis

- One short paragraph, optional but useful.
- The aperiodic exponent established here as a regional descriptor returns in Ch. 3 as a predictor of cognitive outcomes in aging, and in Ch. 4 as a correlate of stimulus-driven gamma synchrony.
- Chapters 3 and 4 are standalone studies, but both assume that the aperiodic parameters they measure carry interpretable regional information. This chapter tests that assumption.
- > **Honest framing given standalone chapters:** don't oversell the connection. Ch. 3 and 4 work without Ch. 2, but Ch. 2 adds the interpretive scaffold that makes the thesis cohere.

---

## 2. Methods

### §2.0 Analytic strategy and pre-specification

New short section. One paragraph.

- Primary analyses (planned a priori): Spearman rank correlation across regions for exponent, offset, peak prevalence per band, and dominant peak frequency; permutation inference (10,000 permutations) for each; aperiodic correction followed by re-running peak detection.
- Robustness analyses: specparam + IRASA run in parallel on the same data; sensitivity check on the R² < 0.90 exclusion.
- Post-hoc decisions: state plainly which analytic choices were made after inspecting the data — for example, the dataset-specific frequency binning for HD-EEG was necessitated by the failure of the 4% condition; the decision to focus cross-modal comparison on rank correlation rather than parametric correlation was made in light of the normalisation mismatch between datasets.
- > **Why:** methodologically literate examiners will read for this. Honest labelling of a posteriori decisions is a strength, not a weakness.

### §2.1 Datasets

- **iEEG atlas (Frauscher):** 1785 channels, 38 MICCAI regions, 60 s resting wakefulness eyes closed, 0.5–80 Hz, 200 Hz sampling, unit-total-power normalised.
- **HD-EEG (Mantini / Liu 2018):** 19 healthy adults, 256-channel EGI, 5 min resting state eyes open, sLORETA on individual anatomy.
- Only one HD-EEG dataset in this chapter; condition differs from iEEG.
- > Drop all commented-out Nencki/Shou text.

### §2.2 Source reconstruction

- Mantini: sLORETA on individual anatomy, as in [@liu2018DetectingLargeScaleBrain].
- This chapter does not re-run source reconstruction; source time courses were obtained from the original pipeline and parcellated to MICCAI regions via Desikan-Killiany → MICCAI mapping (supplementary table).

### §2.3 Preprocessing

- iEEG: Frauscher pipeline (0.5–80 Hz band-pass, 200 Hz resampling, 60 s artefact-free segments).
- HD-EEG: Liu 2018 pipeline.

### §2.4 Power spectral density estimation

- Welch's method: 2 s window, 1 s overlap, 0.5–80 Hz, 0.5 Hz frequency resolution.
- iEEG: unit-total-power normalisation (Frauscher procedure).
- HD-EEG: source-current units (nAm²/Hz).
- **Normalisation mismatch:** explicit sentence here flagging that the two datasets are in different units, and that this is why Spearman rank correlation is used for cross-modal comparison. Offset comparisons are interpreted with particular caution given the offset–exponent structural coupling [@hill2022PeriodicAperiodic; @merkin2023AperiodicBrain].

### §2.5 Oscillatory peak detection — Frauscher procedure

- K-means clustering on 160 frequency bins (Euclidean, 100 repetitions); increasing k until a no-peak group is identified (mean spectrum lower than the maximum among other groups); 50% nearest to centroid retained as the no-peak set.
- 22 frequency intervals defined such that each bin contains ≥ 4% of total mean power (iEEG).
- Peak presence per region × bin: one-sided two-sample KS test of per-region spectral distribution against no-peak set, at α = 0.05 with Dunnett's correction across 42 regions × 22 bins.
- **Deviation for HD-EEG:** the 4% condition failed for source-reconstructed spectra (power concentrated in alpha, higher bins below threshold); dataset-specific bins were computed. Noted as a methodological caveat.

### §2.6 Spectral parameterisation — specparam

- Fixed aperiodic mode, 1–40 Hz, peak width 1–8 Hz, max 6 peaks, min peak height 0.1, threshold 2 SD.
- Aperiodic parameters of interest: exponent χ, offset b.
- Periodic parameters: centre frequency, power, bandwidth.
- R² and MAE recorded; R² < 0.90 flagged for sensitivity check.

### §2.7 Aperiodic estimation — IRASA

- Resampling factors h = 1.1–1.9, step 0.05.
- Geometric mean of up/down-sampled spectra; median across h yields fractal component.
- Exponent estimated by linear fit in log-log space, 1–40 Hz, matching the specparam fitting range.
- Complementary to specparam; used as robustness check, not as an independent analysis.

### §2.8 Regional aggregation and cross-modal correspondence

- iEEG: mean across all channels in each MICCAI region.
- HD-EEG: per-subject-per-label values, then averaged across subjects for group-level regional estimates.
- Cross-modal Spearman rank correlation across the 38 regions for each parameter (exponent, offset, peak prevalence per band, dominant peak frequency).
- Permutation null (10,000 permutations, shuffling region labels) — controls for the small number of spatial units and avoids parametric assumptions.
- DK → MICCAI mapping: anatomical correspondence; where multiple DK labels map to one MICCAI region, parameter values averaged.

### §2.9 Oscillatory re-analysis after aperiodic correction

- Two parallel corrections: specparam aperiodic component subtracted from each PSD; IRASA fractal component subtracted from each PSD.
- Frauscher peak detection procedure re-applied to both corrected PSDs.
- **Stability defined per region × bin:** peak present before correction and after correction → retained; present before, absent after → lost after correction; absent before, present after → newly detected after correction.
- Agreement between specparam and IRASA corrections reported as a secondary check.

---

## 3. Results

### §3.1 Reproducing the Frauscher atlas (reference baseline)

- iEEG: 8 clusters, no-peak set identified, qualitatively matches Frauscher classification.
- HD-EEG: 6 clusters, no-peak set identified, narrower range of spectral shapes consistent with spatial smoothing from source reconstruction.
- **Figure 2:** combined two-panel figure (A: iEEG 8 clusters, B: HD-EEG 6 clusters).
- **Deviation:** 4% binning condition fails for HD-EEG; dataset-specific bins used. One sentence noting that the lower cluster count and dataset-specific bins are both reasons direct cluster-to-cluster comparison is not meaningful; downstream regional outputs are the unit of comparison.

### §3.2 Regional broadband profiles within each modality (Aim 1)

- Exponent and offset per region for iEEG and HD-EEG, with both specparam and IRASA.
- Report:
  - Spatial pattern within each modality (any lobar or posterior-anterior gradient, without over-claiming).
  - specparam vs IRASA agreement within each dataset (Spearman ρ across regions).
  - Range of exponent and offset values per modality.
  - Any regions with outlier values or unusually low model fit.
- **Figure 3:** bar/dot plots per region, grouped by lobe. Panels A–D: iEEG exponent, HD-EEG exponent, iEEG offset, HD-EEG offset. Panel E: specparam vs IRASA agreement scatter.
- **Figure 3C (promoted from supplementary):** R² distribution for HD-EEG specparam fits, with threshold marked. This defends the sensitivity-check paragraph in §4.4.
- Effect-size reporting: "ρ = X, corresponding to roughly Y% of regional variance shared between methods."

### §3.3 Regional oscillatory profiles within each modality (Aim 1)

- Per-region, per-bin peak prevalence in both datasets.
- Dominant peak frequency per region.
- **Figure 4:** heatmap, regions × frequency bins, colour-coded by peak presence (from KS test). Panels A: iEEG; B: HD-EEG. Frequency binning difference between datasets annotated.
- Report which bands dominate which regions in each dataset; flag discrepancies.

### §3.4 Cross-modal correspondence (Aim 2)

- Spearman rank correlation across 38 regions between iEEG and HD-EEG for:
  - Aperiodic exponent (specparam and IRASA)
  - Aperiodic offset (rank-only, given units mismatch)
  - Peak prevalence per band
  - Dominant peak frequency per region
- Permutation p-values for each.
- **Figure 5:** scatter plots with Spearman ρ and permutation p — exponent panel, offset panel, specparam vs IRASA consistency panel.
- **Figure 6:** cross-modal peak agreement — pooled band × region scatter as main panel; band-split version to supplementary.
- Report any posterior-vs-non-posterior difference in agreement — this is partial evidence bearing on the EO/EC confound, not a clean test, but worth stating.
- Effect-size interpretation: "ρ = 0.4 corresponds to roughly 16% of shared variance across regions."

### §3.5 Effect of aperiodic correction on oscillatory findings (Aim 3)

- Within each modality: how many peaks are retained, lost, newly detected after specparam correction; same for IRASA correction.
- Cross-modal: cross-modal peak agreement recomputed after each correction, compared to pre-correction agreement.
- **Figure 7:** three aligned heatmaps (before, after specparam, after IRASA) with a summary panel. Colour: retained / lost after correction / newly detected after correction (not "robust"/"confounded" to avoid overclaiming).
- **Figure 8:** cross-modal agreement before vs after aperiodic correction — paired comparison. Framing: "examines whether the Afnan et al. pattern extends from MEG to HD-EEG."
- Agreement between specparam-based and IRASA-based conclusions reported as a robustness check.

---

## 4. Discussion

### §4.1 Regional organisation is detectable in both modalities

- Within-modality, both iEEG and HD-EEG show systematic regional variation in oscillatory and aperiodic features.
- This supports the premise that spectral features used elsewhere in the thesis are defensible regional descriptors of brain activity.
- Broadly consistent with [@frauscher2018AtlasNormalIntracranial] and [@afnan2023ValidatingMEGSource] as prior references, not competing framings.
- If the posterior-anterior frequency gradient is present in the data, one sentence on convergence with [@mahjoory2020FrequencyGradientHuman] and [@kalamangalam2020NeurophysiologicalBrainMap] — but not a paragraph-length engagement.
- > **Restored depth:** the gradient literature can come back here if the results support it. What you were cutting was its appearance in the Introduction framing. Discussion use is fine.

### §4.2 Partial cross-modal preservation and its limits

- Summarise the cross-modal correspondence pattern: which features correlate, which don't, which regions contribute to agreement/disagreement.
- Consistent with Afnan 2023 where applicable — partial agreement, improvement after aperiodic correction, lateral regions better recovered than deep/medial.
- **Where disagreement comes from (multiple sources, honestly unseparated):**
  - Source reconstruction blurs spatial information, especially for deep/medial regions [@tait2021SystematicEvaluationSource].
  - iEEG sampling bias toward non-epileptogenic regions in epilepsy patients.
  - EO/EC mismatch between datasets; strongest effect posteriorly, where alpha is dominant.
  - Can cite [@geller2014EyeClosureCauses] for published ECoG-level quantification of the eye-closure effect, as a way of bounding how large the EO/EC contribution could plausibly be without running a within-modality sensitivity analysis of my own (which would require a second HD-EEG dataset not included here).
- **Where agreement is strongest:** note which features and regions, relate to [@jensen2026AlphaRhythmPhysiology] for alpha's spatial stability across tasks and methods.
- **Comparison with Afnan 2023 in both directions:** MEG source reconstruction has better spatial properties than HD-EEG, so lower agreement is expected. This chapter extends to aperiodic features, which Afnan treated only as a correction step.
- > This is the heavy paragraph. 3–4 paragraphs is appropriate for a thesis Discussion.

### §4.3 Aperiodic correction changes the oscillatory interpretation

- Summary of which peaks were retained vs lost after correction.
- Interpretation: peaks lost after correction were likely driven by regional variation in broadband shape rather than narrowband rhythmic activity — same conclusion as Afnan 2023 in a different modality.
- Implication for the wider literature that does not model the aperiodic background: some of what is reported as regional oscillatory variation may be aperiodic variation [@donoghue2020ParameterizingNeuralPower; @gerster2022SeparatingNeuralOscillations]. The data here contribute an additional test case at the atlas-region level.
- specparam vs IRASA convergence: if the two methods give convergent conclusions, that strengthens the result; if not, the divergence itself is informative.

### §4.4 Methodological caveats

Compact prose form, promoted from the currently commented list:

- **Parameterisation is model- and range-dependent.** Fitting range set to 1–40 Hz for both specparam and IRASA; different ranges would yield different exponents [@boncompte2026AperiodicExponentBrain; @gerster2022SeparatingNeuralOscillations; @donoghue2021MethodologicalConsiderationsStudying]. This is an instance of the "forking paths" concern developed in Chapter 1.
- **Normalisation mismatch** between iEEG (unit total power) and HD-EEG (source-current units) — reason Spearman rank is used and why offset comparisons are treated with caution. Offset–exponent structural coupling in specparam's fixed model means normalisation differences propagate to both parameters jointly.
- **Frequency binning deviation** for HD-EEG (4% condition failed).
- **Low-R² source-reconstructed spectra** likely reflect leakage/noise rather than flat neural spectra [@tait2021SystematicEvaluationSource; @brake2024NeurophysiologicalBasisAperiodic]. Sensitivity check: R² < 0.90 exclusion did / did not change regional rankings [to fill in].
- **iEEG sampling bias** toward non-epileptogenic tissue in epilepsy patients.
- **Single HD-EEG dataset.** No within-modality replication; no within-dataset EO/EC contrast. A second HD-EEG dataset with eyes-closed recordings would allow both. This remains for future work.
- > **Restored structure:** bullet-like but in prose form. Don't let this balloon. Three paragraphs maximum.

### §4.5 Implications for later thesis chapters

Short section (1 paragraph is enough given the standalone nature of the other chapters).

- The aperiodic exponent and offset as measured in source-reconstructed EEG have partial but not full cross-modal agreement with iEEG. This means their interpretation in Ch. 3 (aging and education in LEMON) and Ch. 4 (gamma synchrony in MEG) should treat them as regional descriptors that carry real signal, with the qualification that source reconstruction introduces region-specific biases that are likely largest for deep and medial structures.
- No stronger claim can be made honestly with one HD-EEG comparison dataset.
- > **Honest framing:** since the other chapters are standalone, this section is explicit that Ch. 2 offers interpretive scaffolding rather than a precondition.

---

## 5. Conclusion

One paragraph:

- Spectral features show regional organisation in both iEEG and HD-EEG with source reconstruction.
- Organisation is partly preserved across the two modalities.
- Aperiodic correction changes some oscillatory conclusions, consistent with [@afnan2023ValidatingMEGSource] in MEG.
- These findings support the use of aperiodic and oscillatory parameters as regional descriptors in the thesis's later studies, with the specific methodological constraints named above.
- The EO/EC difference between the available datasets is the main limitation that cannot be resolved with the data used here.

---

## 6. References *(auto)*

---

## Cuts retained from Option A plan

- "Ideas" section at end of current draft: delete, absorb P-A gradient point into §4.1 if results support it.
- Mellem 2017 (intrinsic multi-band biases): delete — not addressed by any of this chapter's analyses.
- Gao 2020 and Shafiei 2020 (timescale hierarchy): delete — you already flagged as not fitting.
- Armonaite 2026 (scale-free fingerprints): delete from Intro; can return in Discussion §4.1 only if the data show fingerprint-like structure.
- Janiukstyte 2023 and Kozma 2024 from Intro: delete; Janiukstyte can return in §4.2 as a point of comparison.
- All commented-out Nencki and Shou sections across Methods: delete.
- Jensen 2026 at current line 48 (spurious travelling waves): delete — not relevant.
- Jensen 2026 at current line 50 (alpha stability): keep, in §4.2.

## What I restored from the first outline version

- Explicit §1.2 paragraph citing Frauscher and Kalamangalam as reference picture, rather than collapsing them into §1.1.
- Fuller §1.4 on source-reconstruction limitations with multiple citations — not one passing sentence.
- Robustness strategy language (specparam + IRASA in parallel) as a labelled component of the Aims section.
- Discussion §4.1 can bring back posterior-anterior gradient convergence if the data support it — cut from Intro, usable in Discussion.
- Discussion §4.4 explicitly labels the forking-paths issue as an instance of Chapter 1's critique. One sentence. Cohesion with Ch. 1 without forced cross-referencing.
- Explicit statement in §1.5 that HD-EEG is expected to have lower cross-modal agreement than MEG — context for interpreting §4.2.

## What I added that is genuinely new

- **§1.7 predictions** (three short statements, not hypotheses).
- **§1.8 operationalisation** (one paragraph defining the four key constructs).
- **§1.9 forward links** (optional short paragraph noting Ch. 3 and Ch. 4 usage).
- **§2.0 analytic strategy and pre-specification** (one paragraph on what was planned vs post-hoc).
- **§4.5 implications for later chapters** (short, honest, aligned with standalone-chapter reality).
- Effect-size interpretation language in Results (not only p-values and ρ, but what ρ = X means for shared regional variance).

## Realistic week-long plan

| Day | Task |
|---|---|
| 1 | Trim Introduction — apply cuts, draft new §1.1, §1.5, §1.6 paragraphs. Add §1.7 predictions, §1.8 operationalisation, §1.9 forward links. Tighten §1.4. |
| 2 | Methods: delete commented-out sections, write §2.0 analytic strategy, add normalisation statements to §2.4 and §2.8. |
| 3 | Results 3.2, 3.3 — straight reporting from existing analyses; produce Fig 3, Fig 4. |
| 4 | Results 3.4 — cross-modal correspondence; produce Fig 5, Fig 6. |
| 5 | Results 3.5 — aperiodic correction; produce Fig 7, Fig 8. |
| 6 | Discussion 4.1, 4.2, 4.3. |
| 7 | Discussion 4.4, 4.5, Conclusion, Abstract, final read-through. |

Fig 1 (pipeline diagram) and Fig S2 (specparam fit quality) fit into day 1–2 slack or day 7 if time allows. Supplementary Figure S3 (normalisation sensitivity) is optional and can be dropped if time tightens.
