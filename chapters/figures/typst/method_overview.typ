#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: (left: 30pt, rest: 14pt), fill: white)
#set text(font: "Source Sans 3", size: 9pt)

// --- palette --------------------------------------------------------------
#let c-ieeg  = rgb("#2166ac")   // iEEG  (blue)
#let c-hdeeg = rgb("#d6604d")   // HD-EEG (warm)
#let c-share = rgb("#4f6275")   // shared processing (slate)
#let c-out   = rgb("#1aa17a")   // outputs / concordance (green)
#let c-panel = rgb("#8a8f99")   // question-panel frames (grey)

// --- node helpers ---------------------------------------------------------
#let card(pos, title, body, col, nm) = node(
  pos,
  align(left, stack(
    spacing: 3pt,
    text(weight: "bold", size: 8.5pt, fill: col.darken(8%), title),
    text(size: 7.5pt, fill: black.lighten(8%), body),
  )),
  stroke: 1pt + col,
  fill: col.lighten(90%),
  corner-radius: 4pt,
  inset: 7pt,
  name: nm,
)

#let chip(pos, body, col, nm) = node(
  pos,
  text(weight: "bold", size: 8.5pt, fill: white, body),
  stroke: none,
  fill: col,
  corner-radius: 4pt,
  inset: 7pt,
  name: nm,
)

#let plabel(pos, body, col) = node(
  pos,
  rotate(-90deg, reflow: true,
    text(size: 7.5pt, weight: "bold", tracking: 0.6pt, fill: col, upper(body))),
  stroke: none,
)

#diagram(
  spacing: (13mm, 9mm),
  node-inset: 7pt,

  // ===== INPUTS =====
  card((0, 0), "iEEG reference atlas",
    [MNI Open atlas\ 106 patients · 1772 ch\ 38 regions · eyes-closed], c-ieeg, <ieeg>),
  card((2, 0), "HD-EEG dataset",
    [256-ch GSN · 19 subjects\ eyes-open resting state], c-hdeeg, <hdeeg>),
  card((2, 1), "eLORETA source recon.",
    [→ MICCAI ROIs\ 1444 virtual sensors → 38 regions], c-hdeeg, <eloreta>),

  // ===== SHARED PROCESSING =====
  card((1, 2), "Common spectral processing",
    [60 s · Welch PSD (2 s, 50%, 0.5 Hz)\ unit-power normalisation], c-share, <welch>),

  // ===== Q1 : uncorrected spectra =====
  card((0, 3.4), "K-means + no-peak ref.",
    [data-driven clusters\ KS screen → Wilcoxon prevalence], c-share, <kmeans>),
  card((2, 3.4), "Canonical-band power",
    [relative power per band\ (δ θ α β γ)], c-share, <bands>),

  // ===== Q2 : parametrisation =====
  card((1, 4.8), "Aperiodic model selection",
    [specparam · knee vs fixed\ per-modality BIC + bootstrap], c-share, <aper>),
  card((0, 5.9), "Modelled peaks",
    [centre freq · amplitude\ per-band prevalence], c-share, <peaks>),
  card((2, 5.9), "Subtraction residual",
    [PSD − fitted 1/f\ oscillatory spectrum], c-share, <resid>),

  // ===== concordance =====
  chip((1, 7.1), [Cross-modal concordance\ #text(size: 7pt, weight: "regular")[rel. power · residual · modelled peaks]], c-out, <conc>),

  // ===== Q3 : stability =====
  card((0, 8.4), "Knee vs fixed",
    [exponent agreement\ (rank corr.)], c-out, <s-knee>),
  card((1, 8.4), "Fitting range",
    [1–80 vs 1–45 Hz], c-out, <s-range>),
  card((2, 8.4), "IRASA vs specparam",
    [decomposition\ over 2–40 Hz], c-out, <s-irasa>),

  // ===== panel frames (drawn behind via enclose) =====
  node(enclose: (<kmeans>, <bands>), stroke: (paint: c-panel, dash: "dashed", thickness: 0.7pt),
    fill: luma(250), corner-radius: 7pt, inset: 15pt, name: <q1>),
  node(enclose: (<aper>, <peaks>, <resid>), stroke: (paint: c-panel, dash: "dashed", thickness: 0.7pt),
    fill: luma(250), corner-radius: 7pt, inset: 15pt, name: <q2>),
  node(enclose: (<s-knee>, <s-range>, <s-irasa>), stroke: (paint: c-out, dash: "dashed", thickness: 0.7pt),
    fill: c-out.lighten(95%), corner-radius: 7pt, inset: 15pt, name: <q3>),

  plabel((-0.62, 3.4), "Q1 · uncorrected", c-share),
  plabel((-0.62, 5.35), "Q2 · parametrised", c-share),
  plabel((-0.62, 8.4), "Q3 · stability", c-out),

  // ===== edges =====
  edge(<ieeg>, (0, 2), <welch>, "-|>", stroke: c-ieeg),
  edge(<hdeeg>, <eloreta>, "-|>", stroke: c-hdeeg),
  edge(<eloreta>, (2, 2), <welch>, "-|>", stroke: c-hdeeg),

  edge(<welch>, <kmeans>, "-|>", stroke: c-share),
  edge(<welch>, <bands>, "-|>", stroke: c-share),

  edge(<q1.south>, <aper>, "-|>", stroke: c-share),
  edge(<aper>, <peaks>, "-|>", stroke: c-share),
  edge(<aper>, <resid>, "-|>", stroke: c-share),

  edge(<peaks>, <conc>, "-|>", stroke: c-out),
  edge(<resid>, <conc>, "-|>", stroke: c-out),
  edge(<bands>, (2.9, 3.4), (2.9, 7.1), <conc>, "-|>", stroke: c-out + 0.7pt),

  edge(<aper.east>, (3.2, 4.8), (3.2, 8.4), <q3.east>, "-|>",
    label: text(size: 6.5pt, fill: c-out, "aperiodic fits"), label-side: right,
    stroke: (paint: c-out, dash: "dotted", thickness: 0.8pt)),
)
