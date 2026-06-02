#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page(width: auto, height: auto, margin: 16pt, fill: white)

#let c-blue = rgb("#1f78b4")
#let c-red = rgb("#d94436")
#let c-slate = rgb("#4f6275")
#let c-green = rgb("#1aa17a")
#let c-orange = rgb("#e08a18")

#let bx(body) = text(font: "DejaVu Sans Mono", size: 8pt, body)
#let bxL(body) = text(font: "DejaVu Sans Mono", size: 8pt, align(left, body))

#diagram(
  spacing: (30mm, 14mm),
  node-inset: 8pt,
  node-shape: rect,

  // --- nodes ---
  node((0, 0), bx[iEEG Reference Atlas\ (106 Patients,\ eyes-closed)], stroke: 1.4pt + c-blue, name: <ieeg>),
  node((2, 0), bx[HD-EEG Dataset\ (19 Subjects,\ eyes-open)], stroke: 1.4pt + c-red, name: <hdeeg>),
  node(
    (2, 1),
    bx[eLORETA Source\ Localization\ (Mapped to 38\ MICCAI regions)],
    stroke: 1.4pt + c-red,
    name: <eloreta>,
  ),
  node(
    (1, 2),
    bx[Welch PSD & Unit\ Normalization\ (Identical Welch\ settings)],
    stroke: 1.4pt + c-slate,
    name: <welch>,
  ),
  node(
    (1, 3),
    bxL[1. Raw Spectra Comparison\ - k-means clustering\ #h(6pt) of profiles\ - Baseline cross-modal\ #h(6pt) overlap],
    stroke: 1.4pt + c-slate,
    name: <s1>,
  ),
  node(
    (1, 4),
    bxL[
      2. Aperiodic Fitting &\ #h(6pt) Removal\ - Specparam fitting\ #h(6pt) (Knee / Fixed)\ - Subtract 1/f background
    ],
    stroke: 1.4pt + c-slate,
    name: <s2>,
  ),
  node(
    (0, 5),
    bxL[3. Residual Spectra\ #h(6pt) Comparison\ - Oscillatory overlap\ #h(6pt) after 1/f removal],
    stroke: 1.4pt + c-green,
    name: <s3>,
  ),
  node(
    (2, 5),
    bxL[
      4. Stability Assessments\ - Exponent vs. range &\ #h(6pt) method choices\ - Hierarchical Bayesian\ #h(6pt) regression
    ],
    stroke: 1.4pt + c-orange,
    name: <s4>,
  ),

  // --- edges ---
  edge(<ieeg>, (0, 2), <welch>, "-|>", stroke: c-blue), // iEEG bypasses eLORETA
  edge(<hdeeg>, <eloreta>, "-|>", stroke: c-red),
  edge(<eloreta>, (2, 2), <welch>, "-|>", stroke: c-red),
  edge(<welch>, <s1>, "-|>", stroke: c-slate),
  edge(<s1>, <s2>, "-|>", stroke: c-slate),
  edge(<s2>, (1, 4.55), (0, 4.55), <s3>, "-|>", stroke: c-green),
  edge(<s2>, (1, 4.55), (2, 4.55), <s4>, "-|>", stroke: c-orange),
)
