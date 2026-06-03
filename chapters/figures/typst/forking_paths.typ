#import "@preview/fletcher:0.5.9" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 16pt, fill: white)

#let c-blue = rgb("#1f78b4")
#let c-mag = rgb("#c2185b")
#let mono(body) = text(font: "DejaVu Sans Mono", size: 9pt, body)

#diagram(
  spacing: (24mm, 11mm),
  node-inset: 8pt,
  node-shape: rect,

  // --- nodes ---
  node((0, 2), mono[Raw\ Modality\ Inputs], stroke: 1.6pt + c-blue, extrude: (0, 4), name: <in>),
  node((1, 0), mono[Path 1: Uncorrected\ Spectra (Naive)], stroke: 1.4pt + c-blue, name: <p1>),
  node((1, 4), mono[Path 2: Spectral\ Parameterization], stroke: 1.4pt + c-mag, name: <p2>),
  node((2, 0.8), mono[Fixed (1/f)\ Model], stroke: 1.4pt + c-blue, name: <fixed>),
  node((2, 2.2), mono[Knee\ Model], stroke: 1.4pt + c-mag, name: <knee>),
  node((2, 4), mono[Peak\ Parameters], stroke: 1.4pt + c-mag, name: <peak>),

  // --- edges: left split (shared trunk at x=0.5) ---
  edge(<in>, (0.5, 2), (0.5, 0), <p1>, [Spectral\ Processing], "-|>", stroke: c-blue),
  edge(<in>, (0.5, 2), (0.5, 4), <p2>, [Spectral\ Processing], "-|>", stroke: c-mag),

  // --- edges: right split (shared trunk at x=1.55) ---
  edge(<p2>, (1.55, 4), (1.55, 0.8), <fixed>, [Aperiodic\ Modeling], "-|>", label-pos: 0.82, stroke: c-blue),
  edge(<p2>, (1.55, 4), (1.55, 2.2), <knee>, "-|>", stroke: c-mag),
  edge(<p2>, <peak>, [Oscillatory\ Extraction], "-|>", stroke: c-mag),
)
