#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 16pt, fill: white)
#set text(font: "Source Sans 3", size: 9pt)

// ##########################################################################
//  CONFIG — edit these to restyle the whole figure
// ##########################################################################

// palette (modality = colour)
#let c-ieeg  = rgb("#2166ac")   // iEEG  (blue)
#let c-hdeeg = rgb("#d6604d")   // HD-EEG (warm red)
#let c-share = rgb("#3f5165")   // shared processing (slate)
#let c-out   = rgb("#1aa17a")   // outputs / concordance (green)
#let c-curve = rgb("#2b2b2b")   // spectrum line in glyphs
#let c-grid  = luma(180)        // glyph axes

// glyph drawing box, in cetz units (origin bottom-left, y points up)
#let gw = 50    // width
#let gh = 30    // height

// layout grid — node positions in diagram coords. Move a column/row here.
#let col-in    = 0       // inputs
#let col-psd   = 1.15    // shared PSD
#let col-treat = 2.5     // the three background treatments + stability
#let col-conc  = 3.95    // concordance (punchline)
#let row-top   = -0.15   // relative-band branch
#let row-mid   = 1       // PSD · modelled peaks · concordance (the spine)
#let row-bot   = 2.15    // residual branch
#let row-strip = 3.5     // stability strip
#let row-inhi  = 0       // iEEG input
#let row-inlo  = 2       // HD-EEG input

// ##########################################################################
//  GLYPH PRIMITIVES — small reusable draw pieces (cetz)
// ##########################################################################

// L-shaped axes
#let frame() = {
  import cetz.draw: line
  line((3, 3), (3, gh - 1), stroke: 0.5pt + c-grid)
  line((3, 3), (gw, 3), stroke: 0.5pt + c-grid)
}
// a spectrum / line through a list of points
#let curve(pts, col: c-curve, w: 1.4pt) = {
  import cetz.draw: line
  line(..pts, stroke: w + col)
}
// a vertical highlight band between two x-values
#let band(x1, x2, col) = {
  import cetz.draw: rect
  rect((x1, 3), (x2, gh - 1), fill: col, stroke: none)
}
// a dashed straight line (the fitted aperiodic background)
#let apline(p1, p2, col) = {
  import cetz.draw: line
  line(p1, p2, stroke: (paint: col, dash: "dashed", thickness: 1pt))
}

// ##########################################################################
//  CURVE SHAPES — tweak any glyph by editing its point list
// ##########################################################################

// 1/f decay with an alpha bump (the reference spectrum)
#let psd-pts = (
  (3, 27), (8, 24), (13, 21.5), (18, 21.2), (22, 23), (25, 21),
  (29, 16), (34, 12.5), (40, 9.5), (47, 7.5),
)
// second modality in the concordance glyph (hand-offset to overlap)
#let psd-hd-pts = (
  (3, 25), (8, 22.5), (13, 20), (18, 20.5), (22, 22), (25, 20),
  (29, 15.5), (34, 13), (40, 10.5), (47, 8.5),
)
// oscillatory residual: flat baseline + bumps
#let resid-pts = (
  (3, 6), (9, 6.5), (14, 9), (18, 15), (21, 20), (24, 14),
  (28, 7.5), (34, 6.2), (40, 8), (43, 6.5), (47, 6),
)
// aperiodic lines (log-log)
#let fixed-pts = ((3, 24), (47, 7))                       // straight
#let knee-pts  = ((3, 22), (16, 21), (24, 17), (47, 6))   // bent at the knee

// ##########################################################################
//  GLYPHS — assembled from primitives + shapes (draw order = back to front)
// ##########################################################################

#let g-psd   = cetz.canvas(length: 1pt, { frame(); curve(psd-pts) })

#let g-band  = cetz.canvas(length: 1pt, {
  band(16, 24, c-share.lighten(68%)); frame(); curve(psd-pts)
})
#let g-fit   = cetz.canvas(length: 1pt, {
  band(18, 25, c-out.lighten(85%)); frame()
  apline((3, 26), (47, 7), c-hdeeg); curve(psd-pts)
})
#let g-resid = cetz.canvas(length: 1pt, {
  frame(); curve(resid-pts, col: c-out.darken(5%))
})
#let g-aper  = cetz.canvas(length: 1pt, {
  frame()
  curve(fixed-pts, col: c-ieeg, w: 1.3pt)
  curve(knee-pts, col: c-hdeeg, w: 1.3pt)
})
#let g-conc  = cetz.canvas(length: 1pt, {
  frame()
  curve(psd-pts, col: c-ieeg)
  curve(psd-hd-pts, col: c-hdeeg)
})

// ##########################################################################
//  NODE HELPERS
// ##########################################################################

// glyph card: glyph on top, bold title, grey sub-line. fill/strk for emphasis.
#let gcard(pos, glyph, title, sub, col, nm, fill: white, strk: 0.9pt) = node(
  pos,
  stack(
    spacing: 4pt,
    glyph,
    text(weight: "bold", size: 8pt, fill: col.darken(10%), title),
    text(size: 6.8pt, fill: luma(70), sub),
  ),
  stroke: strk + col.lighten(15%),
  fill: fill,
  corner-radius: 5pt,
  inset: 7pt,
  name: nm,
)
// solid colour input chip
#let inchip(pos, title, sub, col, nm) = node(
  pos,
  align(left, stack(
    spacing: 2pt,
    text(weight: "bold", size: 9pt, fill: white, title),
    text(size: 6.8pt, fill: white.darken(4%), sub),
  )),
  stroke: none, fill: col, corner-radius: 5pt, inset: 8pt, name: nm,
)
// italic verb label that sits on a connector
#let vlab(b) = text(size: 7pt, style: "italic", fill: luma(55), b)

// ##########################################################################
//  DIAGRAM
//
//  layout map (cols →, rows ↓):
//             col-in        col-psd     col-treat          col-conc
//   top                                 Relative band
//   mid     iEEG / HD-EEG   PSD ─────►  Modelled peaks ──► Concordance
//   bot                                 Residual
//   strip                               Aperiodic stab.
// ##########################################################################

#diagram(
  spacing: (15mm, 7mm),
  node-inset: 7pt,

  // --- inputs ---
  inchip((col-in, row-inhi), "iEEG atlas", [106 patients · 38 regions\ eyes-closed], c-ieeg, <ieeg>),
  inchip((col-in, row-inlo), "HD-EEG", [19 subjects · eyes-open\ eLORETA → 38 regions], c-hdeeg, <hdeeg>),

  // --- shared spectrum ---
  gcard((col-psd, row-mid), g-psd, "Unit-power PSD", "Welch, identical settings", c-share, <psd>),

  // --- three background treatments ---
  gcard((col-treat, row-top), g-band, "Relative band power", "background ignored", c-share, <band>),
  gcard((col-treat, row-mid), g-fit, "Modelled peaks", "specparam · knee/fixed", c-out, <fit>),
  gcard((col-treat, row-bot), g-resid, "Subtraction residual", "background removed", c-out, <resid>),

  // --- comparison (the punchline) ---
  gcard((col-conc, row-mid), g-conc, "Cross-modal concordance", "per-region overlap · rank corr.", c-out, <conc>,
    fill: c-out.lighten(95%), strk: 1.5pt),

  // --- stability strip ---
  gcard((col-treat, row-strip), g-aper, "Aperiodic stability", "knee vs fixed · 1-80/1-45 Hz · IRASA", c-ieeg, <aper>),

  // --- edges: inputs funnel into PSD ---
  edge(<ieeg>, <psd>, "-|>", stroke: c-ieeg),
  edge(<hdeeg>, <psd>, "-|>", stroke: c-hdeeg),

  // --- edges: PSD branches to the three treatments (verbs on connectors) ---
  edge(<psd>, <band>, "-|>", label: vlab[k-means / bands], stroke: c-share),
  edge(<psd>, <fit>, "-|>", label: vlab[parameterise], stroke: c-share),
  edge(<psd>, <resid>, "-|>", label: vlab[subtract 1/f], stroke: c-share),

  // --- edges: three treatments funnel into concordance ---
  edge(<band>, <conc>, "-|>", stroke: c-out.lighten(10%)),
  edge(<fit>, <conc>, "-|>", stroke: c-out.lighten(10%)),
  edge(<resid>, <conc>, "-|>", stroke: c-out.lighten(10%)),

  // --- edge: stability is a stress-test of the aperiodic fits ---
  edge(<psd>, <aper>, "-|>", label: vlab[vary fit choices], label-side: right,
    stroke: (paint: c-ieeg, dash: "dotted")),
)
