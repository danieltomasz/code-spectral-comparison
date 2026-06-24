// No-peak channel table. Compile standalone:  typst compile tbl_no_peak.typ
// then include the PDF in the Quarto doc and add the caption there.
#set page(width: auto, height: auto, margin: 6pt)
#set text(size: 9pt, font: "Arial")

#let data = csv("tbl_no_peak.csv")
#let rows = data.slice(1) // drop the CSV header row
#let body = rows.filter(r => r.at(0) != "All lobes")
#let total = rows.find(r => r.at(0) == "All lobes")

#let total_fill = rgb("#EFEFEF")

#let data_row(r, is_total) = {
  let f = if is_total { total_fill } else { white }
  let s = if is_total { strong } else { c => c }
  (
    table.cell(fill: f, align: left)[#s(r.at(0))],
    table.cell(fill: f)[#s(r.at(1))],
    table.cell(fill: f)[#s(r.at(2))],
    table.cell(fill: f)[#s(r.at(3))],
    table.cell(fill: f)[#s(r.at(4))],
  )
}

#table(
  columns: 5,
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 8pt, y: 4.5pt),

  table.hline(stroke: 0.9pt),
  table.header(
    table.cell(rowspan: 2, align: bottom)[*Lobe*],
    table.cell(colspan: 2, align: center)[*iEEG atlas*],
    table.cell(colspan: 2, align: center)[*source HD-EEG*],
    table.hline(start: 1, end: 3, stroke: 0.4pt),
    table.hline(start: 3, end: 5, stroke: 0.4pt),
    [Channels], [No peak], [Channels], [No peak],
  ),
  table.hline(stroke: 0.4pt),

  ..body.map(r => data_row(r, false)).flatten(),

  table.hline(stroke: 0.4pt),
  ..data_row(total, true),

  table.hline(stroke: 0.9pt),
)
