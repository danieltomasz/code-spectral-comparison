#!/usr/bin/env python3
"""Generate the chapter-2 analysis-pipeline schematic (figure1_pipeline.svg).

Structure mirrors the intro / results: a shared preprocessing + PSD stage, a
shared specparam-parameterisation step, three spectral *descriptors* (D1
non-parametric, D2 relative band power, D3 modelled peaks), and an auxiliary
stability footer. Edit the LAYOUT lists below and re-run; output is written to
chapters/figures/chapter2/diagrams/figure1_pipeline.svg.
"""
from pathlib import Path

W, H = 1180, 712
FONT = "Helvetica, Arial, sans-serif"

# palette (kept from the previous hand-drawn figure)
TEAL = dict(dark="#154f4c", mid="#2c8a85", light="#e4f2f1", title="#154f4c")
AMBER = dict(dark="#6e4513", mid="#c2802b", light="#fbf1e1", title="#6e4513")
PURPLE = dict(dark="#3c2f57", mid="#66508f", light="#eee9f4", title="#3c2f57")
BLUE = dict(dark="#234166", mid="#3f6fa3", light="#e9eff6", title="#234166")
RED = dict(dark="#7a2c20", mid="#c25a4a", light="#fceae6", title="#7a2c20")
GREY = dict(dark="#2b3036", mid="#707883", light="#f4f5f6", line="#d8dadf", title="#3a3f45")

SUB = "#565b62"  # subtitle text


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size, fill, *, weight="normal", anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>'
    )


def rrect(x, y, w, h, *, fill, stroke, sw=1.4, rx=9):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def box(x, y, w, h, title, sub, c, *, fill=None, title_fill=None):
    """A light item box: bold coloured title + grey subtitle."""
    out = [rrect(x, y, w, h, fill=fill or c["light"], stroke=c["mid"], sw=1.3)]
    cx = x + w / 2
    if sub:
        out.append(text(cx, y + h / 2 - 3, title, 12.5, title_fill or c["title"], weight="bold"))
        out.append(text(cx, y + h / 2 + 13, sub, 9.6, SUB))
    else:
        out.append(text(cx, y + h / 2 + 4.5, title, 12.5, title_fill or c["title"], weight="bold"))
    return "\n".join(out)


def header(x, y, w, h, title, question, c):
    """Solid coloured column header: white title + light question."""
    out = [rrect(x, y, w, h, fill=c["dark"], stroke=c["dark"], sw=1, rx=10)]
    cx = x + w / 2
    out.append(text(cx, y + 20, title, 15, "#ffffff", weight="bold"))
    out.append(text(cx, y + 38, question, 10.4, "#e9e6ef" if c is PURPLE else "#eef6f5"))
    return "\n".join(out)


def arrow(pts, *, color=GREY["mid"], sw=2.0):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<polyline points="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#arrow)"/>'
    )


els = []
defs = (
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY["mid"]}"/></marker></defs>'
)

# ---- sources -------------------------------------------------------------
els.append(box(250, 22, 300, 60,
               "iEEG reference atlas", "MNI Open iEEG · 106 patients · 1772 ch · 38 regions",
               BLUE))
els.append(box(630, 22, 300, 60,
               "HD-EEG dataset", "19 subjects · 256-channel · 5 min resting · eyes-open",
               RED))

# ---- shared preprocessing container -------------------------------------
els.append(rrect(150, 102, 880, 116, fill=GREY["light"], stroke=GREY["line"], sw=1.4, rx=12))
els.append(text(168, 120, "SHARED PREPROCESSING", 11, GREY["mid"], weight="bold", anchor="start"))
els.append(text(168, 178, "iEEG: no", 9.6, SUB, anchor="start"))
els.append(text(168, 190, "source step", 9.6, SUB, anchor="start"))
# eLORETA (HD only)
els.append(box(640, 122, 290, 40,
               "eLORETA source reconstruction", "→ 1444 sensors · 38 MICCAI regions",
               RED, fill=RED["light"]))
# Welch PSD (shared, centred)
els.append(box(370, 170, 430, 40,
               "Welch PSD & unit-power normalisation",
               "2 s Hamming · 50% overlap · 0.5 Hz · 200 Hz · 60 s",
               GREY, fill="#ffffff", title_fill=GREY["title"]))

# ---- shared specparam parameterisation ----------------------------------
els.append(box(300, 244, 330, 46,
               "specparam fit", "knee (iEEG) · fixed (HD) · BIC-selected",
               GREY, fill="#ffffff", title_fill=GREY["title"]))
els.append(box(650, 244, 230, 46,
               "subject-clustered bootstrap", "inference (Pearson / Spearman)",
               GREY, fill="#ffffff", title_fill=GREY["title"]))

# ---- three descriptor columns -------------------------------------------
COLS = [55, 425, 795]
CW = 330
HY, HH = 322, 50          # header
BY = 388                  # first item box
BH, BG = 46, 14           # item height / gap


def col_items(x, items, c):
    for i, (t, s) in enumerate(items):
        els.append(box(x, BY + i * (BH + BG), CW, BH, t, s, c))


# D1 non-parametric
els.append(header(COLS[0], HY, CW, HH, "D1 · Non-parametric",
                  "Atlas-style recovery, no 1/f model", TEAL))
col_items(COLS[0], [
    ("k-means clustering", "+ no-peak reference (Frauscher)"),
    ("Regional peak test", "22 intervals · KS + rank-sum"),
], TEAL)

# D2 relative band power
els.append(header(COLS[1], HY, CW, HH, "D2 · Relative band power",
                  "Uncorrected vs 1/f-removed residual", AMBER))
col_items(COLS[1], [
    ("Band-power maps", "region × band · canonical + Frauscher"),
    ("Cross-modal correlation", "per band across regions"),
    ("Band-overlap (Afnan)", "reliability & rhythm-presence masks"),
], AMBER)

# D3 modelled peaks
els.append(header(COLS[2], HY, CW, HH, "D3 · Modelled peaks",
                  "Read from specparam Gaussians", PURPLE))
col_items(COLS[2], [
    ("Peak prevalence", "fit-success rate per band"),
    ("Modelled power", "highest-peak Gaussian amplitude"),
    ("Peak centre frequency", "dominant frequency, unbinned"),
], PURPLE)

# ---- auxiliary footer ----------------------------------------------------
FY = BY + 3 * (BH + BG) + 16
els.append(rrect(55, FY, 1070, 60, fill="#f7f7f8", stroke=GREY["line"], sw=1.4, rx=11))
els.append(text(74, FY + 24, "Auxiliary · Stability of aperiodic estimates",
                12, GREY["title"], weight="bold", anchor="start"))
els.append(text(74, FY + 44,
                "Knee vs fixed (exponent)   ·   Fitting range (1–80 vs 1–45 Hz)"
                "   ·   IRASA vs specparam (2–40 Hz)", 10.4, SUB, anchor="start"))
els.append(text(1106, FY + 44, "agreement: Pearson / Spearman", 9.6, GREY["mid"],
                anchor="end", italic=True))

# ---- arrows --------------------------------------------------------------
# sources -> preprocessing
els.append(arrow([(400, 82), (400, 170)]))                       # iEEG -> Welch
els.append(arrow([(780, 82), (780, 122)]))                       # HD -> eLORETA
els.append(arrow([(785, 162), (785, 170)]))                      # eLORETA -> Welch
# Welch -> specparam fit + bootstrap (down)
els.append(arrow([(560, 210), (560, 244)]))
els.append(arrow([(700, 210), (760, 244)], sw=1.8))
# Welch -> D1 (non-parametric, bypasses specparam): elbow to D1 header
els.append(arrow([(370, 190), (220, 190), (220, HY)]))
# specparam -> D2, D3
els.append(arrow([(560, 290), (560, 310), (COLS[1] + CW / 2, 310), (COLS[1] + CW / 2, HY)]))
els.append(arrow([(765, 290), (765, 310), (COLS[2] + CW / 2, 310), (COLS[2] + CW / 2, HY)]))
# aperiodic estimates -> auxiliary footer
els.append(arrow([(880, 267), (1150, 267), (1150, FY)], sw=1.8, color=GREY["line"]))

svg = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="{FONT}" xmlns="http://www.w3.org/2000/svg">\n'
    f'{defs}\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n'
    + "\n".join(els)
    + "\n</svg>\n"
)

out = Path(__file__).resolve().parents[1] / "chapters/figures/chapter2/diagrams/figure1_pipeline.svg"
out.write_text(svg)
print("wrote", out, f"({len(svg)} bytes)")
