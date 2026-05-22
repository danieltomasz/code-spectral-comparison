import os
import math
import random

# Create directory if it doesn't exist
#os.makedirs("/Users/daniel/.gemini/antigravity/scratch/specparam_figure", exist_ok=True)

# Grid and Canvas Dimensions
width = 1350
height = 500
padding_left = 75
padding_right = 35
padding_top = 70
padding_bottom = 70

panel_width = 360
panel_height = 340
gap = 60

# Panel positions (left x coordinates)
x_a = padding_left
x_b = x_a + panel_width + gap
x_c = x_b + panel_width + gap

panel_y = padding_top

# ==================== DATA GENERATION ====================

# Panel A: Linear Frequency (1 to 40 Hz), log Power
freqs = [f for f in range(1, 41)]
b_a = 2.45
chi_a = 1.25
k_a = 2.5  # Knee parameter

def aperiodic_a(f):
    return b_a - math.log10(k_a + f**chi_a)

def peak_a(f):
    amp = 0.52
    center = 10.5
    w = 2.0
    return amp * math.exp(-((f - center)**2) / (2 * w**2))

# Seed for reproducibility
random.seed(12345)

original_spectrum = []
full_fit = []
aperiodic_fit = []

for f in freqs:
    ap = aperiodic_a(f)
    pk = peak_a(f)
    # Add high-frequency noise that decays slightly at higher frequencies
    noise = random.normalvariate(0, 0.018) + 0.005 * math.sin(f * 0.8)
    
    orig = ap + pk + noise
    fit = ap + pk
    
    original_spectrum.append((f, orig))
    full_fit.append((f, fit))
    aperiodic_fit.append((f, ap))

# Panel B: log(Frequency) vs log(Power)
# log10(f) spans ~1.8 Hz (0.25) to ~80 Hz (1.90), log-spaced
log_freqs = [0.25 + 1.65 * i / 49 for i in range(50)]
pivot_x = 0.75
pivot_y = 1.25

chi_ref = 1.1
chi_steep = 1.75
chi_flat = 0.45

ref_b = []
steep_b = []
flat_b = []

for lf in log_freqs:
    ref_b.append((lf, pivot_y - chi_ref * (lf - pivot_x)))
    steep_b.append((lf, pivot_y - chi_steep * (lf - pivot_x)))
    flat_b.append((lf, pivot_y - chi_flat * (lf - pivot_x)))

# Panel C: Offset shifts in log-log space
# Parallel lines
offset_ref = pivot_y
offset_high = pivot_y + 0.45
offset_low = pivot_y - 0.45

ref_c = []
high_c = []
low_c = []

for lf in log_freqs:
    ref_c.append((lf, offset_ref - chi_ref * (lf - pivot_x)))
    high_c.append((lf, offset_high - chi_ref * (lf - pivot_x)))
    low_c.append((lf, offset_low - chi_ref * (lf - pivot_x)))


# ==================== COORDINATE MAPPING ====================

def map_coords(x, y, x_min, x_max, y_min, y_max, panel_x_offset, panel_y_offset):
    px = panel_x_offset + ((x - x_min) / (x_max - x_min)) * panel_width
    py = panel_y_offset + panel_height - ((y - y_min) / (y_max - y_min)) * panel_height
    return px, py

# Ranges
y_min_a, y_max_a = 0.2, 2.7
y_min_bc, y_max_bc = 0.0, 2.5
log_f_min, log_f_max = 0.2, 1.95

# Log-spaced frequency ticks (Hz labels at log10 positions) for Panels B and C
ticks_x_bc = [(0.301, "2"), (1.0, "10"), (1.301, "20"), (1.602, "40"), (1.903, "80")]

# Start building the SVG
svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')

# Add defs (styles, gradients, markers)
svg.append('''  <defs>
    <!-- Modern academic typography -->
    <style>
      .panel-letter { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 26px; font-weight: 800; fill: #0f172a; }
      .panel-title { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #1e293b; }
      .axis-label { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }
      .axis-tick-label { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11px; fill: #475569; }
      .legend-text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 500; fill: #334155; }
      .annotation-text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 700; }
      .annotation-subtext { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10px; font-weight: 500; }
      .formula-title { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10px; font-weight: 700; fill: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
      .formula-text { font-family: "Georgia", "Times New Roman", serif; font-size: 13px; font-style: italic; fill: #0f172a; }
      .subscript { font-size: 9px; font-style: normal; baseline-shift: sub; }
      .superscript { font-size: 9px; font-style: normal; baseline-shift: super; }
      .math-symbol { font-family: "Georgia", serif; font-style: normal; }
      
      /* Grid and tick styles */
      .grid-line { stroke: #f1f5f9; stroke-width: 1; stroke-dasharray: 2,2; }
      .axis-line { stroke: #475569; stroke-width: 1.5; stroke-linecap: round; }
      .tick-line { stroke: #475569; stroke-width: 1.2; }
    </style>
    
    <!-- Marker definitions for elegant arrows -->
    <marker id="arrow-teal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0d9488" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1d4ed8" />
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ea580c" />
    </marker>
    <marker id="arrow-grey" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    
    <!-- Shading gradients -->
    <linearGradient id="grad-periodic" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0d9488" stop-opacity="0.22" />
      <stop offset="100%" stop-color="#0d9488" stop-opacity="0.01" />
    </linearGradient>
    <linearGradient id="grad-aperiodic" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2563eb" stop-opacity="0.14" />
      <stop offset="100%" stop-color="#2563eb" stop-opacity="0.005" />
    </linearGradient>
  </defs>''')

# Clip paths to prevent lines/shading from crossing outside the plot areas
svg.append(f'''  <defs>
    <clipPath id="clip-panel-b">
      <rect x="{x_b}" y="{panel_y}" width="{panel_width}" height="{panel_height}" />
    </clipPath>
    <clipPath id="clip-panel-c">
      <rect x="{x_c}" y="{panel_y}" width="{panel_width}" height="{panel_height}" />
    </clipPath>
  </defs>''')


# White background canvas
svg.append(f'  <rect width="{width}" height="{height}" fill="#ffffff" />')

# ==========================================
# ==================== PANEL A ====================
# ==========================================
svg.append('  <!-- ==================== PANEL A ==================== -->')

# Generate mapped points
orig_pts = [map_coords(pt[0], pt[1], 1, 40, y_min_a, y_max_a, x_a, panel_y) for pt in original_spectrum]
fit_pts = [map_coords(pt[0], pt[1], 1, 40, y_min_a, y_max_a, x_a, panel_y) for pt in full_fit]
ap_pts = [map_coords(pt[0], pt[1], 1, 40, y_min_a, y_max_a, x_a, panel_y) for pt in aperiodic_fit]

# Aperiodic Shading Area
ap_shade_path = f"M {ap_pts[0][0]:.1f} {ap_pts[0][1]:.1f} "
for pt in ap_pts[1:]:
    ap_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
ap_bottom_right = map_coords(40, y_min_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
ap_bottom_left = map_coords(1, y_min_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
ap_shade_path += f"L {ap_bottom_right[0]:.1f} {ap_bottom_right[1]:.1f} L {ap_bottom_left[0]:.1f} {ap_bottom_left[1]:.1f} Z"

# Periodic Shading Area (roughly between index 3 and 17, corresponding to f=4 to f=18)
pe_pts_fit = []
pe_pts_ap = []
for i, f in enumerate(freqs):
    if 4 <= f <= 19:
        pe_pts_fit.append(fit_pts[i])
        pe_pts_ap.append(ap_pts[i])

pe_shade_path = f"M {pe_pts_fit[0][0]:.1f} {pe_pts_fit[0][1]:.1f} "
for pt in pe_pts_fit[1:]:
    pe_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
for pt in reversed(pe_pts_ap):
    pe_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
pe_shade_path += "Z"

# 1. Shaded Regions
svg.append(f'  <path d="{ap_shade_path}" fill="url(#grad-aperiodic)" />')
svg.append(f'  <path d="{pe_shade_path}" fill="url(#grad-periodic)" />')

# 2. Grid lines
for val in [0.5, 1.0, 1.5, 2.0, 2.5]:
    _, gy = map_coords(1, val, 1, 40, y_min_a, y_max_a, x_a, panel_y)
    svg.append(f'  <line x1="{x_a}" y1="{gy:.1f}" x2="{x_a + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for val in [10, 20, 30, 40]:
    gx, _ = map_coords(val, y_min_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# 3. Axis Lines
svg.append(f'  <line x1="{x_a}" y1="{panel_y}" x2="{x_a}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_a}" y1="{panel_y + panel_height}" x2="{x_a + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_f in [1, 10, 20, 30, 40]:
    tx, ty = map_coords(tick_f, y_min_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_f}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0, 2.5]:
    tx, ty = map_coords(1, tick_y, 1, 40, y_min_a, y_max_a, x_a, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y}</text>')

# 4. Plots
# Original spectrum (slate-400, thin, noisy)
orig_path = f"M {orig_pts[0][0]:.1f} {orig_pts[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in orig_pts[1:]])
svg.append(f'  <path d="{orig_path}" fill="none" stroke="#94a3b8" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" />')

# Aperiodic fit (blue-600, dashed)
ap_path = f"M {ap_pts[0][0]:.1f} {ap_pts[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in ap_pts[1:]])
svg.append(f'  <path d="{ap_path}" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-dasharray="5,4" stroke-linecap="round" />')

# Full fit (teal-600, thick, smooth)
fit_path = f"M {fit_pts[0][0]:.1f} {fit_pts[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in fit_pts[1:]])
svg.append(f'  <path d="{fit_path}" fill="none" stroke="#0d9488" stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round" />')

# Pink dotted line at f=1 indicating broadband offset b
offset_val_a = aperiodic_a(1)
off_x, off_y = map_coords(1, offset_val_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
off_bottom_x, off_bottom_y = map_coords(1, y_min_a, 1, 40, y_min_a, y_max_a, x_a, panel_y)
svg.append(f'  <line x1="{off_x:.1f}" y1="{off_y:.1f}" x2="{off_x:.1f}" y2="{off_bottom_y:.1f}" stroke="#ea580c" stroke-width="1.8" stroke-dasharray="2,3" stroke-linecap="round" />')
svg.append(f'  <circle cx="{off_x:.1f}" cy="{off_y:.1f}" r="4" fill="#ea580c" stroke="#ffffff" stroke-width="1" />')

# 5. Text & Annotations
# Titles
svg.append(f'  <text x="{x_a}" y="{panel_y - 25}" class="panel-letter">A</text>')
svg.append(f'  <text x="{x_a + 25}" y="{panel_y - 25}" class="panel-title">Spectral Parameterization</text>')

# Axis Labels
svg.append(f'  <text x="{x_a + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz)</text>')
svg.append(f'  <text x="{x_a - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_a - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Feature Labels
# Periodic Activity
peak_x, peak_y = map_coords(10.5, aperiodic_a(10.5) + peak_a(10.5), 1, 40, y_min_a, y_max_a, x_a, panel_y)
peak_lbl_x, peak_lbl_y = map_coords(4.5, 2.45, 1, 40, y_min_a, y_max_a, x_a, panel_y)
svg.append(f'  <text x="{peak_lbl_x:.1f}" y="{peak_lbl_y - 6:.1f}" class="annotation-text" fill="#0f766e" text-anchor="start">Periodic Activity</text>')
svg.append(f'  <text x="{peak_lbl_x:.1f}" y="{peak_lbl_y + 6:.1f}" class="annotation-subtext" fill="#0f766e" text-anchor="start">(Gaussian Peaks, G<tspan class="subscript">n</tspan>)</text>')
svg.append(f'  <path d="M {(peak_lbl_x - 5):.1f} {peak_lbl_y:.1f} Q {(peak_x + 12):.1f} {(peak_y - 25):.1f} {peak_x:.1f} {(peak_y - 6):.1f}" fill="none" stroke="#0d9488" stroke-width="1.2" marker-end="url(#arrow-teal)" />')

# Aperiodic Activity
ap_lbl_x, ap_lbl_y = map_coords(28.0, 1.05, 1, 40, y_min_a, y_max_a, x_a, panel_y)
ap_curve_x, ap_curve_y = map_coords(28.0, aperiodic_a(28.0), 1, 40, y_min_a, y_max_a, x_a, panel_y)
svg.append(f'  <text x="{ap_lbl_x:.1f}" y="{ap_lbl_y - 6:.1f}" class="annotation-text" fill="#1d4ed8" text-anchor="middle">Aperiodic Activity</text>')
svg.append(f'  <text x="{ap_lbl_x:.1f}" y="{ap_lbl_y + 6:.1f}" class="annotation-subtext" fill="#1d4ed8" text-anchor="middle">(1/f Component, L)</text>')
svg.append(f'  <path d="M {ap_lbl_x:.1f} {(ap_lbl_y + 12):.1f} L {ap_curve_x:.1f} {(ap_curve_y - 6):.1f}" fill="none" stroke="#1d4ed8" stroke-width="1.2" marker-end="url(#arrow-blue)" />')

# Offset Label
off_lbl_x, off_lbl_y = map_coords(1, aperiodic_a(1), 1, 40, y_min_a, y_max_a, x_a, panel_y)
svg.append(f'  <text x="{off_lbl_x + 15}" y="{off_lbl_y + 15}" class="annotation-text" fill="#c2410c" text-anchor="start">Offset (b)</text>')
svg.append(f'  <path d="M {off_lbl_x + 12} {off_lbl_y + 11} L {off_lbl_x + 2} {off_lbl_y + 2}" fill="none" stroke="#ea580c" stroke-width="1.2" marker-end="url(#arrow-orange)" />')

# 6. Formula Box (Aesthetic card)
form_box_x = x_a + 175
form_box_y = panel_y + 15
svg.append(f'  <rect x="{form_box_x}" y="{form_box_y}" width="175" height="72" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />')
svg.append(f'  <text x="{form_box_x + 12}" y="{form_box_y + 18}" class="formula-title">SpecParam Model</text>')
svg.append(f'  <text x="{form_box_x + 12}" y="{form_box_y + 38}" class="formula-text">P(f) = L(f) + <tspan class="math-symbol">∑</tspan><tspan class="subscript">n</tspan> G<tspan class="subscript">n</tspan>(f)</text>')
svg.append(f'  <text x="{form_box_x + 12}" y="{form_box_y + 57}" class="formula-text" font-size="12.5">L(f) = b <tspan class="math-symbol">−</tspan> <tspan class="math-symbol">log</tspan><tspan class="subscript">10</tspan>(k + f<tspan class="superscript">χ</tspan>)</text>')

# 7. Legend Box
leg_x = x_a + 175
leg_y = panel_y + 97
svg.append(f'  <rect x="{leg_x}" y="{leg_y}" width="175" height="98" rx="6" fill="#ffffff" fill-opacity="0.9" stroke="#e2e8f0" stroke-width="1.2" />')

# Legend Items
# Original Spectrum
svg.append(f'  <line x1="{leg_x + 12}" y1="{leg_y + 16}" x2="{leg_x + 32}" y2="{leg_y + 16}" stroke="#94a3b8" stroke-width="1.5" />')
svg.append(f'  <text x="{leg_x + 40}" y="{leg_y + 19}" class="legend-text">Original Spectrum</text>')

# Full Model Fit
svg.append(f'  <line x1="{leg_x + 12}" y1="{leg_y + 36}" x2="{leg_x + 32}" y2="{leg_y + 36}" stroke="#0d9488" stroke-width="2.5" />')
svg.append(f'  <text x="{leg_x + 40}" y="{leg_y + 39}" class="legend-text">Full Model Fit</text>')

# Aperiodic Fit (L)
svg.append(f'  <line x1="{leg_x + 12}" y1="{leg_y + 56}" x2="{leg_x + 32}" y2="{leg_y + 56}" stroke="#2563eb" stroke-width="2" stroke-dasharray="4,2.5" />')
svg.append(f'  <text x="{leg_x + 40}" y="{leg_y + 59}" class="legend-text">Aperiodic Fit (L)</text>')

# Broadband Offset (b)
svg.append(f'  <line x1="{leg_x + 12}" y1="{leg_y + 76}" x2="{leg_x + 32}" y2="{leg_y + 76}" stroke="#ea580c" stroke-width="2" stroke-dasharray="1.5,2" />')
svg.append(f'  <text x="{leg_x + 40}" y="{leg_y + 79}" class="legend-text">Broadband Offset (b)</text>')


# ==========================================
# ==================== PANEL B ====================
# ==========================================
svg.append('  <!-- ==================== PANEL B ==================== -->')

# Generate B mapped points
ref_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y) for pt in ref_b]
steep_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y) for pt in steep_b]
flat_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y) for pt in flat_b]

# B Shading
b_shade_path = f"M {ref_pts_b[0][0]:.1f} {ref_pts_b[0][1]:.1f} "
for pt in ref_pts_b[1:]:
    b_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
b_bottom_right = map_coords(log_f_max - 0.05, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
b_bottom_left = map_coords(log_f_min + 0.05, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
b_shade_path += f"L {b_bottom_right[0]:.1f} {b_bottom_right[1]:.1f} L {b_bottom_left[0]:.1f} {b_bottom_left[1]:.1f} Z"

svg.append(f'  <path d="{b_shade_path}" fill="url(#grad-aperiodic)" clip-path="url(#clip-panel-b)" />')

# Grids in B
for val in [0.5, 1.0, 1.5, 2.0]:
    _, gy = map_coords(0, val, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
    svg.append(f'  <line x1="{x_b}" y1="{gy:.1f}" x2="{x_b + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for tick_lf, _ in ticks_x_bc:
    gx, _ = map_coords(tick_lf, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# Axes in B
svg.append(f'  <line x1="{x_b}" y1="{panel_y}" x2="{x_b}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_b}" y1="{panel_y + panel_height}" x2="{x_b + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_lf, tick_lbl in ticks_x_bc:
    tx, ty = map_coords(tick_lf, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_lbl}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0]:
    tx, ty = map_coords(log_f_min, tick_y, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y:.1f}</text>')

# Curves in B
ref_path_b = f"M {ref_pts_b[0][0]:.1f} {ref_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in ref_pts_b[1:]])
steep_path_b = f"M {steep_pts_b[0][0]:.1f} {steep_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in steep_pts_b[1:]])
flat_path_b = f"M {flat_pts_b[0][0]:.1f} {flat_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in flat_pts_b[1:]])

svg.append(f'  <g clip-path="url(#clip-panel-b)">')
svg.append(f'    <path d="{ref_path_b}" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3,3" />')
svg.append(f'    <path d="{steep_path_b}" fill="none" stroke="#1e3a8a" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'    <path d="{flat_path_b}" fill="none" stroke="#3b82f6" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'  </g>')

# Pivot point indicator
piv_x, piv_y_scr = map_coords(pivot_x, pivot_y, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
svg.append(f'  <circle cx="{piv_x:.1f}" cy="{piv_y_scr:.1f}" r="4.5" fill="#0f172a" stroke="#ffffff" stroke-width="1.5" />')

# Dynamic rotation arrows in B
# Left of pivot (low frequency, e.g. log_f = 0.4)
# Here, steeper curve is HIGHER than reference. The arrow points UP from reference to steep.
ref_val_left = pivot_y - chi_ref * (0.4 - pivot_x)
steep_val_left = pivot_y - chi_steep * (0.4 - pivot_x)
ar1_x, ar1_y_start = map_coords(0.4, ref_val_left, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
ar1_x, ar1_y_end = map_coords(0.4, steep_val_left, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
# Let's shift arrow slightly to avoid overlaps
svg.append(f'  <path d="M {ar1_x:.1f} {ar1_y_start:.1f} Q {(ar1_x - 12):.1f} {((ar1_y_start + ar1_y_end)/2):.1f} {ar1_x:.1f} {ar1_y_end:.1f}" fill="none" stroke="#475569" stroke-width="1.2" marker-end="url(#arrow-grey)" />')

# Right of pivot (high frequency, e.g. log_f = 1.35)
# Here, flatter curve is HIGHER than reference. The arrow points UP from reference to flat.
ref_val_right = pivot_y - chi_ref * (1.35 - pivot_x)
flat_val_right = pivot_y - chi_flat * (1.35 - pivot_x)
ar2_x, ar2_y_start = map_coords(1.35, ref_val_right, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
ar2_x, ar2_y_end = map_coords(1.35, flat_val_right, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
svg.append(f'  <path d="M {ar2_x:.1f} {ar2_y_start:.1f} Q {(ar2_x + 12):.1f} {((ar2_y_start + ar2_y_end)/2):.1f} {ar2_x:.1f} {ar2_y_end:.1f}" fill="none" stroke="#475569" stroke-width="1.2" marker-end="url(#arrow-grey)" />')

# B Titles
svg.append(f'  <text x="{x_b}" y="{panel_y - 25}" class="panel-letter">B</text>')
svg.append(f'  <text x="{x_b + 25}" y="{panel_y - 25}" class="panel-title">Exponent Variation (Slope, <tspan class="math-symbol">χ</tspan>)</text>')

# Axis Labels
svg.append(f'  <text x="{x_b + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'  <text x="{x_b - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_b - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Annotations for B
# Steeper Spectrum (Increment)
st_lbl_x, st_lbl_y = map_coords(0.3, 2.42, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
svg.append(f'  <text x="{st_lbl_x:.1f}" y="{st_lbl_y - 12:.1f}" class="annotation-text" fill="#1e3a8a" text-anchor="start">Exponent Increment (Steeper)</text>')
svg.append(f'  <text x="{st_lbl_x:.1f}" y="{st_lbl_y:.1f}" class="annotation-subtext" fill="#1e3a8a" text-anchor="start">↑ Exponent (<tspan class="math-symbol">χ</tspan>)</text>')

# Flatter Spectrum (Decrement)
fl_lbl_x, fl_lbl_y = map_coords(1.35, 1.45, log_f_min, log_f_max, y_min_bc, y_max_bc, x_b, panel_y)
svg.append(f'  <text x="{fl_lbl_x:.1f}" y="{fl_lbl_y - 12:.1f}" class="annotation-text" fill="#3b82f6" text-anchor="end">Exponent Decrement (Flatter)</text>')
svg.append(f'  <text x="{fl_lbl_x:.1f}" y="{fl_lbl_y:.1f}" class="annotation-subtext" fill="#3b82f6" text-anchor="end">↓ Exponent (<tspan class="math-symbol">χ</tspan>)</text>')

# Formula display in Panel B
formula_b_x = x_b + 180
formula_b_y = panel_y + 35
svg.append(f'  <rect x="{formula_b_x}" y="{formula_b_y}" width="165" height="42" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />')
svg.append(f'  <text x="{formula_b_x + 10}" y="{formula_b_y + 16}" class="formula-title">Aperiodic Component (k=0)</text>')
svg.append(f'  <text x="{formula_b_x + 10}" y="{formula_b_y + 33}" class="formula-text" font-size="12">L(f) = b <tspan class="math-symbol">−</tspan> <tspan font-weight="bold">χ</tspan> <tspan class="math-symbol">log</tspan><tspan class="subscript">10</tspan>(f)</text>')


# ==========================================
# ==================== PANEL C ====================
# ==========================================
svg.append('  <!-- ==================== PANEL C ==================== -->')

# Generate C mapped points
ref_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y) for pt in ref_c]
high_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y) for pt in high_c]
low_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y) for pt in low_c]

# C Shading (under ref)
c_shade_path = f"M {ref_pts_c[0][0]:.1f} {ref_pts_c[0][1]:.1f} "
for pt in ref_pts_c[1:]:
    c_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
c_bottom_right = map_coords(log_f_max - 0.05, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
c_bottom_left = map_coords(log_f_min + 0.05, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
c_shade_path += f"L {c_bottom_right[0]:.1f} {c_bottom_right[1]:.1f} L {c_bottom_left[0]:.1f} {c_bottom_left[1]:.1f} Z"

svg.append(f'  <path d="{c_shade_path}" fill="url(#grad-aperiodic)" clip-path="url(#clip-panel-c)" />')

# Grids in C
for val in [0.5, 1.0, 1.5, 2.0, 2.5]:
    _, gy = map_coords(0, val, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
    svg.append(f'  <line x1="{x_c}" y1="{gy:.1f}" x2="{x_c + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for tick_lf, _ in ticks_x_bc:
    gx, _ = map_coords(tick_lf, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# Axes in C
svg.append(f'  <line x1="{x_c}" y1="{panel_y}" x2="{x_c}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_c}" y1="{panel_y + panel_height}" x2="{x_c + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_lf, tick_lbl in ticks_x_bc:
    tx, ty = map_coords(tick_lf, y_min_bc, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_lbl}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0, 2.5]:
    tx, ty = map_coords(log_f_min, tick_y, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y:.1f}</text>')

# Curves in C
ref_path_c = f"M {ref_pts_c[0][0]:.1f} {ref_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in ref_pts_c[1:]])
high_path_c = f"M {high_pts_c[0][0]:.1f} {high_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in high_pts_c[1:]])
low_path_c = f"M {low_pts_c[0][0]:.1f} {low_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in low_pts_c[1:]])

svg.append(f'  <g clip-path="url(#clip-panel-c)">')
svg.append(f'    <path d="{ref_path_c}" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3,3" />')
svg.append(f'    <path d="{high_path_c}" fill="none" stroke="#c2410c" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'    <path d="{low_path_c}" fill="none" stroke="#ea580c" stroke-width="2.2" stroke-linecap="round" stroke-dasharray="6,4" />')
svg.append(f'  </g>')

# Vertical indicators at log_freq = 0.25 (for offsets)
c_off_ref_x, c_off_ref_y = map_coords(0.25, offset_ref - chi_ref * (0.25 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
c_off_high_x, c_off_high_y = map_coords(0.25, offset_high - chi_ref * (0.25 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
c_off_low_x, c_off_low_y = map_coords(0.25, offset_low - chi_ref * (0.25 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)

# Orange dotted vertical lines for offsets
svg.append(f'  <line x1="{c_off_ref_x:.1f}" y1="{c_off_ref_y:.1f}" x2="{c_off_ref_x:.1f}" y2="{panel_y + panel_height:.1f}" stroke="#ea580c" stroke-width="1.2" stroke-dasharray="2,2" />')
svg.append(f'  <line x1="{c_off_high_x:.1f}" y1="{c_off_high_y:.1f}" x2="{c_off_high_x:.1f}" y2="{panel_y + panel_height:.1f}" stroke="#ea580c" stroke-width="1.2" stroke-dasharray="2,2" />')
svg.append(f'  <line x1="{c_off_low_x:.1f}" y1="{c_off_low_y:.1f}" x2="{c_off_low_x:.1f}" y2="{panel_y + panel_height:.1f}" stroke="#ea580c" stroke-width="1.2" stroke-dasharray="2,2" />')

# Circle markers at offset lines
svg.append(f'  <circle cx="{c_off_ref_x:.1f}" cy="{c_off_ref_y:.1f}" r="3.5" fill="#94a3b8" stroke="#ffffff" stroke-width="1" />')
svg.append(f'  <circle cx="{c_off_high_x:.1f}" cy="{c_off_high_y:.1f}" r="3.5" fill="#c2410c" stroke="#ffffff" stroke-width="1" />')
svg.append(f'  <circle cx="{c_off_low_x:.1f}" cy="{c_off_low_y:.1f}" r="3.5" fill="#ea580c" stroke="#ffffff" stroke-width="1" />')

# Arrows indicating shift up and down
# Arrow up (at log_f = 0.5)
ar_up_x, ar_up_y_start = map_coords(0.5, offset_ref - chi_ref * (0.5 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
ar_up_x, ar_up_y_end = map_coords(0.5, offset_high - chi_ref * (0.5 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
svg.append(f'  <line x1="{ar_up_x:.1f}" y1="{ar_up_y_start:.1f}" x2="{ar_up_x:.1f}" y2="{(ar_up_y_end + 7):.1f}" stroke="#c2410c" stroke-width="1.5" marker-end="url(#arrow-orange)" />')

# Arrow down (at log_f = 1.1)
ar_dn_x, ar_dn_y_start = map_coords(1.1, offset_ref - chi_ref * (1.1 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
ar_dn_x, ar_dn_y_end = map_coords(1.1, offset_low - chi_ref * (1.1 - pivot_x), log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
svg.append(f'  <line x1="{ar_dn_x:.1f}" y1="{ar_dn_y_start:.1f}" x2="{ar_dn_x:.1f}" y2="{(ar_dn_y_end - 7):.1f}" stroke="#ea580c" stroke-width="1.5" marker-end="url(#arrow-orange)" />')

# C Titles
svg.append(f'  <text x="{x_c}" y="{panel_y - 25}" class="panel-letter">C</text>')
svg.append(f'  <text x="{x_c + 25}" y="{panel_y - 25}" class="panel-title">Offset Variation (Shift, <tspan class="math-symbol">b</tspan>)</text>')

# Axis Labels
svg.append(f'  <text x="{x_c + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'  <text x="{x_c - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_c - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Annotations for C
# Increment Label
c_inc_lbl_x, c_inc_lbl_y = map_coords(0.25, 2.35, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
svg.append(f'  <text x="{c_inc_lbl_x:.1f}" y="{c_inc_lbl_y - 5:.1f}" class="annotation-text" fill="#c2410c" text-anchor="start">Offset Increment</text>')
svg.append(f'  <text x="{c_inc_lbl_x:.1f}" y="{c_inc_lbl_y + 7:.1f}" class="annotation-subtext" fill="#c2410c" text-anchor="start">↑ Offset (b) / ↑ Broadband Power</text>')

# Decrement Label
c_dec_lbl_x, c_dec_lbl_y = map_coords(1.08, 0.20, log_f_min, log_f_max, y_min_bc, y_max_bc, x_c, panel_y)
svg.append(f'  <text x="{c_dec_lbl_x:.1f}" y="{c_dec_lbl_y - 5:.1f}" class="annotation-text" fill="#ea580c" text-anchor="end">Offset Decrement</text>')
svg.append(f'  <text x="{c_dec_lbl_x:.1f}" y="{c_dec_lbl_y + 7:.1f}" class="annotation-subtext" fill="#ea580c" text-anchor="end">↓ Offset (b) / ↓ Broadband Power</text>')

# Formula display in Panel C
formula_c_x = x_c + 180
formula_c_y = panel_y + 35
svg.append(f'  <rect x="{formula_c_x}" y="{formula_c_y}" width="165" height="42" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />')
svg.append(f'  <text x="{formula_c_x + 10}" y="{formula_c_y + 16}" class="formula-title">Aperiodic Component (k=0)</text>')
svg.append(f'  <text x="{formula_c_x + 10}" y="{formula_c_y + 33}" class="formula-text" font-size="12">L(f) = <tspan font-weight="bold">b</tspan> <tspan class="math-symbol">−</tspan> χ <tspan class="math-symbol">log</tspan><tspan class="subscript">10</tspan>(f)</text>')

svg.append('</svg>')

# Save SVG to file
svg_content = "\n".join(svg)
with open("/Users/daniel/PhD/spectral-comparison/code/diagrams/specparam_parameterization.svg", "w") as f:
    f.write(svg_content)

print("/Users/daniel/PhD/spectral-comparison/code/diagrams/specparam_parameterization.svg")
