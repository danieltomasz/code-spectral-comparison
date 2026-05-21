import os
import math

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

# Frequency range from 1 to 80 Hz, log-spaced
log_f_min, log_f_max = 0.0, 1.95 # log10(1) to slightly above log10(80) = 1.903
log_freqs = [0.0 + 1.903 * i / 99 for i in range(100)]

# Base parameters
b = 3.0
chi_ref = 1.5
k_ref = 10.0

# Panel A: Fixed vs Knee
fixed_a = []
knee_a = []
for lf in log_freqs:
    f_val = 10**lf
    fixed_a.append((lf, b - chi_ref * lf))
    knee_a.append((lf, b - math.log10(k_ref + f_val**chi_ref)))

# Panel B: Knee Parameter Variation (k)
k_low = 2.0
k_high = 50.0

knee_b_low = []
knee_b_ref = []
knee_b_high = []
for lf in log_freqs:
    f_val = 10**lf
    knee_b_low.append((lf, b - math.log10(k_low + f_val**chi_ref)))
    knee_b_ref.append((lf, b - math.log10(k_ref + f_val**chi_ref)))
    knee_b_high.append((lf, b - math.log10(k_high + f_val**chi_ref)))

# Panel C: Exponent Variation (chi)
chi_low = 1.0
chi_high = 2.0

knee_c_low = []
knee_c_ref = []
knee_c_high = []
for lf in log_freqs:
    f_val = 10**lf
    knee_c_low.append((lf, b - math.log10(k_ref + f_val**chi_low)))
    knee_c_ref.append((lf, b - math.log10(k_ref + f_val**chi_ref)))
    knee_c_high.append((lf, b - math.log10(k_ref + f_val**chi_high)))


# ==================== COORDINATE MAPPING ====================

def map_coords(x, y, x_min, x_max, y_min, y_max, panel_x_offset, panel_y_offset):
    px = panel_x_offset + ((x - x_min) / (x_max - x_min)) * panel_width
    py = panel_y_offset + panel_height - ((y - y_min) / (y_max - y_min)) * panel_height
    return px, py

# Ranges
y_min, y_max = 0.0, 3.0

# Define log-spaced ticks with original frequency values as labels
ticks_x = [
    (0.0, "1"),      # log10(1)
    (0.301, "2"),    # log10(2)
    (0.699, "5"),    # log10(5)
    (1.0, "10"),     # log10(10)
    (1.301, "20"),    # log10(20)
    (1.602, "40"),    # log10(40)
    (1.903, "80")     # log10(80)
]

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
      .subscript { font-size: 9px; font-style: normal; }
      .superscript { font-size: 9px; font-style: normal; }
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
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1e3a8a" />
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ea580c" />
    </marker>
    <marker id="arrow-grey" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    
    <!-- Shading gradients -->
    <linearGradient id="grad-teal" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0d9488" stop-opacity="0.14" />
      <stop offset="100%" stop-color="#0d9488" stop-opacity="0.005" />
    </linearGradient>
  </defs>''')

# Clip paths for the panels
svg.append(f'''  <defs>
    <clipPath id="clip-panel-a">
      <rect x="{x_a}" y="{panel_y}" width="{panel_width}" height="{panel_height}" />
    </clipPath>
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

# Generate A mapped points
fixed_pts_a = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_a, panel_y) for pt in fixed_a]
knee_pts_a = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_a, panel_y) for pt in knee_a]

# A Shading (under Knee)
a_shade_path = f"M {knee_pts_a[0][0]:.1f} {knee_pts_a[0][1]:.1f} "
for pt in knee_pts_a[1:]:
    a_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
a_bottom_right = map_coords(log_f_max, y_min, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
a_bottom_left = map_coords(log_f_min, y_min, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
a_shade_path += f"L {a_bottom_right[0]:.1f} {a_bottom_right[1]:.1f} L {a_bottom_left[0]:.1f} {a_bottom_left[1]:.1f} Z"

svg.append(f'  <path d="{a_shade_path}" fill="url(#grad-teal)" clip-path="url(#clip-panel-a)" />')

# Grids in A
for val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    _, gy = map_coords(0, val, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
    svg.append(f'  <line x1="{x_a}" y1="{gy:.1f}" x2="{x_a + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for tick_lf, _ in ticks_x:
    gx, _ = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# Axes in A
svg.append(f'  <line x1="{x_a}" y1="{panel_y}" x2="{x_a}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_a}" y1="{panel_y + panel_height}" x2="{x_a + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_lf, tick_lbl in ticks_x:
    tx, ty = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_lbl}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    tx, ty = map_coords(0, tick_y, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y:.1f}</text>')

# Curves in A (clipped)
fixed_path_a = f"M {fixed_pts_a[0][0]:.1f} {fixed_pts_a[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in fixed_pts_a[1:]])
knee_path_a = f"M {knee_pts_a[0][0]:.1f} {knee_pts_a[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in knee_pts_a[1:]])

svg.append(f'  <g clip-path="url(#clip-panel-a)">')
svg.append(f'    <path d="{fixed_path_a}" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="4,3" stroke-linecap="round" />')
svg.append(f'    <path d="{knee_path_a}" fill="none" stroke="#0d9488" stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round" />')
svg.append(f'  </g>')

# Knee Frequency Indicator
# Knee frequency lf = log10(k_ref**(1/chi_ref)) = log10(10**(1/1.5)) = log10(10**0.667) = 0.667 (approx 4.6 Hz)
lf_knee = 1.0 / 1.5
y_knee = b - math.log10(k_ref + (10**lf_knee)**chi_ref)
kx, ky = map_coords(lf_knee, y_knee, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
k_bottom_x, k_bottom_y = map_coords(lf_knee, y_min, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)

svg.append(f'  <line x1="{kx:.1f}" y1="{ky:.1f}" x2="{kx:.1f}" y2="{k_bottom_y:.1f}" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="2.5,2.5" />')
svg.append(f'  <circle cx="{kx:.1f}" cy="{ky:.1f}" r="4" fill="#ea580c" stroke="#ffffff" stroke-width="1" />')

# Text & Annotations for A
svg.append(f'  <text x="{x_a}" y="{panel_y - 25}" class="panel-letter">A</text>')
svg.append(f'  <text x="{x_a + 25}" y="{panel_y - 25}" class="panel-title">Fixed vs. Knee Aperiodic Component</text>')

# Axis Labels
svg.append(f'  <text x="{x_a + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'  <text x="{x_a - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_a - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Feature Labels
# Fixed mode label (slate)
svg.append(f'  <text x="{x_a + 250}" y="{panel_y + 195}" class="annotation-text" fill="#64748b" text-anchor="start">Fixed Mode</text>')
svg.append(f'  <text x="{x_a + 250}" y="{panel_y + 207}" class="annotation-subtext" fill="#64748b" text-anchor="start">(No Knee: Straight Line)</text>')

# Knee mode label (teal)
svg.append(f'  <text x="{x_a + 250}" y="{panel_y + 120}" class="annotation-text" fill="#0f766e" text-anchor="start">Knee Mode</text>')
svg.append(f'  <text x="{x_a + 250}" y="{panel_y + 132}" class="annotation-subtext" fill="#0f766e" text-anchor="start">(Bends at Knee Freq.)</text>')

# Knee Frequency text (orange)
svg.append(f'  <text x="{kx + 10}" y="{panel_y + 250}" class="annotation-text" fill="#c2410c" text-anchor="start">Knee Frequency</text>')
svg.append(f'  <text x="{kx + 10}" y="{panel_y + 262}" class="annotation-subtext" fill="#c2410c" text-anchor="start">f<tspan class="subscript">k</tspan> = k<tspan class="superscript">1/χ</tspan> (≈ 4.6 Hz)</text>')

# Plateau label (teal)
plat_x, plat_y = map_coords(0.12, 2.1, log_f_min, log_f_max, y_min, y_max, x_a, panel_y)
svg.append(f'  <text x="{plat_x:.1f}" y="{plat_y - 20:.1f}" class="annotation-text" fill="#0f766e" text-anchor="start">Low-Frequency Plateau</text>')
svg.append(f'  <text x="{plat_x:.1f}" y="{plat_y - 8:.1f}" class="annotation-subtext" fill="#0f766e" text-anchor="start">Flat region: L(f) ≈ b - log(k)</text>')
svg.append(f'  <path d="M {plat_x + 60:.1f} {plat_y - 3:.1f} Q {plat_x + 40:.1f} {plat_y + 8:.1f} {plat_x + 10:.1f} {plat_y + 15:.1f}" fill="none" stroke="#0d9488" stroke-width="1" marker-end="url(#arrow-teal)" />')

# Formula Display Box
form_box_x = x_a + 175
form_box_y = panel_y + 15
svg.append(f'  <rect x="{form_box_x}" y="{form_box_y}" width="175" height="58" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />')
svg.append(f'  <text x="{form_box_x + 10}" y="{form_box_y + 18}" class="formula-title">Aperiodic Component</text>')
svg.append(f'  <text x="{form_box_x + 10}" y="{form_box_y + 36}" class="formula-text" font-size="12.5">L(f) = b <tspan class="math-symbol">−</tspan> log(<tspan fill="#ea580c" font-weight="bold">k</tspan> + f<tspan class="superscript">χ</tspan>)</text>')


# ==========================================
# ==================== PANEL B ====================
# ==========================================
svg.append('  <!-- ==================== PANEL B ==================== -->')

# Generate B mapped points
low_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_b, panel_y) for pt in knee_b_low]
ref_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_b, panel_y) for pt in knee_b_ref]
high_pts_b = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_b, panel_y) for pt in knee_b_high]

# B Shading (under ref)
b_shade_path = f"M {ref_pts_b[0][0]:.1f} {ref_pts_b[0][1]:.1f} "
for pt in ref_pts_b[1:]:
    b_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
b_bottom_right = map_coords(log_f_max, y_min, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
b_bottom_left = map_coords(log_f_min, y_min, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
b_shade_path += f"L {b_bottom_right[0]:.1f} {b_bottom_right[1]:.1f} L {b_bottom_left[0]:.1f} {b_bottom_left[1]:.1f} Z"

svg.append(f'  <path d="{b_shade_path}" fill="url(#grad-teal)" clip-path="url(#clip-panel-b)" />')

# Grids in B
for val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    _, gy = map_coords(0, val, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
    svg.append(f'  <line x1="{x_b}" y1="{gy:.1f}" x2="{x_b + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for tick_lf, _ in ticks_x:
    gx, _ = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# Axes in B
svg.append(f'  <line x1="{x_b}" y1="{panel_y}" x2="{x_b}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_b}" y1="{panel_y + panel_height}" x2="{x_b + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_lf, tick_lbl in ticks_x:
    tx, ty = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_lbl}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    tx, ty = map_coords(0, tick_y, log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y:.1f}</text>')

# Curves in B (clipped)
low_path_b = f"M {low_pts_b[0][0]:.1f} {low_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in low_pts_b[1:]])
ref_path_b = f"M {ref_pts_b[0][0]:.1f} {ref_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in ref_pts_b[1:]])
high_path_b = f"M {high_pts_b[0][0]:.1f} {high_pts_b[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in high_pts_b[1:]])

svg.append(f'  <g clip-path="url(#clip-panel-b)">')
svg.append(f'    <path d="{low_path_b}" fill="none" stroke="#22c55e" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'    <path d="{ref_path_b}" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="3,3" stroke-linecap="round" />')
svg.append(f'    <path d="{high_path_b}" fill="none" stroke="#3b82f6" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'  </g>')

# B Titles
svg.append(f'  <text x="{x_b}" y="{panel_y - 25}" class="panel-letter">B</text>')
svg.append(f'  <text x="{x_b + 25}" y="{panel_y - 25}" class="panel-title">Knee Parameter Variation (k)</text>')

# Axis Labels
svg.append(f'  <text x="{x_b + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'  <text x="{x_b - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_b - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Annotations for B (safely positioned to avoid overlap)
# Low k label (Green)
svg.append(f'  <text x="{x_b + 12}" y="{panel_y + 40}" class="annotation-text" fill="#15803d" text-anchor="start">Low Knee (k = 2)</text>')
svg.append(f'  <text x="{x_b + 12}" y="{panel_y + 52}" class="annotation-subtext" fill="#15803d" text-anchor="start">↑ Plateau / Shift Knee Left</text>')

# High k label (Blue)
svg.append(f'  <text x="{x_b + 12}" y="{panel_y + 230}" class="annotation-text" fill="#1d4ed8" text-anchor="start">High Knee (k = 50)</text>')
svg.append(f'  <text x="{x_b + 12}" y="{panel_y + 242}" class="annotation-subtext" fill="#1d4ed8" text-anchor="start">↓ Plateau / Shift Knee Right</text>')

# Vertical Plateau shifts arrows
ar_b_low_x, ar_b_low_y_start = map_coords(0.1, b - math.log10(k_ref + (10**0.1)**chi_ref), log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
ar_b_low_x, ar_b_low_y_end = map_coords(0.1, b - math.log10(k_low + (10**0.1)**chi_ref), log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
svg.append(f'  <line x1="{ar_b_low_x:.1f}" y1="{ar_b_low_y_start:.1f}" x2="{ar_b_low_x:.1f}" y2="{(ar_b_low_y_end + 7):.1f}" stroke="#15803d" stroke-width="1.2" marker-end="url(#arrow-orange)" />')

ar_b_high_x, ar_b_high_y_start = map_coords(0.1, b - math.log10(k_ref + (10**0.1)**chi_ref), log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
ar_b_high_x, ar_b_high_y_end = map_coords(0.1, b - math.log10(k_high + (10**0.1)**chi_ref), log_f_min, log_f_max, y_min, y_max, x_b, panel_y)
svg.append(f'  <line x1="{ar_b_high_x:.1f}" y1="{ar_b_high_y_start:.1f}" x2="{ar_b_high_x:.1f}" y2="{(ar_b_high_y_end - 7):.1f}" stroke="#1d4ed8" stroke-width="1.2" marker-end="url(#arrow-orange)" />')

# Convergence Label at high frequencies
svg.append(f'  <text x="{x_b + 348}" y="{panel_y + 295}" class="annotation-text" fill="#64748b" text-anchor="end">Asymptotic Convergence</text>')
svg.append(f'  <text x="{x_b + 348}" y="{panel_y + 307}" class="annotation-subtext" fill="#64748b" text-anchor="end">Identical Slope (-χ)</text>')


# ==========================================
# ==================== PANEL C ====================
# ==========================================
svg.append('  <!-- ==================== PANEL C ==================== -->')

# Generate C mapped points
low_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_c, panel_y) for pt in knee_c_low]
ref_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_c, panel_y) for pt in knee_c_ref]
high_pts_c = [map_coords(pt[0], pt[1], log_f_min, log_f_max, y_min, y_max, x_c, panel_y) for pt in knee_c_high]

# C Shading (under ref)
c_shade_path = f"M {ref_pts_c[0][0]:.1f} {ref_pts_c[0][1]:.1f} "
for pt in ref_pts_c[1:]:
    c_shade_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
c_bottom_right = map_coords(log_f_max, y_min, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
c_bottom_left = map_coords(log_f_min, y_min, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
c_shade_path += f"L {c_bottom_right[0]:.1f} {c_bottom_right[1]:.1f} L {c_bottom_left[0]:.1f} {c_bottom_left[1]:.1f} Z"

svg.append(f'  <path d="{c_shade_path}" fill="url(#grad-teal)" clip-path="url(#clip-panel-c)" />')

# Grids in C
for val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    _, gy = map_coords(0, val, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
    svg.append(f'  <line x1="{x_c}" y1="{gy:.1f}" x2="{x_c + panel_width}" y2="{gy:.1f}" class="grid-line" />')

for tick_lf, _ in ticks_x:
    gx, _ = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
    svg.append(f'  <line x1="{gx:.1f}" y1="{panel_y}" x2="{gx:.1f}" y2="{panel_y + panel_height}" class="grid-line" />')

# Axes in C
svg.append(f'  <line x1="{x_c}" y1="{panel_y}" x2="{x_c}" y2="{panel_y + panel_height}" class="axis-line" />')
svg.append(f'  <line x1="{x_c}" y1="{panel_y + panel_height}" x2="{x_c + panel_width}" y2="{panel_y + panel_height}" class="axis-line" />')

# Ticks and tick labels (X)
for tick_lf, tick_lbl in ticks_x:
    tx, ty = map_coords(tick_lf, y_min, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx:.1f}" y2="{(ty + 5):.1f}" class="tick-line" />')
    svg.append(f'  <text x="{tx:.1f}" y="{(ty + 18):.1f}" class="axis-tick-label" text-anchor="middle">{tick_lbl}</text>')

# Ticks and tick labels (Y)
for tick_y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    tx, ty = map_coords(0, tick_y, log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
    svg.append(f'  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{(tx - 5):.1f}" y2="{ty:.1f}" class="tick-line" />')
    svg.append(f'  <text x="{(tx - 8):.1f}" y="{(ty + 4):.1f}" class="axis-tick-label" text-anchor="end">{tick_y:.1f}</text>')

# Curves in C (clipped)
low_path_c = f"M {low_pts_c[0][0]:.1f} {low_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in low_pts_c[1:]])
ref_path_c = f"M {ref_pts_c[0][0]:.1f} {ref_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in ref_pts_c[1:]])
high_path_c = f"M {high_pts_c[0][0]:.1f} {high_pts_c[0][1]:.1f} " + " ".join([f"L {pt[0]:.1f} {pt[1]:.1f}" for pt in high_pts_c[1:]])

svg.append(f'  <g clip-path="url(#clip-panel-c)">')
svg.append(f'    <path d="{low_path_c}" fill="none" stroke="#22d3ee" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'    <path d="{ref_path_c}" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="3,3" stroke-linecap="round" />')
svg.append(f'    <path d="{high_path_c}" fill="none" stroke="#0891b2" stroke-width="2.2" stroke-linecap="round" />')
svg.append(f'  </g>')

# C Titles
svg.append(f'  <text x="{x_c}" y="{panel_y - 25}" class="panel-letter">C</text>')
svg.append(f'  <text x="{x_c + 25}" y="{panel_y - 25}" class="panel-title">Exponent Variation (Slope, χ)</text>')

# Axis Labels
svg.append(f'  <text x="{x_c + panel_width/2}" y="{panel_y + panel_height + 42}" class="axis-label" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'  <text x="{x_c - 48}" y="{panel_y + panel_height/2}" class="axis-label" text-anchor="middle" transform="rotate(-90, {x_c - 48}, {panel_y + panel_height/2})">log(Power)</text>')

# Annotations for C (safely positioned to avoid overlap)
# Common Plateau label
svg.append(f'  <text x="{x_c + 12}" y="{panel_y + 80}" class="annotation-text" fill="#0891b2" text-anchor="start">Common Plateau</text>')
svg.append(f'  <text x="{x_c + 12}" y="{panel_y + 92}" class="annotation-subtext" fill="#0891b2" text-anchor="start">(Since k is constant)</text>')

# Slope variation labels at high frequencies
# Flatter label (cyan)
svg.append(f'  <text x="{x_c + 348}" y="{panel_y + 185}" class="annotation-text" fill="#0891b2" text-anchor="end">Flatter Slope (χ = 1.0)</text>')
svg.append(f'  <text x="{x_c + 348}" y="{panel_y + 197}" class="annotation-subtext" fill="#0891b2" text-anchor="end">↓ Exponent (χ)</text>')

# Steeper label (dark cyan)
svg.append(f'  <text x="{x_c + 348}" y="{panel_y + 270}" class="annotation-text" fill="#0369a1" text-anchor="end">Steeper Slope (χ = 2.0)</text>')
svg.append(f'  <text x="{x_c + 348}" y="{panel_y + 282}" class="annotation-subtext" fill="#0369a1" text-anchor="end">↑ Exponent (χ)</text>')

# Rotation Arrow at high-frequency to show fan-out
ar_c_ref_x, ar_c_ref_y = map_coords(1.6, b - math.log10(k_ref + (10**1.6)**chi_ref), log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
ar_c_low_x, ar_c_low_y = map_coords(1.6, b - math.log10(k_ref + (10**1.6)**chi_low), log_f_min, log_f_max, y_min, y_max, x_c, panel_y)
ar_c_high_x, ar_c_high_y = map_coords(1.6, b - math.log10(k_ref + (10**1.6)**chi_high), log_f_min, log_f_max, y_min, y_max, x_c, panel_y)

svg.append(f'  <path d="M {ar_c_ref_x:.1f} {ar_c_ref_y:.1f} Q {(ar_c_ref_x + 12):.1f} {((ar_c_ref_y + ar_c_low_y)/2):.1f} {ar_c_low_x:.1f} {ar_c_low_y:.1f}" fill="none" stroke="#475569" stroke-width="1.2" marker-end="url(#arrow-grey)" />')
svg.append(f'  <path d="M {ar_c_ref_x:.1f} {ar_c_ref_y:.1f} Q {(ar_c_ref_x - 12):.1f} {((ar_c_ref_y + ar_c_high_y)/2):.1f} {ar_c_high_x:.1f} {ar_c_high_y:.1f}" fill="none" stroke="#475569" stroke-width="1.2" marker-end="url(#arrow-grey)" />')


svg.append('</svg>')

# Save SVG to file
svg_content = "\n".join(svg)
output_path = "/Users/daniel/PhD/spectral-comparison/code/diagrams/specparam_knee.svg"
with open(output_path, "w") as f:
    f.write(svg_content)

print(f"SVG generated successfully at {output_path}")
