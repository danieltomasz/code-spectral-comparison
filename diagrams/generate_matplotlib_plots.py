import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Set up global aesthetic settings for matplotlib to ensure publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Inter']
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#334155'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1.0

# Create diagrams directory if it doesn't exist
DIAGRAMS_DIR = "/Users/daniel/PhD/spectral-comparison/code/diagrams"
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

# Define shared variables for absolute mathematical alignment
ticks_x = np.log10([1, 2, 5, 10, 20, 40, 80])
ticks_labels = ["1", "2", "5", "10", "20", "40", "80"]


# ==============================================================================
# PART 1: SPECTRAL PARAMETERIZATION MODULAR PLOTTING FUNCTIONS
# ==============================================================================

def plot_param_panel_a(ax):
    """
    Plots Panel A: Spectral Parameterization fit overlays
    """
    freqs = np.linspace(1, 40, 200)
    
    # Models
    b_a = 2.45
    chi_a = 1.25
    k_a = 2.5
    
    def aperiodic_func(f):
        return b_a - np.log10(k_a + f**chi_a)
        
    def peak_func(f):
        amp = 0.52
        center = 10.5
        w = 2.0
        return amp * np.exp(-((f - center)**2) / (2 * w**2))
        
    # Generate noisy simulated data
    np.random.seed(12345)
    noise = np.random.normal(0, 0.018, len(freqs)) + 0.005 * np.sin(freqs * 0.8)
    ap_curve = aperiodic_func(freqs)
    peak_curve = peak_func(freqs)
    raw_spectrum = ap_curve + peak_curve + noise
    full_fit = ap_curve + peak_curve
    
    # Plots
    ax.plot(freqs, raw_spectrum, color='#94a3b8', alpha=0.7, lw=1.2, label='Original Spectrum')
    ax.plot(freqs, ap_curve, color='#2563eb', linestyle='--', lw=2.2, label='Aperiodic Fit (L)')
    ax.plot(freqs, full_fit, color='#0d9488', lw=2.8, label='Full Model Fit')
    
    # Shading
    ax.fill_between(freqs, 0.2, ap_curve, color='#2563eb', alpha=0.06)
    ax.fill_between(freqs, ap_curve, full_fit, color='#0d9488', alpha=0.15)
    
    # Offset highlight (at f=1 Hz)
    off_val = aperiodic_func(1)
    ax.axvline(1, ymin=0, ymax=(off_val - 0.2)/(2.7 - 0.2), color='#ea580c', linestyle=':', lw=1.8)
    ax.scatter([1], [off_val], color='#ea580c', edgecolor='white', s=50, zorder=10)
    
    # Annotations
    ax.text(1.2, off_val + 0.08, 'Offset (b)', color='#c2410c', fontweight='bold', fontsize=10)
    
    # Arrow for peak
    peak_max_x = 10.5
    peak_max_y = aperiodic_func(peak_max_x) + peak_func(peak_max_x)
    ax.annotate('Periodic Activity\n(Gaussian Peaks, $G_n$)', 
                xy=(peak_max_x, peak_max_y + 0.03), 
                xytext=(15, 2.3),
                arrowprops=dict(arrowstyle="->", color='#0f766e', lw=1.2, connectionstyle="arc3,rad=-0.15"),
                color='#0f766e', fontweight='bold', fontsize=9.5)
                
    # Label aperiodic
    ax.text(28, 1.25, 'Aperiodic Activity\n($1/f$ Component, $L$)', color='#1d4ed8', fontweight='bold', 
            fontsize=9.5, ha='center')
            
    # Formula box
    formula_text = (
        r"$\bf{SpecParam\ Model}$" + "\n"
        r"$P(f) = L(f) + \sum_{n} G_n(f)$" + "\n"
        r"$L(f) = b - \log_{10}(k + f^{\chi})$"
    )
    ax.text(21, 1.75, formula_text, fontsize=9.5, color='#0f172a',
            bbox=dict(facecolor='#f8fafc', edgecolor='#e2e8f0', boxstyle='round,pad=0.8', lw=1.2))
            
    # Setup axis
    ax.set_xlim(0, 40)
    ax.set_ylim(0.2, 2.7)
    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title('A  Spectral Parameterization', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='#e2e8f0', facecolor='white', fontsize=9)
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_param_panel_b(ax):
    """
    Plots Panel B: Exponent variation sweeps
    """
    log_freqs = np.linspace(0.0, 1.903, 100) # log10(1) to log10(80)
    
    pivot_x = 0.75 # log10(5.62 Hz)
    pivot_y = 1.25
    
    chi_ref = 1.1
    chi_steep = 1.75
    chi_flat = 0.45
    
    y_ref = pivot_y - chi_ref * (log_freqs - pivot_x)
    y_steep = pivot_y - chi_steep * (log_freqs - pivot_x)
    y_flat = pivot_y - chi_flat * (log_freqs - pivot_x)
    
    # Plots
    ax.plot(log_freqs, y_ref, color='#94a3b8', linestyle='--', lw=1.5, label='Reference')
    ax.plot(log_freqs, y_steep, color='#1e3a8a', lw=2.2, label=r'Steeper ($\chi = 1.75$)')
    ax.plot(log_freqs, y_flat, color='#3b82f6', lw=2.2, label=r'Flatter ($\chi = 0.45$)')
    
    # Shading under reference
    ax.fill_between(log_freqs, 0.0, y_ref, color='#2563eb', alpha=0.04)
    
    # Pivot circle
    ax.scatter([pivot_x], [pivot_y], color='#0f172a', edgecolor='white', s=50, zorder=10)
    
    # Curved rotation arrows
    ax.annotate('', xy=(0.4, pivot_y - chi_steep * (0.4 - pivot_x)), 
                xytext=(0.4, pivot_y - chi_ref * (0.4 - pivot_x)),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, connectionstyle="arc3,rad=-0.2", mutation_scale=10))
                
    ax.annotate('', xy=(1.45, pivot_y - chi_flat * (1.45 - pivot_x)), 
                xytext=(1.45, pivot_y - chi_ref * (1.45 - pivot_x)),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, connectionstyle="arc3,rad=-0.2", mutation_scale=10))
                
    # Labels
    ax.text(0.1, 2.2, 'Exponent Increment\n' + r'(Steeper, $\uparrow$ Exponent $\chi$)', 
            color='#1e3a8a', fontweight='bold', fontsize=9.5, va='bottom')
    ax.text(1.85, 1.45, 'Exponent Decrement\n' + r'(Flatter, $\downarrow$ Exponent $\chi$)', 
            color='#3b82f6', fontweight='bold', fontsize=9.5, ha='right', va='center')
            
    # Formula box
    formula_b = (
        r"$\bf{Aperiodic\ Component\ (k=0)}$" + "\n"
        r"$L(f) = b - \chi \log_{10}(f)$"
    )
    ax.text(1.8, 2.25, formula_b, fontsize=9, color='#0f172a', ha='right',
            bbox=dict(facecolor='#f8fafc', edgecolor='#e2e8f0', boxstyle='round,pad=0.6', lw=1.0))
            
    # Setup axis
    ax.set_xlim(0.0, 1.903)
    ax.set_ylim(0.0, 2.5)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Frequency (Hz, log scale)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title(r'B  Exponent Variation (Slope, $\chi$)', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_param_panel_c(ax):
    """
    Plots Panel C: Offset variation shifts
    """
    log_freqs = np.linspace(0.0, 1.903, 100)
    pivot_x = 0.75
    chi_ref = 1.1
    
    b_ref = 1.25
    b_high = 1.70
    b_low = 0.80
    
    y_ref_c = b_ref - chi_ref * (log_freqs - pivot_x)
    y_high_c = b_high - chi_ref * (log_freqs - pivot_x)
    y_low_c = b_low - chi_ref * (log_freqs - pivot_x)
    
    # Plots
    ax.plot(log_freqs, y_ref_c, color='#94a3b8', linestyle='--', lw=1.5, label='Reference')
    ax.plot(log_freqs, y_high_c, color='#c2410c', lw=2.2, label=r'High ($b = 1.70$)')
    ax.plot(log_freqs, y_low_c, color='#ea580c', linestyle=':', lw=2.2, label=r'Low ($b = 0.80$)')
    
    # Shading under reference
    ax.fill_between(log_freqs, 0.0, y_ref_c, color='#2563eb', alpha=0.04)
    
    # Parallel offset vertical indicators and dots
    dots_x = 0.25 # log10(1.78 Hz)
    ax.axvline(dots_x, color='#ea580c', linestyle=':', lw=1.2, alpha=0.6)
    
    ax.scatter([dots_x], [b_ref - chi_ref * (dots_x - pivot_x)], color='#94a3b8', edgecolor='white', s=35, zorder=10)
    ax.scatter([dots_x], [b_high - chi_ref * (dots_x - pivot_x)], color='#c2410c', edgecolor='white', s=35, zorder=10)
    ax.scatter([dots_x], [b_low - chi_ref * (dots_x - pivot_x)], color='#ea580c', edgecolor='white', s=35, zorder=10)
    
    # Offset shift arrows
    ax.annotate('', xy=(0.55, b_high - chi_ref * (0.55 - pivot_x) - 0.08), 
                xytext=(0.55, b_ref - chi_ref * (0.55 - pivot_x) + 0.05),
                arrowprops=dict(arrowstyle="->", color='#c2410c', lw=1.5, mutation_scale=12))
                
    ax.annotate('', xy=(1.15, b_low - chi_ref * (1.15 - pivot_x) + 0.08), 
                xytext=(1.15, b_ref - chi_ref * (1.15 - pivot_x) - 0.05),
                arrowprops=dict(arrowstyle="->", color='#ea580c', lw=1.5, mutation_scale=12))
                
    # Labels
    ax.text(dots_x + 0.05, 2.3, 'Offset Increment\n' + r'$\uparrow$ Offset ($b$) / $\uparrow$ Broadband Power', 
            color='#c2410c', fontweight='bold', fontsize=9.5, va='center')
            
    ax.text(1.1, 0.25, 'Offset Decrement\n' + r'$\downarrow$ Offset ($b$) / $\downarrow$ Broadband Power', 
            color='#ea580c', fontweight='bold', fontsize=9.5, ha='right', va='center')
            
    # Formula box
    formula_c = (
        r"$\bf{Aperiodic\ Component\ (k=0)}$" + "\n"
        r"$L(f) = \bf{b} - \chi \log_{10}(f)$"
    )
    ax.text(1.8, 2.25, formula_c, fontsize=9, color='#0f172a', ha='right',
            bbox=dict(facecolor='#f8fafc', edgecolor='#e2e8f0', boxstyle='round,pad=0.6', lw=1.0))
            
    # Setup axis
    ax.set_xlim(0.0, 1.903)
    ax.set_ylim(0.0, 2.5)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Frequency (Hz, log scale)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title('C  Offset Variation (Shift, $b$)', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ==============================================================================
# PART 2: KNEE DECOMPOSITION MODULAR PLOTTING FUNCTIONS
# ==============================================================================

def plot_knee_panel_a(ax):
    """
    Plots Knee Panel A: Fixed vs Knee aperiodic profiles
    """
    b = 3.0
    chi_ref = 1.5
    k_ref = 10.0
    
    log_freqs = np.linspace(0.0, 1.95, 150)
    freqs = 10**log_freqs
    
    y_fixed = b - chi_ref * log_freqs
    y_knee = b - np.log10(k_ref + freqs**chi_ref)
    
    # Curves
    ax.plot(log_freqs, y_fixed, color='#94a3b8', linestyle='--', lw=1.8, label='Fixed Mode')
    ax.plot(log_freqs, y_knee, color='#0d9488', lw=2.8, label='Knee Mode')
    
    # Shading under knee
    ax.fill_between(log_freqs, 0.0, y_knee, color='#0d9488', alpha=0.06)
    
    # Knee frequency line: fk = k**(1/chi) = 10**(1/1.5) = 10**0.667 = 4.64 Hz
    lf_knee = 1.0 / 1.5
    y_knee_pt = b - np.log10(k_ref + (10**lf_knee)**chi_ref)
    ax.axvline(lf_knee, ymin=0, ymax=y_knee_pt/3.0, color='#ea580c', linestyle='--', lw=1.5)
    ax.scatter([lf_knee], [y_knee_pt], color='#ea580c', edgecolor='white', s=55, zorder=10)
    
    # Labels
    ax.text(lf_knee + 0.05, 0.7, 'Knee Frequency\n' + r'$f_k = k^{1/\chi}$ ($\approx 4.6$ Hz)', 
            color='#c2410c', fontweight='bold', fontsize=9.5)
            
    # Plateau arrow
    plat_x = 0.1
    plat_y = b - np.log10(k_ref + (10**plat_x)**chi_ref)
    ax.annotate('Low-Frequency Plateau\n' + r'Flat region: $L(f) \approx b - \log_{10}(k)$', 
                xy=(plat_x, plat_y), 
                xytext=(0.1, 2.5),
                arrowprops=dict(arrowstyle="->", color='#0f766e', lw=1.2, connectionstyle="arc3,rad=0.15"),
                color='#0f766e', fontweight='bold', fontsize=9.5)
                
    ax.text(1.2, 1.45, 'Fixed Mode\n(No Knee: Straight Line)', color='#64748b', fontweight='bold', fontsize=9.5)
    ax.text(1.25, 0.95, 'Knee Mode\n(Bends at Knee Freq.)', color='#0f766e', fontweight='bold', fontsize=9.5)
    
    # Formula box
    formula_a = (
        r"$\bf{Aperiodic\ Component}$" + "\n"
        r"$L(f) = b - \log_{10}(\bf{k} + f^{\chi})$"
    )
    ax.text(1.8, 2.7, formula_a, fontsize=9.5, color='#0f172a', ha='right',
            bbox=dict(facecolor='#f8fafc', edgecolor='#e2e8f0', boxstyle='round,pad=0.8', lw=1.2))
            
    # Setup axis
    ax.set_xlim(0.0, 1.903)
    ax.set_ylim(0.0, 3.0)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Frequency (Hz, log scale)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title('A  Fixed vs. Knee Aperiodic Component', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_knee_panel_b(ax):
    """
    Plots Knee Panel B: k sweep variations
    """
    b = 3.0
    chi_ref = 1.5
    k_ref = 10.0
    
    log_freqs = np.linspace(0.0, 1.95, 150)
    freqs = 10**log_freqs
    
    k_low = 2.0
    k_high = 50.0
    
    y_low_k = b - np.log10(k_low + freqs**chi_ref)
    y_ref_k = b - np.log10(k_ref + freqs**chi_ref)
    y_high_k = b - np.log10(k_high + freqs**chi_ref)
    
    # Curves
    ax.plot(log_freqs, y_low_k, color='#22c55e', lw=2.2, label='Low Knee ($k=2$)')
    ax.plot(log_freqs, y_ref_k, color='#64748b', linestyle='--', lw=1.5, label='Reference ($k=10$)')
    ax.plot(log_freqs, y_high_k, color='#3b82f6', lw=2.2, label='High Knee ($k=50$)')
    
    # Shading under reference
    ax.fill_between(log_freqs, 0.0, y_ref_k, color='#0d9488', alpha=0.04)
    
    # Knee frequency dots and lines
    for k_val, mk_color in [(k_low, '#15803d'), (k_ref, '#475569'), (k_high, '#1d4ed8')]:
        lf_k = np.log10(k_val) / chi_ref
        y_k_pt = b - np.log10(2 * k_val)
        ax.axvline(lf_k, ymin=0, ymax=y_k_pt/3.0, color=mk_color, linestyle=':', lw=1.0, alpha=0.7)
        ax.scatter([lf_k], [y_k_pt], color=mk_color, edgecolor='white', s=35, zorder=10)
        ax.text(lf_k + 0.03, y_k_pt + 0.05, r'$f_k$', color=mk_color, fontsize=8.5, fontweight='bold')
        
    # Vertical plateau arrows
    ax.annotate('', xy=(0.06, b - np.log10(k_low + (10**0.06)**chi_ref) - 0.08), 
                xytext=(0.06, b - np.log10(k_ref + (10**0.06)**chi_ref) + 0.05),
                arrowprops=dict(arrowstyle="->", color='#15803d', lw=1.2, mutation_scale=10))
                
    ax.annotate('', xy=(0.06, b - np.log10(k_high + (10**0.06)**chi_ref) + 0.08), 
                xytext=(0.06, b - np.log10(k_ref + (10**0.06)**chi_ref) - 0.05),
                arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2, mutation_scale=10))
                
    # Labels
    ax.text(0.12, 2.7, 'Low Knee (k = 2)\n' + r'$\uparrow$ Plateau / Shift Knee Left', 
            color='#15803d', fontweight='bold', fontsize=9.5, va='center')
            
    ax.text(0.12, 1.25, 'High Knee (k = 50)\n' + r'$\downarrow$ Plateau / Shift Knee Right', 
            color='#1d4ed8', fontweight='bold', fontsize=9.5, va='center')
            
    ax.text(1.85, 0.45, 'Asymptotic Convergence\n' + r'Identical Slope ($-\chi$)', 
            color='#64748b', fontweight='bold', fontsize=9.5, ha='right', va='center')
            
    # Setup axis
    ax.set_xlim(0.0, 1.903)
    ax.set_ylim(0.0, 3.0)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Frequency (Hz, log scale)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title('B  Knee Parameter Variation (k)', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_knee_panel_c(ax):
    """
    Plots Knee Panel C: chi sweep variations in knee model
    """
    b = 3.0
    chi_ref = 1.5
    k_ref = 10.0
    
    log_freqs = np.linspace(0.0, 1.95, 150)
    freqs = 10**log_freqs
    
    chi_low = 1.0
    chi_high = 2.0
    
    y_low_chi = b - np.log10(k_ref + freqs**chi_low)
    y_ref_chi = b - np.log10(k_ref + freqs**chi_ref)
    y_high_chi = b - np.log10(k_ref + freqs**chi_high)
    
    # Curves
    ax.plot(log_freqs, y_low_chi, color='#22d3ee', lw=2.2, label=r'Flatter ($\chi = 1.0$)')
    ax.plot(log_freqs, y_ref_chi, color='#64748b', linestyle='--', lw=1.5, label=r'Reference ($\chi = 1.5$)')
    ax.plot(log_freqs, y_high_chi, color='#0891b2', lw=2.2, label=r'Steeper ($\chi = 2.0$)')
    
    # Shading under reference
    ax.fill_between(log_freqs, 0.0, y_ref_chi, color='#0d9488', alpha=0.04)
    
    # Fan out rotation arrows
    rot_x = 1.25 # log10(17.8 Hz)
    y_ref_rot = b - np.log10(k_ref + (10**rot_x)**chi_ref)
    y_low_rot = b - np.log10(k_ref + (10**rot_x)**chi_low)
    y_high_rot = b - np.log10(k_ref + (10**rot_x)**chi_high)
    
    ax.annotate('', xy=(rot_x, y_low_rot - 0.05), xytext=(rot_x, y_ref_rot + 0.03),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, connectionstyle="arc3,rad=0.2", mutation_scale=10))
                
    ax.annotate('', xy=(rot_x, y_high_rot + 0.05), xytext=(rot_x, y_ref_rot - 0.03),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, connectionstyle="arc3,rad=0.2", mutation_scale=10))
                
    # Labels
    ax.text(0.12, 2.45, 'Common Plateau\n(Since k is constant)', color='#0891b2', fontweight='bold', fontsize=9.5, va='center')
    
    ax.text(1.85, 1.85, r'Flatter Slope ($\chi = 1.0$)' + '\n' + r'(Decreased Exponent, $\downarrow \chi$)', 
            color='#0891b2', fontweight='bold', fontsize=9.5, ha='right', va='center')
            
    ax.text(1.85, 0.65, r'Steeper Slope ($\chi = 2.0$)' + '\n' + r'(Increased Exponent, $\uparrow \chi$)', 
            color='#0369a1', fontweight='bold', fontsize=9.5, ha='right', va='center')
            
    # Setup axis
    ax.set_xlim(0.0, 1.903)
    ax.set_ylim(0.0, 3.0)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Frequency (Hz, log scale)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('log(Power)', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title(r'C  Exponent Variation (Slope, $\chi$)', fontweight='bold', fontsize=13, pad=15, loc='left')
    ax.grid(True, which='both', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ==============================================================================
# MAIN BATCH EXECUTION RUNNER
# ==============================================================================

def generate_combined_plots():
    """
    Generates the original, horizontal 3-panel combined images
    """
    # 1. Parameterization Combined
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=300)
    plot_param_panel_a(axes[0])
    plot_param_panel_b(axes[1])
    plot_param_panel_c(axes[2])
    plt.tight_layout()
    output_path = os.path.join(DIAGRAMS_DIR, "specparam_parameterization_matplotlib.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined parameterization figure: {output_path}")

    # 2. Knee Combined
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=300)
    plot_knee_panel_a(axes[0])
    plot_knee_panel_b(axes[1])
    plot_knee_panel_c(axes[2])
    plt.tight_layout()
    output_path = os.path.join(DIAGRAMS_DIR, "specparam_knee_matplotlib.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined knee figure: {output_path}")


def generate_individual_plots():
    """
    Generates single standalone panels (A, B, C) with perfectly square-ish ratios
    """
    # Define sizes optimized for standalone slide grid items
    figsize_standalone = (6.2, 5.2)

    # 1. Parameterization Individual Panels
    param_panels = [
        ("specparam_param_A.png", plot_param_panel_a),
        ("specparam_param_B.png", plot_param_panel_b),
        ("specparam_param_C.png", plot_param_panel_c)
    ]
    for filename, plot_func in param_panels:
        fig, ax = plt.subplots(figsize=figsize_standalone, dpi=300)
        plot_func(ax)
        plt.tight_layout()
        output_path = os.path.join(DIAGRAMS_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved standalone parameterization panel: {output_path}")

    # 2. Knee Individual Panels
    knee_panels = [
        ("specparam_knee_A.png", plot_knee_panel_a),
        ("specparam_knee_B.png", plot_knee_panel_b),
        ("specparam_knee_C.png", plot_knee_panel_c)
    ]
    for filename, plot_func in knee_panels:
        fig, ax = plt.subplots(figsize=figsize_standalone, dpi=300)
        plot_func(ax)
        plt.tight_layout()
        output_path = os.path.join(DIAGRAMS_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved standalone knee panel: {output_path}")


if __name__ == "__main__":
    print("Running physical EEG spectral simulation and generating Matplotlib figures...")
    generate_combined_plots()
    generate_individual_plots()
    print("All figures successfully created in the 'diagrams/' folder!")
