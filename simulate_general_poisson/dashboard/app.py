import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from shiny import App, ui, render, reactive
import shinyswatch

# Define sampling rate and default simulation values
FS = 250.0  # Hz

def sample_damping_rates(n, w1, mu1, sigma1, mu2, sigma2):
    """
    Sample n damping rates gamma from the bimodal truncated Gaussian mixture distribution.
    """
    # Choose mode for each sample
    modes = np.random.choice([1, 2], size=n, p=[w1, 1.0 - w1])
    samples = np.zeros(n)
    
    for i in range(n):
        if modes[i] == 1:
            val = -1.0
            while val < 0.05:  # Truncate at 0.05 to avoid zero/negative damping
                val = np.random.normal(mu1, sigma1)
        else:
            val = -1.0
            while val < 0.05:
                val = np.random.normal(mu2, sigma2)
        samples[i] = val
    return samples

def simulate_oscillatory_poisson(T, fs, f0, lambda_rate, w1, mu1, sigma1, mu2, sigma2):
    """
    Simulate a filtered Poisson process of randomly timed, randomly damped oscillatory pulses.
    Also returns the Rice decomposition components Xc(t) and Xs(t), and the envelope A(t).
    """
    t = np.arange(0, T, 1.0 / fs)
    X = np.zeros_like(t)
    Xc = np.zeros_like(t)
    Xs = np.zeros_like(t)
    
    # Generate Poisson arrival times in [0, T]
    t_k = []
    curr_t = 0.0
    np.random.seed(42)  # Replicable Poisson arrivals for smooth slider experience
    while True:
        dt = np.random.exponential(1.0 / lambda_rate)
        curr_t += dt
        if curr_t > T:
            break
        t_k.append(curr_t)
        
    t_k = np.array(t_k)
    num_pulses = len(t_k)
    if num_pulses == 0:
        return t, X, Xc, Xs, np.zeros_like(t)
        
    # Draw damping rates and amplitudes
    gammas = sample_damping_rates(num_pulses, w1, mu1, sigma1, mu2, sigma2)
    # Pulses have random zero-mean amplitudes
    amplitudes = np.random.normal(0, 1.5, num_pulses)
    
    omega0 = 2.0 * np.pi * f0
    
    # Sum the pulses
    for tk, gamma, amp in zip(t_k, gammas, amplitudes):
        start_idx = int(np.ceil(tk * fs))
        if start_idx >= len(t):
            continue
        
        # Truncate pulse after 5 / gamma seconds to speed up simulation significantly
        duration = 5.0 / max(gamma, 0.05)
        end_idx = min(len(t), int(np.ceil((tk + duration) * fs)))
        
        pulse_t = t[start_idx:end_idx] - tk
        decay = amp * np.exp(-gamma * pulse_t)
        
        # Generative process pulse: amp * e^(-gamma*t) * cos(omega0*t)
        X[start_idx:end_idx] += decay * np.cos(omega0 * pulse_t)
        
        # Rice decomposition components:
        # Xc(t) = sum_k amp * cos(omega0 * tk) * e^(-gamma * (t - tk))
        # Xs(t) = sum_k amp * sin(omega0 * tk) * e^(-gamma * (t - tk))
        Xc[start_idx:end_idx] += decay * np.cos(omega0 * tk)
        Xs[start_idx:end_idx] += decay * np.sin(omega0 * tk)
        
    # Calculate Rice envelope
    envelope = np.sqrt(Xc**2 + Xs**2)
    return t, X, Xc, Xs, envelope

def compute_analytical_psd(freqs, f0, lambda_rate, w1, mu1, sigma1, mu2, sigma2):
    """
    Compute the exact analytical PSD S(f) using the discretized Fredholm Lorentzian mixture
    integrating over the bimodal damping distribution.
    """
    # Create a dense grid of damping rates gamma for numerical integration
    gamma_grid = np.linspace(0.01, 80.0, 300)
    d_gamma = gamma_grid[1] - gamma_grid[0]
    
    # Truncated normal distribution for Mode 1
    p1 = np.exp(-0.5 * ((gamma_grid - mu1) / sigma1) ** 2)
    p1_sum = np.sum(p1) * d_gamma
    if p1_sum > 0:
        p1 /= p1_sum
    else:
        p1 = np.zeros_like(gamma_grid)
        
    # Truncated normal distribution for Mode 2
    p2 = np.exp(-0.5 * ((gamma_grid - mu2) / sigma2) ** 2)
    p2_sum = np.sum(p2) * d_gamma
    if p2_sum > 0:
        p2 /= p2_sum
    else:
        p2 = np.zeros_like(gamma_grid)
        
    # Mixed PDF
    p_gamma = w1 * p1 + (1.0 - w1) * p2
    
    # Compute PSD: S(f) = (lambda * <A^2>) / (8 * pi^2) * \int (p(gamma) / ((f-f0)^2 + (gamma/2pi)^2)) d_gamma
    mean_a2 = 1.5**2  # Variance of standard amplitude distribution
    const = (lambda_rate * mean_a2) / (8.0 * np.pi**2)
    
    psd = np.zeros_like(freqs)
    # Vectorized computation for speed
    for idx, f in enumerate(freqs):
        # Lorentzian kernel
        lorentzian = 1.0 / ((f - f0)**2 + (gamma_grid / (2.0 * np.pi))**2)
        psd[idx] = const * np.sum(p_gamma * lorentzian * d_gamma)
        
    return psd, gamma_grid, p_gamma

def fit_aperiodic_slope(freqs, psd, fit_range):
    """
    Fit a 1/f^beta line to the PSD in log-log space over fit_range.
    Returns the fitted offset A and exponent beta.
    """
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    f_fit = freqs[mask]
    psd_fit = psd[mask]
    
    if len(f_fit) < 2:
        return 0.0, 0.0
        
    log_f = np.log10(f_fit)
    log_psd = np.log10(np.maximum(psd_fit, 1e-15))
    
    # Fit line: log10(PSD) = offset - beta * log10(f)
    B, A = np.polyfit(log_f, log_psd, 1)
    beta = -B
    offset = A
    return offset, beta


# ==========================================
# SHINY UI CODE
# ==========================================
app_ui = ui.page_fluid(
    ui.head_content(
        # Load MathJax CDN for LaTeX mathematical rendering
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
        # Load custom styling
        ui.include_css(Path(__file__).parent / "styles.css")
    ),
    
    # Header Area
    ui.div(
        ui.h2("Damped Oscillator Poisson Mixture Dashboard", class_="mt-3 mb-1 font-weight-bold"),
        ui.p("Simulating alpha oscillations and 1/f shoulders from a single physical generating process.", class_="text-muted mb-4", style="font-size: 1.05rem;"),
        class_="container-fluid p-0 pt-2"
    ),
    
    # Mobile Tabs Navigation (only visible on mobile)
    ui.HTML("""
        <ul class="nav nav-tabs mobile-tabs d-md-none" id="mobileTab" role="tablist">
            <li class="nav-item" role="presentation" style="flex: 1;">
                <button class="nav-link active w-100" id="params-tab" data-bs-toggle="tab" data-bs-target="#params-panel" type="button" role="tab" aria-controls="params-panel" aria-selected="true">
                    Configuration
                </button>
            </li>
            <li class="nav-item" role="presentation" style="flex: 1;">
                <button class="nav-link w-100" id="results-tab" data-bs-toggle="tab" data-bs-target="#results-panel" type="button" role="tab" aria-controls="results-panel" aria-selected="false">
                    Dashboard Panels
                </button>
            </li>
        </ul>
    """),
    
    # Main Content Container
    ui.row(
        # Parameter Sidebar (Left)
        ui.div(
            # Mixture mode 1: Weak Damping (Alpha Oscillation)
            ui.div(
                ui.h5("Mode 1: Weak Damping (Alpha Peak)", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #0f766e;"),
                ui.input_slider("mu1", "Mean Damping (μ₁)", min=0.1, max=10.0, value=1.5, step=0.1),
                ui.input_slider("sigma1", "Width (σ₁)", min=0.1, max=5.0, value=0.5, step=0.1),
                ui.input_slider("w1", "Relative Mixture Weight (w₁)", min=0.0, max=1.0, value=0.25, step=0.05),
                ui.p("Generates the sharp alpha oscillatory component around f₀.", class_="text-muted small mb-0"),
                class_="p-3 rounded mb-4",
                style="background-color: #f0fdfa; border: 1px solid #cbd5e1; border-left: 4px solid #0f766e;"
            ),
            
            # Mixture mode 2: Heavy Damping (1/f Background)
            ui.div(
                ui.h5("Mode 2: Heavy Damping (1/f Shoulders)", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #1e293b;"),
                ui.input_slider("mu2", "Mean Damping (μ₂)", min=10.0, max=80.0, value=30.0, step=1.0),
                ui.input_slider("sigma2", "Width (σ₂)", min=2.0, max=30.0, value=10.0, step=1.0),
                ui.output_ui("weight_mode2_text"),
                ui.p("Generates the broad 1/f-like spectral shoulders around f₀.", class_="text-muted small mb-0"),
                class_="p-3 rounded mb-4",
                style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #475569;"
            ),
            
            # Global process parameters
            ui.div(
                ui.h5("Process & Simulation Settings", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #0f766e;"),
                ui.input_slider("f0", "Alpha Center Frequency (f₀)", min=5.0, max=20.0, value=10.0, step=0.5),
                ui.input_slider("lambda_rate", "Poisson Arrival Rate (λ)", min=5, max=100, value=25, step=5),
                ui.input_slider("noise_std", "Additive White Noise (SD)", min=0.0, max=0.5, value=0.08, step=0.01),
                ui.input_slider("sim_duration", "Simulation Duration (T)", min=1.0, max=10.0, value=5.0, step=0.5),
                ui.input_slider("fit_min_f", "Fit Exponent Range Min (Hz)", min=12, max=25, value=15, step=1),
                ui.input_slider("fit_max_f", "Fit Exponent Range Max (Hz)", min=26, max=60, value=40, step=1),
                class_="p-3 rounded",
                style="background-color: #f5f3ff; border: 1px solid #cbd5e1; border-left: 4px solid #7c3aed;"
            ),
            id="params-panel",
            class_="tab-pane fade show active mobile-tab-pane col-md-4 col-lg-3 pe-md-4",
            role="tabpanel",
            aria_labelledby="params-tab"
        ),
        
        # Results Section (Right / Tabs)
        ui.div(
            # Navigation Tabs for the dashboard panels
            ui.navset_pill(
                # Tab 1: Theory and Damping Distribution
                ui.nav_panel(
                    "Theory & Damping (p(γ))",
                    ui.div(
                        ui.div("Conceptual Context: Single Generative Process", class_="sphinx-admonition-title"),
                        ui.div(
                            ui.p(
                                "Classical EEG analysis assumes that oscillatory activity (e.g. alpha rhythm) and scale-free background noise "
                                "(\\(1/f^\\beta\\)) arise from two distinct, independent mechanisms. "
                                "However, this dashboard demonstrates that a "
                                "<strong>single generative process</strong> of stochastically timed, randomly damped linear oscillatory pulses: "
                            ),
                            ui.div(
                                "$$X(t) = \\sum_k A_k\\, e^{-\\gamma_k(t-t_k)}\\cos\\bigl(2\\pi f_0(t-t_k)\\bigr)\\, \\Theta(t-t_k)$$",
                                style="text-align: center; margin: 1rem 0;"
                            ),
                            ui.p(
                                "can simultaneously account for both spectral features. By manipulating the "
                                "<strong>damping distribution \\(P_\\gamma\\)</strong> (discretized on a grid), "
                                "we show that a weakly-damped mode creates the sharp oscillatory alpha peak, "
                                "while a heavily-damped mode generates the broad scale-free shoulders around \\(f_0\\)."
                            ),
                            class_="sphinx-admonition-body"
                        ),
                        class_="sphinx-admonition mt-2"
                    ),
                    
                    ui.row(
                        ui.div(
                            ui.div(
                                ui.h6("Damping Mixture Distribution p(γ)", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_damping_pdf", height="340px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        ),
                        ui.div(
                            ui.div(
                                ui.h6("Representative Pulse Shapes", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_pulses", height="340px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        )
                    )
                ),
                
                # Tab 2: Simulated Signal and Rice Envelope
                ui.nav_panel(
                    "Simulated Signal & Rice Envelope",
                    ui.div(
                        ui.div("Rice Envelope-and-Carrier Decomposition", class_="sphinx-admonition-title"),
                        ui.div(
                            ui.p(
                                "Following Rice's mathematical analysis, any narrowband random process can be represented in carrier-and-envelope form: "
                            ),
                            ui.div(
                                "$$X(t) = A(t)\\cos\\bigl(2\\pi f_0 t + \\varphi(t)\\bigr) = X_c(t)\\cos(2\\pi f_0 t) + X_s(t)\\sin(2\\pi f_0 t)$$",
                                style="text-align: center; margin: 1rem 0;"
                            ),
                            ui.p(
                                "where \\(X_c(t)\\) and \\(X_s(t)\\) are monotonic exponential shot noises (slowly varying envelope components). "
                                "Below, we plot the simulated stochastic trace \\(X(t)\\) alongside its exact Rice amplitude envelope "
                                "\\(A(t) = \\sqrt{X_c^2(t) + Xs^2(t)}\\) to illustrate the waxing-and-waning dynamics."
                            ),
                            class_="sphinx-admonition-body"
                        ),
                        class_="sphinx-admonition mt-2"
                    ),
                    
                    ui.div(
                        ui.div(
                            ui.h6("Simulated Time Series X(t) & Rice Envelope A(t)", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                            ui.output_plot("plot_time_series", height="440px"),
                            class_="card"
                        )
                    )
                ),
                
                # Tab 3: Power Spectral Density
                ui.nav_panel(
                    "Power Spectral Density (PSD)",
                    ui.row(
                        ui.div(
                            ui.div(
                                ui.h6("Log-Log Power Spectrum (PSD)", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_psd_loglog", height="360px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        ),
                        ui.div(
                            ui.div(
                                ui.h6("Semi-Log Power Spectrum (PSD)", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_psd_semilog", height="360px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        )
                    ),
                    
                    # Parameters & Comparison table
                    ui.div(
                        ui.div(
                            ui.h6("Comparative Model Parameters & Exponent Fits", class_="card-header"),
                            ui.div(
                                ui.output_ui("comparison_table"),
                                class_="card-body p-0"
                            ),
                            class_="card mt-3 mb-2"
                        )
                    )
                )
            ),
            id="results-panel",
            class_="tab-pane fade mobile-tab-pane col-md-8 col-lg-9 ps-md-4",
            role="tabpanel",
            aria_labelledby="results-tab"
        ),
        class_="tab-content mobile-tab-content pt-2"
    ),
    title="Damped Oscillator Poisson Mixture Dashboard",
    theme=shinyswatch.theme.minty()
)


# ==========================================
# SHINY SERVER CODE
# ==========================================
def server(input, output, session):
    
    # 1. Synchronize Fit Range sliders to prevent overlap
    @reactive.Effect
    def _():
        fit_min = input.fit_min_f()
        ui.update_slider("fit_max_f", min=fit_min + 1)
        
    @reactive.Effect
    def _():
        fit_max = input.fit_max_f()
        ui.update_slider("fit_min_f", max=fit_max - 1)
        
    # 2. Text rendering of the second mode weight dynamically
    @output
    @render.ui
    def weight_mode2_text():
        w2 = 1.0 - input.w1()
        return ui.HTML(f"<div style='font-size: 0.9rem; font-weight: 600; color: #475569; margin: 0.5rem 0;'>Relative Mixture Weight (w₂): <span style='color: #0f766e;'>{w2:.2f}</span></div>")

    # 3. Core calculations (reactive calc)
    @reactive.calc
    def run_simulation_and_analysis():
        # Get UI parameters
        mu1_val = input.mu1()
        sigma1_val = input.sigma1()
        w1_val = input.w1()
        
        mu2_val = input.mu2()
        sigma2_val = input.sigma2()
        
        f0_val = input.f0()
        lambda_val = input.lambda_rate()
        noise_std_val = input.noise_std()
        T_val = input.sim_duration()
        
        fit_range = [input.fit_min_f(), input.fit_max_f()]
        
        # A. Run Simulation
        t, X, Xc, Xs, envelope = simulate_oscillatory_poisson(
            T_val, FS, f0_val, lambda_val, w1_val, mu1_val, sigma1_val, mu2_val, sigma2_val
        )
        
        # Add White Noise
        np.random.seed(123)  # Replicable additive noise
        if noise_std_val > 0:
            X_noisy = X + np.random.normal(0, noise_std_val, len(X))
        else:
            X_noisy = X.copy()
            
        # B. Compute Welch PSD (Empirical)
        # Ensure we have a reasonable window size (at least 2 seconds if possible)
        nperseg = int(min(len(X_noisy), FS * 2.0))
        freqs_emp, psd_emp = welch(X_noisy, fs=FS, nperseg=nperseg, noverlap=nperseg//2)
        
        # Filter frequencies between 1.0 and 80.0 Hz for cleaner viewing
        freq_mask = (freqs_emp >= 1.0) & (freqs_emp <= 80.0)
        freqs_emp_filtered = freqs_emp[freq_mask]
        psd_emp_filtered = psd_emp[freq_mask]
        
        # C. Compute Analytical PSD on the same frequency grid
        psd_ana, gamma_grid, p_gamma = compute_analytical_psd(
            freqs_emp_filtered, f0_val, lambda_val, w1_val, mu1_val, sigma1_val, mu2_val, sigma2_val
        )
        
        # Add the white noise level to analytical PSD for alignment: PSD_noisy = PSD_signal + 2 * dt * sigma^2
        # For continuous white noise PSD density: S_noise(f) = 2 * noise_std^2 / fs
        noise_density = 2.0 * (noise_std_val**2) / FS
        psd_ana_noisy = psd_ana + noise_density
        
        # D. Fit exponents
        # Fit empirical PSD in fit range
        emp_fit_offset, emp_fit_beta = fit_aperiodic_slope(freqs_emp_filtered, psd_emp_filtered, fit_range)
        
        # Fit analytical PSD in fit range
        ana_fit_offset, ana_fit_beta = fit_aperiodic_slope(freqs_emp_filtered, psd_ana_noisy, fit_range)
        
        return {
            "t": t,
            "X": X,
            "X_noisy": X_noisy,
            "Xc": Xc,
            "Xs": Xs,
            "envelope": envelope,
            "freqs": freqs_emp_filtered,
            "psd_emp": psd_emp_filtered,
            "psd_ana": psd_ana_noisy,
            "psd_ana_pure": psd_ana,
            "gamma_grid": gamma_grid,
            "p_gamma": p_gamma,
            "emp_beta": emp_fit_beta,
            "emp_offset": emp_fit_offset,
            "ana_beta": ana_fit_beta,
            "ana_offset": ana_fit_offset,
            "noise_density": noise_density
        }

    # ==========================================
    # RENDER PLOTS
    # ==========================================
    
    # Render PDF plot of gamma distribution
    @output
    @render.plot
    def plot_damping_pdf():
        res = run_simulation_and_analysis()
        gamma_grid = res["gamma_grid"]
        p_gamma = res["p_gamma"]
        
        mu1_val = input.mu1()
        sigma1_val = input.sigma1()
        w1_val = input.w1()
        
        mu2_val = input.mu2()
        sigma2_val = input.sigma2()
        
        # Calculate individual modes for visual reference
        d_gamma = gamma_grid[1] - gamma_grid[0]
        p1 = np.exp(-0.5 * ((gamma_grid - mu1_val) / sigma1_val) ** 2)
        p1 /= (np.sum(p1) * d_gamma)
        
        p2 = np.exp(-0.5 * ((gamma_grid - mu2_val) / sigma2_val) ** 2)
        p2 /= (np.sum(p2) * d_gamma)
        
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        ax.set_facecolor("#ffffff")
        
        # Clean axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        # Plot distributions
        ax.plot(gamma_grid, p_gamma, color='#0f766e', linewidth=2.5, label='Total Mixture P(γ)')
        ax.fill_between(gamma_grid, p_gamma, color='#0f766e', alpha=0.1)
        
        if w1_val > 0:
            ax.plot(gamma_grid, w1_val * p1, color='#10b981', linestyle='--', linewidth=1.5, label='Mode 1 (Weak Damping)')
        if w1_val < 1:
            ax.plot(gamma_grid, (1.0 - w1_val) * p2, color='#64748b', linestyle='--', linewidth=1.5, label='Mode 2 (Heavy Damping)')
            
        ax.set_xlabel('Damping Rate γ (rad/s)', fontsize=9.5, fontweight='medium', color='#1e293b')
        ax.set_ylabel('Probability Density', fontsize=9.5, fontweight='medium', color='#1e293b')
        ax.set_xlim(0, 70)
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
        
        plt.tight_layout()
        return fig

    # Render pulses
    @output
    @render.plot
    def plot_pulses():
        f0_val = input.f0()
        mu1_val = input.mu1()
        mu2_val = input.mu2()
        
        pulse_t = np.linspace(0, 1.5, 300)
        
        # Damped oscillatory pulses
        # Pulse 1: Weakly damped
        pulse1 = np.exp(-mu1_val * pulse_t) * np.cos(2.0 * np.pi * f0_val * pulse_t)
        # Pulse 2: Heavily damped
        pulse2 = np.exp(-mu2_val * pulse_t) * np.cos(2.0 * np.pi * f0_val * pulse_t)
        
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        ax.set_facecolor("#ffffff")
        
        # Clean axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        ax.plot(pulse_t, pulse1, color='#10b981', linewidth=2.0, label=f'Weak Damping (γ = {mu1_val:.1f} rad/s)')
        ax.plot(pulse_t, pulse2, color='#64748b', linewidth=2.0, alpha=0.75, label=f'Heavy Damping (γ = {mu2_val:.1f} rad/s)')
        
        ax.set_xlabel('Time (seconds)', fontsize=9.5, fontweight='medium', color='#1e293b')
        ax.set_ylabel('Pulse Amplitude', fontsize=9.5, fontweight='medium', color='#1e293b')
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color='#e2e8f0', linestyle='-', linewidth=1.0)
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
        
        plt.tight_layout()
        return fig

    # Render simulated signal and envelope
    @output
    @render.plot
    def plot_time_series():
        res = run_simulation_and_analysis()
        t = res["t"]
        X_noisy = res["X_noisy"]
        envelope = res["envelope"]
        Xc = res["Xc"]
        Xs = res["Xs"]
        
        # Plot only a 2.5-second window to zoom in and clearly see carrier vs envelope
        mask = (t >= 0.5) & (t <= 3.0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), dpi=100, sharex=True)
        ax1.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")
        
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(True, linestyle=':', alpha=0.3, color='#cbd5e1')
            
        # Top subplot: Generative signal X(t) with envelope A(t)
        ax1.plot(t[mask], X_noisy[mask], color='#cbd5e1', linewidth=1.0, alpha=0.8, label='Simulated EEG X(t) (Noisy)')
        ax1.plot(t[mask], res["X"][mask], color='#ef4444', linewidth=1.2, label='Generative Trace (Pure Signal)')
        ax1.plot(t[mask], envelope[mask], color='#0f766e', linewidth=2.0, label='Rice Amplitude Envelope A(t)')
        ax1.plot(t[mask], -envelope[mask], color='#0f766e', linestyle=':', linewidth=1.2, alpha=0.7)
        ax1.set_ylabel('Voltage (V)', fontsize=9.5, fontweight='medium')
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax1.set_title('Generative Process and Slowly-Varying Amplitude Envelope', fontsize=10.5, color='#0f766e', fontweight='semibold')
        
        # Bottom subplot: Low-pass shot noises Xc(t) and Xs(t)
        ax2.plot(t[mask], Xc[mask], color='#2563eb', linewidth=1.5, label='In-phase Carrier Envelope Xc(t)')
        ax2.plot(t[mask], Xs[mask], color='#ea580c', linewidth=1.5, label='Quadrature Carrier Envelope Xs(t)')
        ax2.set_xlabel('Time (seconds)', fontsize=9.5, fontweight='medium')
        ax2.set_ylabel('Amplitude (V)', fontsize=9.5, fontweight='medium')
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax2.set_title('Slowly-Varying Shot Noise Components (Rice Decomposition)', fontsize=10.5, color='#0f766e', fontweight='semibold')
        
        plt.tight_layout()
        return fig

    # Render log-log power spectrum
    @output
    @render.plot
    def plot_psd_loglog():
        res = run_simulation_and_analysis()
        freqs = res["freqs"]
        psd_emp = res["psd_emp"]
        psd_ana = res["psd_ana"]
        
        fit_min = input.fit_min_f()
        fit_max = input.fit_max_f()
        fit_range = [fit_min, fit_max]
        
        fig, ax = plt.subplots(figsize=(6, 4.0), dpi=100)
        ax.set_facecolor("#ffffff")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        # Plot empirical and analytical
        ax.loglog(freqs, psd_emp, color='#94a3b8', alpha=0.6, linewidth=1.2, label='Welch Empirical PSD (Noisy)')
        ax.loglog(freqs, psd_ana, color='#0f766e', linewidth=2.0, label='Analytical Mixture PSD')
        
        # Plot 1/f fit line
        f_fit = np.linspace(fit_min, fit_max, 100)
        # Empirical fit line
        emp_fit_line = 10**(res["emp_offset"]) / (f_fit**(res["emp_beta"]))
        ax.loglog(f_fit, emp_fit_line, color='#f97316', linestyle='--', linewidth=2.2, label=f'1/f^β Fit (β = {res["emp_beta"]:.2f})')
        
        # Highlight fitting range
        ax.axvspan(fit_min, fit_max, color='#f97316', alpha=0.07, label='Fit Exponent Band')
        
        ax.set_xlabel('Frequency (Hz, log scale)', fontsize=9.5, color='#1e293b')
        ax.set_ylabel('Power Density (V²/Hz, log scale)', fontsize=9.5, color='#1e293b')
        
        # Customize ticks
        ax.set_xticks([1, 2, 5, 10, 20, 40, 80])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0', loc='lower left')
        ax.grid(True, which='both', linestyle=':', alpha=0.3, color='#cbd5e1')
        ax.set_xlim(1.0, 80.0)
        
        # Safe dynamic bounds
        ax.set_ylim(bottom=min(np.min(psd_emp), np.min(psd_ana)) * 0.5)
        
        plt.tight_layout()
        return fig

    # Render semi-log power spectrum
    @output
    @render.plot
    def plot_psd_semilog():
        res = run_simulation_and_analysis()
        freqs = res["freqs"]
        psd_emp = res["psd_emp"]
        psd_ana = res["psd_ana"]
        
        fit_min = input.fit_min_f()
        fit_max = input.fit_max_f()
        
        fig, ax = plt.subplots(figsize=(6, 4.0), dpi=100)
        ax.set_facecolor("#ffffff")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        # Convert to log10(Power)
        ax.plot(freqs, np.log10(psd_emp), color='#94a3b8', alpha=0.6, linewidth=1.2, label='Welch Empirical PSD')
        ax.plot(freqs, np.log10(psd_ana), color='#0f766e', linewidth=2.0, label='Analytical Mixture PSD')
        
        # Highlight fitting range
        ax.axvspan(fit_min, fit_max, color='#f97316', alpha=0.07, label='Fit Exponent Band')
        
        ax.set_xlabel('Frequency (Hz, linear scale)', fontsize=9.5, color='#1e293b')
        ax.set_ylabel('log10(Power Density)', fontsize=9.5, color='#1e293b')
        
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0', loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.3, color='#cbd5e1')
        ax.set_xlim(1.0, 60.0)  # Zoom in slightly to highlight the alpha peak and shoulders
        
        plt.tight_layout()
        return fig

    # ==========================================
    # COMPARISON TABLE
    # ==========================================
    @output
    @render.ui
    def comparison_table():
        res = run_simulation_and_analysis()
        
        # Get UI parameters
        mu1 = input.mu1()
        sigma1 = input.sigma1()
        w1 = input.w1()
        
        mu2 = input.mu2()
        sigma2 = input.sigma2()
        w2 = 1.0 - w1
        
        f0 = input.f0()
        lambda_rate = input.lambda_rate()
        
        fit_min = input.fit_min_f()
        fit_max = input.fit_max_f()
        
        return ui.HTML(f"""
            <div class="table-responsive">
                <table class="table table-hover table-bordered mb-0" style="font-size: 0.9rem; border-color: #e2e8f0;">
                    <thead>
                        <tr>
                            <th scope="col" style="width: 30%;">Parameter / Metric</th>
                            <th scope="col" style="width: 35%;">Theoretical (Analytical) Value</th>
                            <th scope="col" style="width: 35%;">Empirical (Simulated) Value</th>
                        </tr>
                    </thead>
                    <tbody style="color: #334155; background-color: #ffffff;">
                        <tr>
                            <td style="font-weight: 500;">Damping Mode 1 (Weak)</td>
                            <td>Mean μ₁ = {mu1:.1f} rad/s, Width σ₁ = {sigma1:.1f}</td>
                            <td>w₁ = {w1:.2f} (Relative Weight)</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Damping Mode 2 (Heavy)</td>
                            <td>Mean μ₂ = {mu2:.1f} rad/s, Width σ₂ = {sigma2:.1f}</td>
                            <td>w₂ = {w2:.2f} (Relative Weight)</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Pulse Center Freq (f₀)</td>
                            <td>{f0:.1f} Hz</td>
                            <td>Peak observed near {f0:.1f} Hz</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Poisson Arrival Rate (λ)</td>
                            <td>{lambda_rate} Hz (Mean frequency)</td>
                            <td>{lambda_rate * input.sim_duration():.0f} total pulses generated</td>
                        </tr>
                        <tr style="background-color: #f0fdfa;">
                            <td style="font-weight: 600; color: #0f766e;">1/f^β Fitting Range</td>
                            <td colspan="2" style="font-weight: 600; color: #0f766e; text-align: center;">{fit_min} Hz to {fit_max} Hz</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600; color: #0f766e;">Aperiodic Exponent (β)</td>
                            <td style="font-weight: 600; color: #0f766e;">β = {res['ana_beta']:.3f}</td>
                            <td style="font-weight: 600; color: #d97706;">β = {res['emp_beta']:.3f}</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600; color: #0f766e;">Aperiodic Fit Offset (A)</td>
                            <td style="font-weight: 600; color: #0f766e;">Offset = {res['ana_offset']:.3f}</td>
                            <td style="font-weight: 600; color: #d97706;">Offset = {res['emp_offset']:.3f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """)

app = App(app_ui, server)
