import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfilt
from shiny import App, ui, render, reactive
import shinyswatch

# Define simulation parameters
FS_SIM = 1000.0  # Hz, standard temporal resolution (1 ms bins) for neural point processes

def generate_smooth_psp(t_vector, tau_decay_ms, tau_rise_ms):
    """
    Generate a double-exponential postsynaptic potential (PSP) waveform.
    Normalized to have a peak height of 1.0 in the time domain.
    """
    tau_d = tau_decay_ms / 1000.0  # Convert to seconds
    tau_r = max(0.1, tau_rise_ms) / 1000.0  # Convert to seconds and avoid division by zero
    
    if abs(tau_d - tau_r) < 1e-5:
        # Avoid singular case when rise = decay
        tau_r = tau_d * 0.99
        
    # Calculate peak time
    t_peak = (tau_d * tau_r / (tau_d - tau_r)) * np.log(tau_d / tau_r)
    # Normalization constant
    norm_const = 1.0 / (np.exp(-t_peak / tau_d) - np.exp(-t_peak / tau_r))
    
    # Calculate waveform
    h = norm_const * (np.exp(-t_vector / tau_d) - np.exp(-t_vector / tau_r))
    return h

def compute_psp_kernel(freqs, tau_decay_ms, tau_rise_ms):
    """
    Compute the magnitude-squared Fourier transform of the PSP filter: |H(f)|^2
    """
    tau_d = tau_decay_ms / 1000.0
    tau_r = max(0.1, tau_rise_ms) / 1000.0
    
    if abs(tau_d - tau_r) < 1e-5:
        tau_r = tau_d * 0.99
        
    t_peak = (tau_d * tau_r / (tau_d - tau_r)) * np.log(tau_d / tau_r)
    norm_const = 1.0 / (np.exp(-t_peak / tau_d) - np.exp(-t_peak / tau_r))
    
    # Analytical Fourier transform magnitude squared:
    # |H(f)|^2 = C^2 * (tau_d - tau_r)^2 / [ (1 + (2*pi*f*tau_d)^2) * (1 + (2*pi*f*tau_r)^2) ]
    numerator = (norm_const * (tau_d - tau_r)) ** 2
    denominator = (1.0 + (2.0 * np.pi * freqs * tau_d) ** 2) * (1.0 + (2.0 * np.pi * freqs * tau_r) ** 2)
    return numerator / denominator

def simulate_fpp_eeg(T, fs_sim, lambda_0, f0, bw, modulation, tau_decay, tau_rise, noise_std, apply_medium_filter, tau_med):
    """
    Simulate a Filtered Point Process EEG/LFP signal in the time domain.
    Returns:
        t: time vector
        Y_noisy: the simulated voltage signal with noise
        lambda_t: the time-varying intensity function (Cox rate)
        event_raster: binary vector of event occurrences (point process)
        Y_pure: pure signal without instrumentation noise
    """
    t = np.arange(0, T, 1.0 / fs_sim)
    dt = 1.0 / fs_sim
    
    # A. Generate the Cox intensity process lambda(t) = lambda_0 + r(t)
    # Generate white noise and filter it to a narrowband bandpass centered at f0
    np.random.seed(42)  # For replicable rate fluctuations across slider changes
    white_noise = np.random.normal(0, 1.0, len(t))
    
    # Bandpass filter centered at f0 with bandwidth bw
    f_low = max(0.1, f0 - bw / 2.0)
    f_high = min(fs_sim / 2.0 - 0.1, f0 + bw / 2.0)
    sos = butter(4, [f_low, f_high], btype='bandpass', fs=fs_sim, output='sos')
    r = sosfilt(sos, white_noise)
    
    # Scale filtered noise to have standard deviation matching modulation depth * lambda_0
    if np.std(r) > 0:
        r = r / np.std(r) * (modulation * lambda_0)
    else:
        r = np.zeros_like(t)
        
    lambda_t = lambda_0 + r
    # Ensure event rate is positive
    lambda_t = np.maximum(0.01, lambda_t)
    
    # B. Generate events using inhomogeneous Poisson point process sampling
    U = np.random.uniform(0.0, 1.0, len(t))
    event_raster = (U < lambda_t * dt).astype(float)
    
    # C. Create smooth PSP waveform and convolve with events
    # We truncate the PSP filter after 5 * tau_decay seconds to optimize convolution speed
    max_filter_t = 5.0 * (tau_decay / 1000.0)
    filter_t = np.arange(0, max_filter_t, dt)
    h_psp = generate_smooth_psp(filter_t, tau_decay, tau_rise)
    
    # Convolve event raster with postsynaptic potential
    Y_pure = np.convolve(event_raster, h_psp)[:len(t)]
    
    # D. Optionally apply extracellular low-pass medium filter
    if apply_medium_filter:
        tau_m = tau_med / 1000.0
        h_med = np.exp(-filter_t / tau_m) / tau_m  # Normalized so integral is 1
        Y_pure = np.convolve(Y_pure, h_med)[:len(t)] * dt
        
    # E. Add instrumentation white noise
    np.random.seed(123)  # Replicable additive noise
    if noise_std > 0:
        Y_noisy = Y_pure + np.random.normal(0, noise_std, len(t))
    else:
        Y_noisy = Y_pure.copy()
        
    return t, Y_noisy, lambda_t, event_raster, Y_pure

def fit_aperiodic_slope(freqs, psd, fit_range):
    """
    Fit a 1/f^beta line to the PSD in log-log space.
    """
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    f_fit = freqs[mask]
    psd_fit = psd[mask]
    
    if len(f_fit) < 2:
        return 0.0, 0.0
        
    log_f = np.log10(f_fit)
    log_psd = np.log10(np.maximum(psd_fit, 1e-15))
    
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
        ui.h2("Filtered Point Process (FPP) EEG Dashboard", class_="mt-3 mb-1 font-weight-bold"),
        ui.p("Simulating realistic EEG signals and 1/f power spectra from smooth postsynaptic potentials.", class_="text-muted mb-4", style="font-size: 1.05rem;"),
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
            # Synaptic Filter Settings (exponents & knee)
            ui.div(
                ui.h5("Synaptic PSP Filter", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #0f766e;"),
                ui.input_slider("tau_decay", "Synaptic Decay Constant (τ_decay, ms)", min=5.0, max=50.0, value=15.0, step=1.0),
                ui.input_slider("tau_rise", "Synaptic Rise Constant (τ_rise, ms)", min=0.5, max=5.0, value=1.5, step=0.1),
                ui.p("Defines the smooth double-exponential post-synaptic current shape.", class_="text-muted small mb-0"),
                class_="p-3 rounded mb-4",
                style="background-color: #f0fdfa; border: 1px solid #cbd5e1; border-left: 4px solid #0f766e;"
            ),
            
            # Point Process Rate Settings (rhythmic alpha vs broadband height)
            ui.div(
                ui.h5("Timing & Firing Rates (Cox Process)", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #1e293b;"),
                ui.input_slider("lambda_0", "Mean Firing Rate (λ₀, Hz)", min=50, max=1000, value=300, step=50),
                ui.input_slider("f0", "Oscillation Center Freq (f₀, Hz)", min=4.0, max=20.0, value=10.0, step=0.5),
                ui.input_slider("bw", "Oscillation Bandwidth (BW, Hz)", min=0.5, max=5.0, value=1.5, step=0.1),
                ui.input_slider("modulation", "Alpha Modulation Depth (m)", min=0.0, max=1.0, value=0.5, step=0.05),
                ui.p("Models rhythmic event rates to generate the alpha bump.", class_="text-muted small mb-0"),
                class_="p-3 rounded mb-4",
                style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #475569;"
            ),
            
            # Additional Extracellular Filter
            ui.div(
                ui.h5("Extracellular Medium Options", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #7c3aed;"),
                ui.input_checkbox("apply_medium_filter", "Apply Extracellular Low-pass", value=False),
                ui.panel_conditional(
                    "input.apply_medium_filter === true",
                    ui.input_slider("tau_med", "Medium Decay Constant (τ_med, ms)", min=1.0, max=20.0, value=5.0, step=0.5),
                    ui.p("Simulates diffusion filtering, adding another pole to the slope.", class_="text-muted small mb-0")
                ),
                class_="p-3 rounded mb-4",
                style="background-color: #faf5ff; border: 1px solid #cbd5e1; border-left: 4px solid #7c3aed;"
            ),
            
            # Global Simulation & Fitting Settings
            ui.div(
                ui.h5("Global Settings & Fitting", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #d97706;"),
                ui.input_slider("noise_std", "White Noise SD (Instrumentation)", min=0.0, max=0.5, value=0.08, step=0.01),
                ui.input_slider("sim_duration", "Simulation Duration (T)", min=1.0, max=8.0, value=4.0, step=0.5),
                ui.input_slider("fit_min_f", "Fit Exponent Range Min (Hz)", min=15, max=30, value=20, step=1),
                ui.input_slider("fit_max_f", "Fit Exponent Range Max (Hz)", min=35, max=100, value=60, step=5),
                class_="p-3 rounded",
                style="background-color: #fffbeb; border: 1px solid #cbd5e1; border-left: 4px solid #d97706;"
            ),
            id="params-panel",
            class_="tab-pane fade show active mobile-tab-pane col-md-4 col-lg-3 pe-md-4",
            role="tabpanel",
            aria_labelledby="params-tab"
        ),
        
        # Results Section (Right / Tabs)
        ui.div(
            # Navigation Tabs
            ui.navset_pill(
                # Tab 1: FPP Theory and Waveforms
                ui.nav_panel(
                    "FPP Theory & Filters",
                    ui.div(
                        ui.div("The Filtered Point Process (FPP) Concept (Bloniasz et al., 2025)", class_="sphinx-admonition-title"),
                        ui.div(
                            ui.p(
                                "An extracellular neural field recording (EEG/LFP) arises from the superposition of many underlying postsynaptic potentials (PSPs). "
                                "Instead of modeling each pulse as a bandpass filter, the FPP framework models the signal as a point process "
                                "convolved with a smooth low-pass synaptic waveform: "
                            ),
                            ui.div(
                                "$$Y(t) = \\sum_n h_{PSP}(t - t_n)$$",
                                style="text-align: center; margin: 0.75rem 0;"
                            ),
                            ui.p(
                                "According to Bartlett's theorem, the power spectrum decomposes linearly into two distinct biological contributors: "
                            ),
                            ui.div(
                                "$$S(f) = |H_{PSP}(f)|^2 \\left[ \\lambda_0 + S_r(f) \\right]$$",
                                style="text-align: center; margin: 0.75rem 0;"
                            ),
                            ui.tags.ul(
                                ui.tags.li(
                                    ui.tags.strong("Synaptic Waveform Filter \\(|H_{PSP}(f)|^2\\): "),
                                    "A smooth double-exponential. Because it acts as a low-pass filter, the white-noise Poisson rate floor \\(\\lambda_0\\) "
                                    "is shaped into a beautiful 1/f-like background that continues to climb all the way to 0 Hz."
                                ),
                                ui.tags.li(
                                    ui.tags.strong("Cox Oscillatory Rate Spectrum \\(S_r(f)\\): "),
                                    "A narrowband peak centered at \\(f_0\\) representing rhythmic synchronization of the event arrival times. "
                                    "This creates the alpha bump organically on top of the 1/f background, resolving the low-frequency rolloff issue."
                                )
                            ),
                            class_="sphinx-admonition-body"
                        ),
                        class_="sphinx-admonition mt-2"
                    ),
                    
                    ui.row(
                        ui.div(
                            ui.div(
                                ui.h6("Synaptic PSP Filter in Time & Frequency", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_synaptic_filter", height="350px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        ),
                        ui.div(
                            ui.div(
                                ui.h6("Point Process Firing Rate Bartlett Spectrum", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                                ui.output_plot("plot_rate_spectrum", height="350px"),
                                class_="card"
                            ),
                            class_="col-lg-6"
                        )
                    )
                ),
                
                # Tab 2: Simulated EEG Trace & Raster
                ui.nav_panel(
                    "Simulated EEG Trace & Raster",
                    ui.div(
                        ui.div("Cox Process Time-Series stacked visualization", class_="sphinx-admonition-title"),
                        ui.div(
                            ui.p(
                                "Below is the real-time simulation of the inhomogeneous Poisson process. "
                                "Observe how the events in the bottom raster plot cluster closely around the peaks of the time-varying intensity "
                                "\\(\\lambda(t)\\) (the alpha wave). Because the postsynaptic potential filter is smooth, their convolution "
                                "results in a beautifully organic LFP/EEG trace \\(Y(t)\\) with waxing-and-waning amplitude modulations, "
                                "replicating biological EEG dynamics perfectly."
                            ),
                            class_="sphinx-admonition-body"
                        ),
                        class_="sphinx-admonition mt-2"
                    ),
                    
                    ui.div(
                        ui.div(
                            ui.h6("EEG Trace, Rate Process, and Firing Raster Stack", class_="card-header bg-transparent font-weight-bold text-secondary text-center"),
                            ui.output_plot("plot_eeg_stack", height="520px"),
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
    title="Filtered Point Process (FPP) EEG Dashboard",
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
        ui.update_slider("fit_max_f", min=fit_min + 5)
        
    @reactive.Effect
    def _():
        fit_max = input.fit_max_f()
        ui.update_slider("fit_min_f", max=fit_max - 5)
        
    # 2. Reactive simulation calc
    @reactive.calc
    def run_simulation_and_analysis():
        # Get UI parameters
        tau_decay_val = input.tau_decay()
        tau_rise_val = input.tau_rise()
        lambda_0_val = input.lambda_0()
        f0_val = input.f0()
        bw_val = input.bw()
        modulation_val = input.modulation()
        
        noise_std_val = input.noise_std()
        T_val = input.sim_duration()
        
        apply_medium_val = input.apply_medium_filter()
        tau_med_val = input.tau_med() if apply_medium_val else 1.0
        
        fit_range = [input.fit_min_f(), input.fit_max_f()]
        
        # A. Run High-Resolution FPP Time-domain Simulation
        t, Y, lambda_t, event_raster, Y_pure = simulate_fpp_eeg(
            T_val, FS_SIM, lambda_0_val, f0_val, bw_val, modulation_val,
            tau_decay_val, tau_rise_val, noise_std_val, apply_medium_val, tau_med_val
        )
        
        # B. Compute Welch PSD (Empirical)
        # 1000 Hz simulation permits high quality Welch spectrum up to 500 Hz
        nperseg = int(FS_SIM * 1.5)  # 1.5 second window
        freqs_emp, psd_emp = welch(Y, fs=FS_SIM, nperseg=nperseg, noverlap=nperseg//2)
        
        # Filter frequencies between 1.0 and 150.0 Hz for high fidelity, clean view
        freq_mask = (freqs_emp >= 1.0) & (freqs_emp <= 150.0)
        freqs_emp_filtered = freqs_emp[freq_mask]
        psd_emp_filtered = psd_emp[freq_mask]
        
        # C. Compute Analytical PSD on the same frequency grid
        # Bartlett spectrum: S(f) = |H_PSP(f)|^2 * [lambda_0 + S_r(f)] * dt
        # First, compute |H_PSP(f)|^2
        h_psp_kernel = compute_psp_kernel(freqs_emp_filtered, tau_decay_val, tau_rise_val)
        
        # Optionally multiply by extracellular low pass filter
        if apply_medium_val:
            tau_m_sec = tau_med_val / 1000.0
            medium_kernel = 1.0 / (1.0 + (2.0 * np.pi * freqs_emp_filtered * tau_m_sec) ** 2)
            h_psp_kernel *= medium_kernel
            
        # The oscillatory rate spectrum S_r(f) is a Gaussian bump centered at f0
        sigma_rate = modulation_val * lambda_0_val
        sigma_osc = bw_val / 2.0
        # Normalization to ensure integrated power matches sigma_rate^2
        amp_osc = (sigma_rate ** 2) / (np.sqrt(2.0 * np.pi) * sigma_osc)
        s_r = amp_osc * np.exp(-0.5 * ((freqs_emp_filtered - f0_val) / sigma_osc) ** 2)
        
        # Bartlett point-process spectrum: S_N = lambda_0 + s_r
        s_n = lambda_0_val + s_r
        
        # Combine: S_field = |H_PSP|^2 * S_N * dt (discrete scaling)
        dt = 1.0 / FS_SIM
        psd_ana = h_psp_kernel * s_n * dt
        
        # Add white noise density floor
        noise_density = 2.0 * (noise_std_val**2) / FS_SIM
        psd_ana_noisy = psd_ana + noise_density
        
        # D. Fit exponents
        emp_fit_offset, emp_fit_beta = fit_aperiodic_slope(freqs_emp_filtered, psd_emp_filtered, fit_range)
        ana_fit_offset, ana_fit_beta = fit_aperiodic_slope(freqs_emp_filtered, psd_ana_noisy, fit_range)
        
        return {
            "t": t,
            "Y": Y,
            "Y_pure": Y_pure,
            "lambda_t": lambda_t,
            "event_raster": event_raster,
            "freqs": freqs_emp_filtered,
            "psd_emp": psd_emp_filtered,
            "psd_ana": psd_ana_noisy,
            "psd_ana_pure": psd_ana,
            "h_psp_kernel": h_psp_kernel,
            "s_n": s_n,
            "s_r": s_r,
            "emp_beta": emp_fit_beta,
            "emp_offset": emp_fit_offset,
            "ana_beta": ana_fit_beta,
            "ana_offset": ana_fit_offset,
            "noise_density": noise_density
        }

    # ==========================================
    # RENDER PLOTS
    # ==========================================
    
    # 1. Render Synaptic PSP filter plot
    @output
    @render.plot
    def plot_synaptic_filter():
        tau_decay_val = input.tau_decay()
        tau_rise_val = input.tau_rise()
        
        t_max = 5.0 * (tau_decay_val / 1000.0)
        filter_t = np.linspace(0, t_max, 500)
        h_psp = generate_smooth_psp(filter_t, tau_decay_val, tau_rise_val)
        
        # Compute frequency response
        freqs_dense = np.linspace(1.0, 150.0, 300)
        kernel = compute_psp_kernel(freqs_dense, tau_decay_val, tau_rise_val)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.3), dpi=100)
        for ax in [ax1, ax2]:
            ax.set_facecolor("#ffffff")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
            
        # Left plot: Time domain waveform
        ax1.plot(filter_t * 1000.0, h_psp, color='#0f766e', linewidth=2.0)
        ax1.set_xlabel('Time (ms)', fontsize=9.0)
        ax1.set_ylabel('Amplitude (norm)', fontsize=9.0)
        ax1.set_title('PSP Waveform h_PSP(t)', fontsize=9.5, fontweight='bold', color='#0f766e')
        # Highlight peak
        t_p = (tau_decay_val * tau_rise_val / (tau_decay_val - tau_rise_val)) * np.log(tau_decay_val / tau_rise_val)
        ax1.axvline(t_p, color='#d97706', linestyle=':', label=f'Peak ({t_p:.1f} ms)')
        ax1.legend(fontsize=7.5)
        
        # Right plot: Frequency domain kernel
        ax2.loglog(freqs_dense, kernel, color='#0f766e', linewidth=2.0)
        ax2.set_xlabel('Frequency (Hz)', fontsize=9.0)
        ax2.set_ylabel('Kernel |H(f)|²', fontsize=9.0)
        ax2.set_title('Spectral Kernel |H_PSP(f)|²', fontsize=9.5, fontweight='bold', color='#0f766e')
        ax2.set_xticks([1, 10, 50, 150])
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        return fig

    # 2. Render Point Process Bartlett spectrum
    @output
    @render.plot
    def plot_rate_spectrum():
        res = run_simulation_and_analysis()
        freqs = res["freqs"]
        s_n = res["s_n"]
        s_r = res["s_r"]
        lambda_0_val = input.lambda_0()
        
        fig, ax = plt.subplots(figsize=(6, 3.3), dpi=100)
        ax.set_facecolor("#ffffff")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
        
        ax.plot(freqs, s_n, color='#7c3aed', linewidth=2.0, label='Point Process Spectrum S_N(f)')
        ax.axhline(lambda_0_val, color='#64748b', linestyle='--', label=f'Broadband Rate Floor (λ₀ = {lambda_0_val} Hz)')
        ax.fill_between(freqs, lambda_0_val, s_n, color='#7c3aed', alpha=0.1, label='Rhythmic Firing Component S_r(f)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=9.0)
        ax.set_ylabel('Power Density (events/Hz)', fontsize=9.0)
        ax.set_title('Bartlett Spectrum: S_N(f) = λ₀ + S_r(f)', fontsize=9.5, fontweight='bold', color='#7c3aed')
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.95)
        ax.set_xlim(1.0, 50.0)
        
        plt.tight_layout()
        return fig

    # 3. Render EEG voltage trace, Firing Rate, and Raster stack
    @output
    @render.plot
    def plot_eeg_stack():
        res = run_simulation_and_analysis()
        t = res["t"]
        Y_noisy = res["Y"]
        lambda_t = res["lambda_t"]
        raster = res["event_raster"]
        
        # Display a 1.0 second zoom window (e.g. from 0.5s to 1.5s) to clearly see events and waves
        zoom_mask = (t >= 0.5) & (t <= 1.5)
        t_zoom = t[zoom_mask]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5.0), dpi=100, sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1, 0.6]})
        
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#ffffff")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(True, linestyle=':', alpha=0.3, color='#cbd5e1')
            
        # A. Top Subplot: EEG voltage signal
        ax1.plot(t_zoom, Y_noisy[zoom_mask], color='#64748b', linewidth=0.8, alpha=0.6, label='Simulated EEG with White Noise')
        ax1.plot(t_zoom, res["Y_pure"][zoom_mask], color='#0f766e', linewidth=1.5, label='Pure Synaptic FPP Signal')
        ax1.set_ylabel('Voltage (norm)', fontsize=8.5)
        ax1.set_title('A. Continuous Extracellular Voltage Y(t) = sum_n h_PSP(t - t_n)', fontsize=9.5, fontweight='bold', color='#0f766e')
        ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
        
        # B. Middle Subplot: Cox Firing Rate Process lambda(t)
        ax2.plot(t_zoom, lambda_t[zoom_mask], color='#7c3aed', linewidth=1.5)
        ax2.axhline(input.lambda_0(), color='#64748b', linestyle=':', alpha=0.7, label=f'Mean Rate λ₀ = {input.lambda_0()} Hz')
        ax2.set_ylabel('Firing Rate (Hz)', fontsize=8.5)
        ax2.set_title('B. Time-Varying Firing Rate Process λ(t) = λ₀ + r(t) (Alpha Modulated)', fontsize=9.5, fontweight='bold', color='#7c3aed')
        ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
        
        # C. Bottom Subplot: Event Raster
        event_indices = np.where(raster[zoom_mask] > 0.5)[0]
        event_times = t_zoom[event_indices]
        
        ax3.vlines(event_times, 0, 1, colors='#111827', linewidth=1.0)
        ax3.set_yticks([])
        ax3.set_xlabel('Time (seconds)', fontsize=9.0)
        ax3.set_title('C. Discrete Poisson Event Arrival Times (Raster)', fontsize=9.5, fontweight='bold', color='#111827')
        
        # Align ticks beautifully
        plt.xlim(0.5, 1.5)
        plt.tight_layout()
        return fig

    # 4. Render Log-Log PSD plot
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
        
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
        ax.set_facecolor("#ffffff")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        # Plot empirical and analytical spectra
        ax.loglog(freqs, psd_emp, color='#94a3b8', alpha=0.5, linewidth=1.0, label='Welch Empirical PSD (Noisy)')
        ax.loglog(freqs, psd_ana, color='#0f766e', linewidth=2.0, label='Bartlett Theoretical PSD')
        
        # Plot 1/f fit line
        f_fit = np.linspace(fit_min, fit_max, 100)
        emp_fit_line = 10**(res["emp_offset"]) / (f_fit**(res["emp_beta"]))
        ax.loglog(f_fit, emp_fit_line, color='#d97706', linestyle='--', linewidth=2.2, label=f'1/f^β Fit (β = {res["emp_beta"]:.2f})')
        
        # Highlight fitting range
        ax.axvspan(fit_min, fit_max, color='#d97706', alpha=0.07, label='Fit Exponent Band')
        
        ax.set_xlabel('Frequency (Hz, log scale)', fontsize=9.0, color='#1e293b')
        ax.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=9.0, color='#1e293b')
        
        # Customize ticks
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 150])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(1.0, 150.0)
        
        # dynamic limits
        ax.set_ylim(bottom=min(np.min(psd_emp), np.min(psd_ana)) * 0.5)
        
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0', loc='lower left')
        ax.grid(True, which='both', linestyle=':', alpha=0.3, color='#cbd5e1')
        
        plt.tight_layout()
        return fig

    # 5. Render Semi-Log PSD plot
    @output
    @render.plot
    def plot_psd_semilog():
        res = run_simulation_and_analysis()
        freqs = res["freqs"]
        psd_emp = res["psd_emp"]
        psd_ana = res["psd_ana"]
        
        fit_min = input.fit_min_f()
        fit_max = input.fit_max_f()
        
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
        ax.set_facecolor("#ffffff")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        
        ax.plot(freqs, np.log10(psd_emp), color='#94a3b8', alpha=0.5, linewidth=1.0, label='Welch Empirical PSD')
        ax.plot(freqs, np.log10(psd_ana), color='#0f766e', linewidth=2.0, label='Theoretical PSD')
        
        # Highlight fitting range
        ax.axvspan(fit_min, fit_max, color='#d97706', alpha=0.07, label='Fit Exponent Band')
        
        ax.set_xlabel('Frequency (Hz, linear scale)', fontsize=9.0, color='#1e293b')
        ax.set_ylabel('log10(Power Density)', fontsize=9.0, color='#1e293b')
        
        ax.legend(fontsize=8, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0', loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.3, color='#cbd5e1')
        ax.set_xlim(1.0, 80.0)  # Zoom in slightly to highlight the alpha peak and background
        
        plt.tight_layout()
        return fig

    # ==========================================
    # COMPARISON TABLE
    # ==========================================
    @output
    @render.ui
    def comparison_table():
        res = run_simulation_and_analysis()
        
        tau_decay_val = input.tau_decay()
        tau_rise_val = input.tau_rise()
        lambda_0_val = input.lambda_0()
        f0_val = input.f0()
        bw_val = input.bw()
        modulation_val = input.modulation()
        
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
                            <td style="font-weight: 500;">Synaptic Timescales</td>
                            <td>τ_decay = {tau_decay_val:.1f} ms, τ_rise = {tau_rise_val:.1f} ms</td>
                            <td>Synaptic peak time: {((tau_decay_val * tau_rise_val / (tau_decay_val - tau_rise_val)) * np.log(tau_decay_val / tau_rise_val) if tau_decay_val != tau_rise_val else tau_decay_val):.1f} ms</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Mean Firing Rate (λ₀)</td>
                            <td>{lambda_0_val} Hz (Poisson base)</td>
                            <td>{len(np.where(res['event_raster'] > 0.5)[0]) / input.sim_duration():.1f} Hz actual event frequency</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Alpha Rhythm Freq (f₀)</td>
                            <td>{f0_val:.1f} Hz (BW = {bw_val:.1f} Hz)</td>
                            <td>Modulation depth: {modulation_val * 100:.0f}%</td>
                        </tr>
                        <tr style="background-color: #f0fdfa;">
                            <td style="font-weight: 600; color: #0f766e;">1/f^β Fitting Bandwidth</td>
                            <td colspan="2" style="font-weight: 600; color: #0f766e; text-align: center;">{fit_min} Hz to {fit_max} Hz</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600; color: #0f766e;">Spectral Exponent (β)</td>
                            <td style="font-weight: 600; color: #0f766e;">β = {res['ana_beta']:.3f}</td>
                            <td style="font-weight: 600; color: #d97706;">β = {res['emp_beta']:.3f}</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600; color: #0f766e;">Aperiodic Offset (A)</td>
                            <td style="font-weight: 600; color: #0f766e;">Offset = {res['ana_offset']:.3f}</td>
                            <td style="font-weight: 600; color: #d97706;">Offset = {res['emp_offset']:.3f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """)

app = App(app_ui, server)
