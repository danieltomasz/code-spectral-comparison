from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from specparam import SpectralModel
from shiny import App, ui, render, reactive
import shinyswatch


def simulate_power_spectrum(freqs, model_type, b, chi, k, add_peak, peak_freq, peak_amp, peak_width, noise_level):
    """
    Simulate a neurophysiological power spectrum.
    
    Returns:
        sim_power (np.ndarray): 1D array of simulated log10 power (with noise)
        true_power (np.ndarray): 1D array of true log10 power (no noise)
        true_aperiodic (np.ndarray): 1D array of true log10 aperiodic component
    """
    # 1/f aperiodic component (with optional knee)
    if model_type == "knee":
        true_aperiodic = b - np.log10(k + freqs**chi)
    else:
        true_aperiodic = b - np.log10(freqs**chi)
        
    true_power = true_aperiodic.copy()
    if add_peak:
        # Gaussian peak: standard deviation scaled to width/2.0
        peak_signal = peak_amp * norm.pdf(freqs, loc=peak_freq, scale=peak_width / 2.0)
        true_power += peak_signal
        
    # Replicable pseudo-random experimental noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, len(freqs)) if noise_level > 0 else np.zeros(len(freqs))
    sim_power = true_power + noise
    
    return sim_power, true_power, true_aperiodic


def fit_spectral_model(freqs, power_spectrum, mode, fit_range, max_n_peaks, peak_threshold, min_peak_height, peak_width_limits, gauss_overlap_thresh):
    """
    Initialize and fit a specparam SpectralModel to a power spectrum.
    """
    fm = SpectralModel(
        aperiodic_mode=mode,
        max_n_peaks=max_n_peaks,
        peak_threshold=peak_threshold,
        min_peak_height=min_peak_height,
        peak_width_limits=peak_width_limits,
        gauss_overlap_thresh=gauss_overlap_thresh,
        verbose=False
    )
    # SpectralModel fits power values in linear space (10**log_power)
    fm.fit(freqs, 10**power_spectrum, fit_range)
    return fm


def extract_fit_results(fm, fit_freqs, mask, sim_power):
    """
    Extract fitted curve coordinates, metrics, and parameters from a SpectralModel.
    """
    if not hasattr(fm, 'results') or fm.results is None or not fm.results.has_model:
        return np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, []
        
    res = fm.results
    mode = fm.modes.aperiodic.name
    
    offset = float(res.params.aperiodic.params[0])
    exponent = float(res.params.aperiodic.params[-1])
    
    if mode == 'knee':
        knee = float(res.params.aperiodic.params[1])
        # Avoid log of zero/negative values
        res_curve = offset - np.log10(max(1e-5, knee) + fit_freqs**exponent)
    else:
        knee = 0.0
        res_curve = offset - np.log10(fit_freqs**exponent)
        
    # Goodness-of-fit metrics
    r2 = float(res.metrics.results['gof_rsquared'])
    mae = float(res.metrics.results['error_mae'])
    
    # Safely extract periodic peaks list: [(cf, pw, bw), ...]
    try:
        pk_params = res.params.periodic.params
        if pk_params is not None and len(pk_params) > 0:
            pk = np.atleast_2d(np.asarray(pk_params, dtype=float))
            peaks = [
                (float(r[0]), float(r[1]), float(r[2]))
                for r in pk
                if r.size >= 3 and not np.any(np.isnan(r[:3]))
            ]
        else:
            peaks = []
    except Exception:
        peaks = []
        
    return res_curve, offset, knee, exponent, r2, mae, peaks

# Define Sphinx-like document layout aligned with specparam-tools.github.io
app_ui = ui.page_fluid(
    # Custom CSS head content for PyData/Sphinx documentation aesthetic
    ui.head_content(
        # Load MathJax CDN for LaTeX mathematical rendering
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
        # Load custom Sphinx stylesheet
        ui.include_css(Path(__file__).parent / "styles.css")
    ),
    
    # Header Area
    ui.div(
        ui.h2("SpecParam Simulation & Fitting", class_="mt-3 mb-1 font-weight-bold", style="color: #2c3e50;"),
        ui.p("Interactive simulation and fitting bias analysis for neurophysiological power spectra.", class_="text-muted mb-4", style="font-size: 1.05rem;"),
        class_="container-fluid p-0 pt-2"
    ),
    
    # Mobile Tabs Navigation (rendered ONLY on mobile)
    ui.HTML("""
        <ul class="nav nav-tabs mobile-tabs d-md-none" id="mobileTab" role="tablist">
            <li class="nav-item" role="presentation" style="flex: 1;">
                <button class="nav-link active w-100" id="params-tab" data-bs-toggle="tab" data-bs-target="#params-panel" type="button" role="tab" aria-controls="params-panel" aria-selected="true">
                    Parameters
                </button>
            </li>
            <li class="nav-item" role="presentation" style="flex: 1;">
                <button class="nav-link w-100" id="results-tab" data-bs-toggle="tab" data-bs-target="#results-panel" type="button" role="tab" aria-controls="results-panel" aria-selected="false">
                    Results
                </button>
            </li>
        </ul>
    """),
    
    # Main Content Container
    ui.row(
        # Parameter Sidebar (desktop: left column, mobile: active tab panel)
        ui.div(
            # Card 1: Simulation Settings Card (Teal 50 background with solid left teal accent)
            ui.div(
                ui.h5("Simulation Configuration", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_radio_buttons(
                    "model_type",
                    "Simulated Model Type",
                    {"knee": "Knee (Plateau)", "fixed": "Fixed (1/f)"},
                    selected="knee"
                ),
                ui.hr(style="margin: 0.75rem 0; border-color: #e2e8f0;"),
                
                ui.input_slider("b", "Offset (b)", min=0.5, max=5.0, value=3.0, step=0.1),
                ui.input_slider("chi", "Exponent (exponent)", min=0.5, max=4.0, value=2.0, step=0.1),
                
                # Conditional Knee Parameter Sliders - only visible when simulating a Knee model
                ui.panel_conditional(
                    "input.model_type === 'knee'",
                    ui.input_slider("fk", "Knee Frequency (Hz)", min=1.0, max=100.0, value=10.0, step=0.5),
                    ui.input_slider("k", "Knee Parameter (k)", min=0.1, max=1000.0, value=100.0, step=0.5),
                    ui.p("Defines the bending knee frequency and corresponding parameter", class_="text-muted small mb-3")
                ),
                
                ui.h5("Peak Configuration", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_checkbox("add_peak", "Add Simulated Peak", value=True),
                
                # Conditional Peak Sliders - only visible when peak is checked
                ui.panel_conditional(
                    "input.add_peak === true",
                    ui.input_slider("peak_freq", "Peak Frequency (Hz)", min=2, max=50, value=10, step=1),
                    ui.input_slider("peak_amp", "Peak Amplitude (log units)", min=0.05, max=10.5, value=0.35, step=0.05),
                    ui.input_slider("peak_width", "Peak Bandwidth (Hz)", min=0.5, max=10.0, value=1.8, step=0.1)
                ),
                
                ui.h5("Noise Configuration", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("noise_level", "Spectral Noise Level (SD)", min=0.0, max=1.6, value=0.04, step=0.01),
                
                class_="p-3 rounded mb-4",
                style="background-color: #f0fdfa; border: 1px solid #cbd5e1; border-left: 4px solid #0f766e;"
            ),
                        # Card 2: Butterworth Filter Configuration (Slate 50 background with solid Slate left border)
            ui.div(
                ui.h5("Butterworth Filter Options", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_checkbox("apply_filter", "Apply Butterworth Filter", value=False),
                
                ui.output_ui("butterworth_filter_ui"),
                class_="p-3 rounded mt-4",
                style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #475569;"
            ),
            
            # Card 3: Fitting Settings Card (Warm Coral 50 background with solid left coral accent)
            ui.div(
                ui.h5("Fitting Configuration", class_="mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                
                ui.h5("Fitting Boundaries", class_="mt-3 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("min_f", "Min Fit Frequency (Hz)", min=1, max=50, value=1, step=1),
                ui.input_slider("max_f", "Max Fit Frequency (Hz)", min=2, max=100, value=100, step=1),
                
                ui.h5("specparam Fit Settings", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("max_n_peaks", "Max Number of Peaks", min=0, max=5, value=2, step=1),
                ui.input_slider("peak_threshold", "Peak Threshold (SD)", min=1.0, max=5.0, value=2.0, step=0.1),
                ui.input_slider("min_peak_height", "Min Peak Height (log units)", min=0.0, max=1.0, value=0.0, step=0.05),
                ui.input_slider("peak_width_limits", "Peak Bandwidth Limits (Hz)", min=0.5, max=15.0, value=[0.5, 12.0], step=0.5),
                ui.input_slider("gauss_overlap_thresh", "Overlap Threshold (SD)", min=0.0, max=2.0, value=0.75, step=0.05),
                
                class_="p-3 rounded",
                style="background-color: #fff8f5; border: 1px solid #cbd5e1; border-left: 4px solid #ea580c;"
            ),
            

            
            id="params-panel",
            class_="tab-pane fade show active mobile-tab-pane col-md-4 col-lg-3 pe-md-4",
            role="tabpanel",
            aria_labelledby="params-tab"
        ),
        
        # Results Content (desktop: right column, mobile: tab panel)
        ui.div(
            # Sphinx-style Note Admonition
            ui.div(
                ui.div("Note: SpecParam Parametrisation", class_="sphinx-admonition-title"),
                ui.div(
                    ui.p(
                        "Electrophysiological neural power spectra exhibit an aperiodic component modeled as: "
                    ),
                    ui.div(
                        "$$S(f) = b - \\log_{10}(k + f^{\\chi})$$",
                        style="text-align: center; margin: 1rem 0;"
                    ),
                    ui.p(
                        "Where \\(b\\) is the offset, \\(\\chi\\) is the exponent, and \\(k\\) is the knee parameter. "
                        "To understand the role and mathematical behavior of the knee parameter \\(k\\), we can examine its limits at low and high frequencies:"
                    ),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("Low Frequency Plateau (\\(f \\ll k^{1/\\chi}\\)): "),
                            "When the frequency \\(f\\) is small enough such that \\(f^{\\chi}\\) is much smaller than \\(k\\), the term becomes negligible. "
                            "This yields a flat, horizontal plateau of height \\(b - \\log_{10}(k)\\), jointly determined by the offset and knee."
                        ),
                        ui.tags.li(
                            ui.tags.strong("High Frequency Power-Law (\\(f \\gg k^{1/\\chi}\\)): "),
                            "When \\(f\\) is large, \\(k\\) becomes negligible, yielding a straight line with a constant negative slope of \\(-\\chi\\) in log-log space."
                        ),
                        ui.tags.li(
                            ui.tags.strong("Knee Frequency (\\(f_k\\)): "),
                            "The Knee Frequency \\(f_k = k^{1/\\chi}\\) marks the transition point (bend) where the power drops by \\(\\log_{10}(2) \\approx 0.3\\) log units (3 dB) "
                            "below the low-frequency plateau. A larger knee parameter \\(k\\) corresponds to a higher knee frequency, shifting the bend to the right."
                        ),
                        style="margin-bottom: 1rem;"
                    ),
                    class_="sphinx-admonition-body"
                ),
                class_="sphinx-admonition"
            ),
            
            # Middle row: Elegant full-width plot with scroll wrapper
            ui.div(
                ui.div(
                    ui.h6(ui.output_text("plot_title"), class_="card-header bg-transparent text-center font-weight-bold text-secondary"),
                    ui.div(
                        ui.div(
                            ui.output_plot("plot_semilog", height="480px"),
                            class_="plot-min-width-wrapper"
                        ),
                        class_="plot-scroll-container"
                    ),
                    ui.div(
                        ui.input_switch("x_log", "Logarithmic Frequency Scale (X-axis)", value=True),
                        ui.input_switch("limit_fit_bounds", "Limit X-axis to fitting boundaries", value=False),
                        style="display: flex; justify-content: center; gap: 2rem; padding: 0.75rem; border-top: 1px solid #e2e8f0; background-color: #f8fafc;"
                    ),
                    class_="card mb-3"
                )
            ),
            
            # Bottom row: Clean HTML table comparison
            ui.div(
                ui.div(
                    ui.h6("Model Parameter Fitting Comparison Table", class_="card-header"),
                    ui.div(
                        ui.output_ui("comparison_table"),
                        class_="card-body p-0"
                    ),
                    class_="card mb-4"
                )
            ),
            
            # Periodic peak comparison
            ui.div(
                ui.div(
                    ui.h6("Periodic Peak Comparison (Simulated vs Fitted)", class_="card-header"),
                    ui.div(
                        ui.output_ui("peak_table"),
                        class_="card-body p-0"
                    ),
                    class_="card mb-4"
                )
            ),
            
            ui.div(
                ui.hr(),
                ui.p("BAPS annual meeting 2026 | NEUROPHYSIOLOGY AND BRAIN MEASUREMENT METHODS session", class_="text-center text-muted small"),
                class_="mt-4"
            ),
            id="results-panel",
            class_="tab-pane fade mobile-tab-pane col-md-8 col-lg-9 ps-md-4",
            role="tabpanel",
            aria_labelledby="results-tab"
        ),
        class_="tab-content mobile-tab-content pt-2"
    ),
    title="SpecParam Knee Simulation & Fitting Dashboard",
    theme=shinyswatch.theme.minty()
)

def server(input, output, session):
    
    # Server-side reactive renderers for conditional UI
    @output
    @render.ui
    def butterworth_filter_ui():
        if not input.apply_filter():
            return None
            
        filter_type = input.filter_type() if "filter_type" in input else "bandpass"
        elements = [
            ui.input_radio_buttons(
                "filter_type",
                "Filter Type",
                {"lowpass": "Lowpass", "highpass": "Highpass", "bandpass": "Bandpass"},
                selected=filter_type
            ),
            ui.input_slider("filter_order", "Filter Order (n)", min=1, max=10, value=input.filter_order() if "filter_order" in input else 7, step=1)
        ]
        
        if filter_type == "lowpass":
            elements.append(
                ui.input_slider("cutoff_lp", "Lowpass Cutoff (Hz)", min=1, max=100, value=input.cutoff_lp() if "cutoff_lp" in input else 8, step=1)
            )
        elif filter_type == "highpass":
            elements.append(
                ui.input_slider("cutoff_hp", "Highpass Cutoff (Hz)", min=1, max=100, value=input.cutoff_hp() if "cutoff_hp" in input else 33, step=1)
            )
        elif filter_type == "bandpass":
            use_fitting = input.use_fitting_bounds() if "use_fitting_bounds" in input else False
            elements.append(
                ui.input_checkbox("use_fitting_bounds", "Use same interval as fitting boundaries", value=use_fitting)
            )
            if not use_fitting:
                elements.append(
                    ui.input_slider("cutoff_bp", "Bandpass Cutoffs (Hz)", min=1, max=100, value=list(input.cutoff_bp()) if "cutoff_bp" in input else [33, 80], step=1)
                )
            else:
                elements.append(
                    ui.p(f"Filter frequencies match fitting boundaries: [{input.min_f()} Hz, {input.max_f()} Hz]", class_="text-muted small mt-2")
                )
            
        return ui.TagList(*elements)

    # 1. Prevent overlap between min_f and max_f by updating slider limits dynamically
    @reactive.Effect
    def _():
        min_f = input.min_f()
        ui.update_slider("max_f", min=min_f + 1)
            
    @reactive.Effect
    def _():
        max_f = input.max_f()
        ui.update_slider("min_f", max=max_f - 1)

    # 2. Synchronize Knee Parameter (k) and Knee Frequency (fk) reactively
    @reactive.Effect
    @reactive.event(input.fk)
    def _():
        try:
            fk = input.fk()
            chi = input.chi()
            if fk is None or chi is None:
                return
            k_calc = fk ** chi
            k_calc = min(1000.0, max(0.1, k_calc))
            current_k = input.k()
            if current_k is not None and abs(current_k - k_calc) > 0.2:
                ui.update_slider("k", value=round(k_calc, 1))
        except Exception:
            pass

    @reactive.Effect
    @reactive.event(input.k)
    def _():
        try:
            k = input.k()
            chi = input.chi()
            if k is None or chi is None:
                return
            fk_calc = k ** (1 / chi)
            fk_calc = min(100.0, max(1.0, fk_calc))
            current_fk = input.fk()
            if current_fk is not None and abs(current_fk - fk_calc) > 0.2:
                ui.update_slider("fk", value=round(fk_calc, 1))
        except Exception:
            pass

    @reactive.Effect
    @reactive.event(input.chi)
    def _():
        try:
            fk = input.fk()
            chi = input.chi()
            if fk is None or chi is None:
                return
            k_calc = fk ** chi
            k_calc = min(1000.0, max(0.1, k_calc))
            current_k = input.k()
            if current_k is not None and abs(current_k - k_calc) > 0.2:
                ui.update_slider("k", value=round(k_calc, 1))
        except Exception:
            pass

    # 3. Reactive simulation and fit runner
    @reactive.calc
    def run_simulation_and_fit():
        # 1. Fetch user inputs from the UI panel
        model_type = input.model_type()
        b_val = input.b()
        chi_val = input.chi()
        
        add_peak_val = input.add_peak()
        peak_freq_val = input.peak_freq() if add_peak_val else 10.0
        peak_amp_val = input.peak_amp() if add_peak_val else 0.35
        peak_width_val = input.peak_width() if add_peak_val else 1.8
        
        noise_level_val = input.noise_level()
        min_f_val = input.min_f()
        max_f_val = input.max_f()
        
        # Specparam algorithm settings
        max_n_peaks_val = input.max_n_peaks()
        peak_threshold_val = input.peak_threshold()
        min_peak_height_val = input.min_peak_height()
        peak_width_limits_val = list(input.peak_width_limits())
        gauss_overlap_thresh_val = input.gauss_overlap_thresh()
        
        # Extract Butterworth filter settings safely
        apply_filter_val = input.apply_filter()
        filter_type_val = "bandpass"
        filter_order_val = 7
        cutoff_lp_val = 8
        cutoff_hp_val = 33
        cutoff_bp_val = [33, 80]
        
        if apply_filter_val:
            try:
                filter_type_val = input.filter_type() or "bandpass"
            except Exception:
                pass
            try:
                filter_order_val = input.filter_order() or 7
            except Exception:
                pass
            try:
                cutoff_lp_val = input.cutoff_lp() or 8
            except Exception:
                pass
            try:
                cutoff_hp_val = input.cutoff_hp() or 33
            except Exception:
                pass
            try:
                if "use_fitting_bounds" in input and input.use_fitting_bounds():
                    cutoff_bp_val = [min_f_val, max_f_val]
                else:
                    cutoff_bp_val = input.cutoff_bp() or [33, 80]
            except Exception:
                pass
        
        if model_type == "knee":
            fk_val = input.fk() or 10.0
            k_val = input.k() or (fk_val ** chi_val)
        else:
            fk_val, k_val = 0.0, 0.0
            
        # 2. Run simulation
        sim_freqs = np.linspace(1, 100, 200)
        sim_power, true_power, true_aperiodic = simulate_power_spectrum(
            sim_freqs, model_type, b_val, chi_val, k_val,
            add_peak_val, peak_freq_val, peak_amp_val, peak_width_val, noise_level_val
        )
        
        # Apply Butterworth Filter to the analytical PSD
        unfiltered_power = None
        if apply_filter_val:
            unfiltered_power = sim_power.copy()
            if filter_type_val == "lowpass":
                fc = cutoff_lp_val
                filt_resp = 1.0 / (1.0 + (sim_freqs / fc) ** (2 * filter_order_val))
            elif filter_type_val == "highpass":
                fc = cutoff_hp_val
                filt_resp = 1.0 / (1.0 + (fc / sim_freqs) ** (2 * filter_order_val))
            else:  # bandpass
                f_low, f_high = cutoff_bp_val[0], cutoff_bp_val[1]
                filt_resp = (1.0 / (1.0 + (f_low / sim_freqs) ** (2 * filter_order_val))) * \
                            (1.0 / (1.0 + (sim_freqs / f_high) ** (2 * filter_order_val)))
                            
            # Multiply linear power by |H(f)|^2 and convert back to log10 space
            sim_power = np.log10(np.maximum(1e-10, (10**sim_power) * filt_resp))
            true_power = np.log10(np.maximum(1e-10, (10**true_power) * filt_resp))
            true_aperiodic = np.log10(np.maximum(1e-10, (10**true_aperiodic) * filt_resp))
        
        # 3. Fit both spectral model variants (Knee vs. Fixed)
        fm_k = fit_spectral_model(
            sim_freqs, sim_power, 'knee', [min_f_val, max_f_val],
            max_n_peaks_val, peak_threshold_val, min_peak_height_val,
            peak_width_limits_val, gauss_overlap_thresh_val
        )
        fm_f = fit_spectral_model(
            sim_freqs, sim_power, 'fixed', [min_f_val, max_f_val],
            max_n_peaks_val, peak_threshold_val, min_peak_height_val,
            peak_width_limits_val, gauss_overlap_thresh_val
        )
        
        # 4. Extract fit coordinates and model metrics
        mask = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        fit_freqs = sim_freqs[mask]
        
        k_res, fit_k_offset, fit_k_knee, fit_k_exponent, r2_k, mae_k, k_peaks = extract_fit_results(
            fm_k, fit_freqs, mask, sim_power
        )
        f_res, fit_f_offset, _, fit_f_exponent, r2_f, mae_f, f_peaks = extract_fit_results(
            fm_f, fit_freqs, mask, sim_power
        )
        
        # True knee frequency calculation (for simulated ground truth)
        true_fk = fk_val if model_type == "knee" else 0.0
        
        # Fitted knee frequency calculation
        fit_k_fk = fit_k_knee**(1 / fit_k_exponent) if (fit_k_knee > 0 and fit_k_exponent > 0) else 0.0
        
        return {
            "sim_freqs": sim_freqs,
            "sim_power": sim_power,
            "unfiltered_power": unfiltered_power,
            "true_power": true_power,
            "true_aperiodic": true_aperiodic,
            "fit_freqs": fit_freqs,
            "k_res": k_res,
            "f_res": f_res,
            "true_k": k_val,
            "true_fk": true_fk,
            "fit_k_offset": fit_k_offset,
            "fit_k_knee": fit_k_knee,
            "fit_k_exponent": fit_k_exponent,
            "fit_k_fk": fit_k_fk,
            "fit_f_offset": fit_f_offset,
            "fit_f_exponent": fit_f_exponent,
            "r2_k": r2_k,
            "r2_f": r2_f,
            "mae_k": mae_k,
            "mae_f": mae_f,
            "k_peaks": k_peaks,
            "f_peaks": f_peaks
        }

    # Helper function to plot clean continuous curves
    def build_plot():
        res = run_simulation_and_fit()
        sim_freqs = res["sim_freqs"]
        sim_power = res["sim_power"]
        true_power = res["true_power"]
        true_aperiodic = res["true_aperiodic"]
        fit_freqs = res["fit_freqs"]
        k_res = res["k_res"]
        f_res = res["f_res"]
        
        model_type = input.model_type()
        add_peak = input.add_peak()
        chi_val = input.chi()
        b_val = input.b()
        
        min_f_val = input.min_f()
        max_f_val = input.max_f()
        
        # Calculate dynamic y limits depending on the maximum values in the fitting range
        mask_in = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        
        all_y_fit = [sim_power[mask_in], true_aperiodic[mask_in]]
        if input.apply_filter() and res.get("unfiltered_power") is not None:
            all_y_fit.append(res["unfiltered_power"][mask_in])
        if add_peak:
            all_y_fit.append(true_power[mask_in])
        if len(k_res) > 0:
            all_y_fit.append(k_res)
        if len(f_res) > 0:
            all_y_fit.append(f_res)
            
        plot_max = max(np.max(y) for y in all_y_fit)
        plot_min = min(np.min(y) for y in all_y_fit)
        
        ymin = plot_min - 0.25
        ymax = plot_max + 0.35
        
        # Academic light-theme figure (standard academic 8:5.2 ratio)
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100, facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        
        # Clean academic axes lines (remove top/right borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Set tick color and font sizes
        ax.tick_params(colors='#475569', width=1.0, labelsize=9)
        
        # Highlight active fitting bounds
        mask_in = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        
        # Semilog plot
        if input.apply_filter() and res.get("unfiltered_power") is not None:
            unfilt = res["unfiltered_power"]
            ax.plot(sim_freqs, unfilt, color='#94a3b8', linewidth=1.0, alpha=0.6, label='Unfiltered Spectrum')
            ax.plot(sim_freqs, sim_power, color='#ef4444', linewidth=1.0, alpha=0.55, label='Filtered Spectrum (outside fit)')
            ax.plot(sim_freqs[mask_in], sim_power[mask_in], color='#ef4444', linewidth=1.2, alpha=0.8, label='Filtered Spectrum (inside fit)')
        else:
            ax.plot(sim_freqs, sim_power, color='#cbd5e1', linewidth=1.0, alpha=0.7, label='Raw Spectrum (outside fit)')
            ax.plot(sim_freqs[mask_in], sim_power[mask_in], color='#94a3b8', linewidth=1.2, alpha=0.9, label='Raw Spectrum (inside fit)')
        
        ax.plot(sim_freqs, true_aperiodic, color='#475569', linestyle=':', linewidth=1.2, alpha=0.8, label='True Aperiodic')
        if add_peak:
            ax.plot(sim_freqs, true_power, color='#475569', linestyle='-', linewidth=1.2, alpha=0.8, label='True Power Spectrum')
        
        if len(k_res) > 0:
            ax.plot(fit_freqs, k_res, color='#0f766e', linewidth=2.0, label='Knee Model Fit')
        if len(f_res) > 0:
            ax.plot(fit_freqs, f_res, color='#ea580c', linestyle='--', linewidth=2.0, label='Fixed Model Fit')
            
        # Vertical True Knee line (only if simulated signal has a knee)
        if model_type == "knee":
            true_fk = res["true_fk"]
            y_true_fk = b_val - np.log10(res["true_k"] + true_fk**chi_val)
            
            # Apply filter response at true_fk if enabled
            if input.apply_filter():
                try:
                    filt_type = input.filter_type() or "bandpass"
                    filt_order = input.filter_order() or 7
                    if filt_type == "lowpass":
                        fc = input.cutoff_lp() or 8
                        fk_resp = 1.0 / (1.0 + (true_fk / fc) ** (2 * filt_order))
                    elif filt_type == "highpass":
                        fc = input.cutoff_hp() or 33
                        fk_resp = 1.0 / (1.0 + (fc / true_fk) ** (2 * filt_order))
                    else:
                        if "use_fitting_bounds" in input and input.use_fitting_bounds():
                            f_low, f_high = input.min_f(), input.max_f()
                        else:
                            c_bp = input.cutoff_bp() or [33, 80]
                            f_low, f_high = c_bp[0], c_bp[1]
                        fk_resp = (1.0 / (1.0 + (f_low / true_fk) ** (2 * filt_order))) * \
                                  (1.0 / (1.0 + (true_fk / f_high) ** (2 * filt_order)))
                    y_true_fk = y_true_fk + np.log10(np.maximum(1e-10, fk_resp))
                except Exception:
                    pass
                    
            ax.vlines(true_fk, ymin, y_true_fk, colors='#e11d48', linestyles='--', alpha=0.75, label=f'True Knee ({true_fk:.2f} Hz)')
        
        # Fitted Knee Line
        if len(k_res) > 0 and res["fit_k_fk"] > 0:
            fit_fk = res["fit_k_fk"]
            if min_f_val <= fit_fk <= max_f_val:
                fit_k_offset = res["fit_k_offset"]
                fit_k_knee = res["fit_k_knee"]
                fit_k_exponent = res["fit_k_exponent"]
                y_fit_fk = fit_k_offset - np.log10(fit_k_knee + fit_fk**fit_k_exponent)
                ax.vlines(fit_fk, ymin, y_fit_fk, colors='#0f766e', linestyles=':', alpha=0.8, label=f'Fitted Knee ({fit_fk:.2f} Hz)')
                    
        if "limit_fit_bounds" in input and input.limit_fit_bounds():
            xlim_low, xlim_high = min_f_val, max_f_val
        else:
            xlim_low, xlim_high = 1, 100

        if input.x_log():
            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz, log scale)', color='#2c3e50', fontsize=10, fontweight='medium')
            
            # Filter log ticks to only those within current display limit
            ticks = [t for t in [1, 2, 5, 10, 20, 50, 100] if xlim_low <= t <= xlim_high]
            if len(ticks) >= 2:
                ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            ax.set_xscale('linear')
            ax.set_xlabel('Frequency (Hz, linear scale)', color='#2c3e50', fontsize=10, fontweight='medium')
            
        ax.set_ylabel('log10(Power)', color='#2c3e50', fontsize=10, fontweight='medium')
        ax.set_xlim(xlim_low, xlim_high)
        ax.set_ylim(ymin, ymax)

        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e2e8f0', alpha=0.7)
        plt.tight_layout()
        return fig

    # 4. Output: Semilog Plot
    @output
    @render.plot
    def plot_semilog():
        return build_plot()

    # 5. Output: Dynamic Plot Title
    @output
    @render.text
    def plot_title():
        if input.x_log():
            return "Log-Log Spectral Representation"
        else:
            return "Semilog Spectral Representation"

    # 6. Output: Structured Sphinx HTML comparison table
    @output
    @render.ui
    def comparison_table():
        res = run_simulation_and_fit()
        model_type = input.model_type()
        
        b_val = input.b()
        chi_val = input.chi()

        # Ground Truth formats
        if model_type == "knee":
            k_val_str = f"{res['true_k']:.1f}"
            fk_val_str = f"{res['true_fk']:.2f} Hz"
        else:
            k_val_str = "0.0 (Fixed 1/f)"
            fk_val_str = "N/A"
            
        # Model Fit values
        k_fit_b = f"{res['fit_k_offset']:.2f}"
        k_fit_chi = f"{res['fit_k_exponent']:.2f}"
        k_fit_k = f"{res['fit_k_knee']:.1f}"
        
        if res['fit_k_fk'] > 0:
            k_fit_fk = f"{res['fit_k_fk']:.2f} Hz"
        else:
            k_fit_fk = "N/A"
            
        f_fit_b = f"{res['fit_f_offset']:.2f}"
        f_fit_chi = f"{res['fit_f_exponent']:.2f}"
        
        # Error indicators
        def format_err(fit, true):
            err = fit - true
            return f"+{err:.2f}" if err >= 0 else f"{err:.2f}"
            
        k_err_b = format_err(res['fit_k_offset'], b_val)
        k_err_chi = format_err(res['fit_k_exponent'], chi_val)
        k_err_k = format_err(res['fit_k_knee'], res['true_k']) if model_type == "knee" else "N/A"
        
        f_err_b = format_err(res['fit_f_offset'], b_val)
        f_err_chi = format_err(res['fit_f_exponent'], chi_val)
        
        return ui.HTML(f"""
            <div class="table-responsive">
                <table class="table table-hover table-bordered mb-0" style="font-size: 0.9rem; border-color: #e2e8f0;">
                    <thead>
                        <tr>
                            <th scope="col" style="font-weight: 600; width: 25%;">Parameter</th>
                            <th scope="col" style="font-weight: 600; width: 25%;">Ground Truth</th>
                            <th scope="col" style="font-weight: 600; width: 25%;">Knee Model Fit (Error)</th>
                            <th scope="col" style="font-weight: 600; width: 25%;">Fixed Model Fit (Error)</th>
                        </tr>
                    </thead>
                    <tbody style="color: #334155; background-color: #ffffff;">
                        <tr>
                            <td style="font-weight: 500;">Offset (b)</td>
                            <td>{b_val:.2f}</td>
                            <td>{k_fit_b} <span class="text-muted small">({k_err_b})</span></td>
                            <td>{f_fit_b} <span class="text-muted small">({f_err_b})</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Exponent (exponent)</td>
                            <td>{chi_val:.2f}</td>
                            <td>{k_fit_chi} <span class="text-muted small">({k_err_chi})</span></td>
                            <td>{f_fit_chi} <span class="text-muted small">({f_err_chi})</span></td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Knee Parameter (k)</td>
                            <td>{k_val_str}</td>
                            <td>{k_fit_k} <span class="text-muted small">({k_err_k})</span></td>
                            <td class="bg-light text-muted">N/A (k=0)</td>
                        </tr>
                        <tr>
                            <td style="font-weight: 500;">Knee Frequency (fk)</td>
                            <td>{fk_val_str}</td>
                            <td>{k_fit_fk}</td>
                            <td class="bg-light text-muted">N/A</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600;">Goodness of Fit (R²)</td>
                            <td>—</td>
                            <td style="font-weight: 600; color: #0f766e;">{res['r2_k']:.3f}</td>
                            <td style="font-weight: 600; color: #ea580c;">{res['r2_f']:.3f}</td>
                        </tr>
                        <tr style="background-color: #f8fafc;">
                            <td style="font-weight: 600;">Mean Absolute Error (MAE)</td>
                            <td>—</td>
                            <td style="font-weight: 600; color: #0f766e;">{res['mae_k']:.3f}</td>
                            <td style="font-weight: 600; color: #ea580c;">{res['mae_f']:.3f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """)

    # 7. Output: Periodic peak comparison (simulated vs fitted)
    @output
    @render.ui
    def peak_table():
        res = run_simulation_and_fit()
        add_peak = input.add_peak()

        def peak_rows(label, color, peaks):
            if not peaks:
                return f"""
                    <tr>
                        <td style="font-weight: 500; color: {color};">{label}</td>
                        <td colspan="3" class="text-muted">No peak estimated</td>
                    </tr>"""
            rows = []
            for idx, (cf, pw, bw) in enumerate(peaks):
                row_label = f"{label} (Peak {idx+1})" if len(peaks) > 1 else label
                rows.append(f"""
                    <tr>
                        <td style="font-weight: 500; color: {color};">{row_label}</td>
                        <td>{cf:.2f}</td>
                        <td>{pw:.3f}</td>
                        <td>{bw:.2f}</td>
                    </tr>""")
            return "\n".join(rows)

        # Simulated peak: convert injected Gaussian to specparam-equivalent
        # PW (height above aperiodic) and BW (2 * std). Sim uses scale = width/2.
        if add_peak:
            cf_t = input.peak_freq()
            bw_t = input.peak_width()
            std_t = bw_t / 2.0
            pw_t = input.peak_amp() / (std_t * np.sqrt(2 * np.pi))
            sim_row = f"""
                    <tr style="background-color: #f8fafc;">
                        <td style="font-weight: 600;">Simulated (ground truth)</td>
                        <td>{cf_t:.2f}</td>
                        <td>{pw_t:.3f}</td>
                        <td>{bw_t:.2f}</td>
                    </tr>"""
        else:
            sim_row = """
                    <tr style="background-color: #f8fafc;">
                        <td style="font-weight: 600;">Simulated (ground truth)</td>
                        <td colspan="3" class="text-muted">No peak simulated</td>
                    </tr>"""

        return ui.HTML(f"""
            <div class="table-responsive">
                <table class="table table-hover table-bordered mb-0" style="font-size: 0.9rem; border-color: #e2e8f0;">
                    <thead>
                        <tr>
                            <th scope="col" style="font-weight: 600; width: 40%;">Source</th>
                            <th scope="col" style="font-weight: 600; width: 20%;">Center Freq (Hz)</th>
                            <th scope="col" style="font-weight: 600; width: 20%;">Power (PW)</th>
                            <th scope="col" style="font-weight: 600; width: 20%;">Bandwidth (Hz)</th>
                        </tr>
                    </thead>
                    <tbody style="color: #334155; background-color: #ffffff;">
                        {sim_row}
                        {peak_rows("Knee Model — estimated", "#0f766e", res['k_peaks'])}
                        {peak_rows("Fixed Model — estimated", "#ea580c", res['f_peaks'])}
                    </tbody>
                </table>
            </div>
        """)

app = App(app_ui, server)
