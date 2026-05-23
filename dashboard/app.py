import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from specparam import SpectralModel
from shiny import App, ui, render, reactive
import shinyswatch

# Define Sphinx-like document layout aligned with specparam-tools.github.io
app_ui = ui.page_fluid(
    # Custom CSS head content for PyData/Sphinx documentation aesthetic
    ui.head_content(
        # Load MathJax CDN for LaTeX mathematical rendering
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
        ui.tags.style("""
            /* Sphinx / PyData Documentation Theme Styles */
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                background-color: #ffffff !important;
                color: #2c3e50 !important;
            }
            .container-fluid {
                padding-left: 1.0rem !important;
                padding-right: 1.0rem !important;
            }
            @media (min-width: 768px) {
                .container-fluid {
                    padding-left: 2.5rem !important;
                    padding-right: 2.5rem !important;
                }
            }
            #plot_semilog {
                width: 100% !important;
                height: auto !important;
                aspect-ratio: 8 / 5.2 !important;
            }
            aside.sidebar {
                background-color: #f8fafc !important;
                border-right: 1px solid #e2e8f0 !important;
            }
            .card {
                border: 1px solid #e2e8f0 !important;
                box-shadow: none !important;
                border-radius: 6px !important;
                background-color: #ffffff !important;
                margin-bottom: 1.5rem;
            }
            .card-header {
                border-bottom: 1px solid #e2e8f0 !important;
                background-color: #f8fafc !important;
                font-weight: 600 !important;
                color: #2c3e50 !important;
                font-size: 0.95rem;
            }
            .card-body {
                background-color: #ffffff !important;
            }
            
            /* Sphinx Note Admonition Directive Style */
            .sphinx-admonition {
                border-left: 4px solid #2980b9 !important;
                background-color: #ebf5fb !important;
                border-radius: 4px;
                padding: 1.25rem;
                margin-bottom: 2rem;
                border-top: 1px solid #d4e6f1;
                border-right: 1px solid #d4e6f1;
                border-bottom: 1px solid #d4e6f1;
            }
            .sphinx-admonition-title {
                font-weight: bold;
                color: #1b4f72;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                display: block;
            }
            .sphinx-admonition-body {
                color: #21618c;
                font-size: 0.875rem;
                line-height: 1.6;
            }
            
            h2, h3, h5, h6 {
                color: #1a365d !important;
            }
            hr {
                border-top: 1px solid #e2e8f0 !important;
                opacity: 1 !important;
            }
            
            /* Styled Tables matching Sphinx outputs */
            table.table {
                border-color: #e2e8f0 !important;
            }
            table.table th {
                border-bottom: 2px solid #cbd5e1 !important;
                background-color: #f8fafc !important;
                color: #1e293b !important;
            }
            table.table td {
                vertical-align: middle !important;
            }

            /* Custom mobile/desktop responsive layout for parameters & results */
            @media (min-width: 768px) {
                ul#mobileTab.mobile-tabs {
                    display: none !important;
                }
                .mobile-tab-content {
                    display: flex !important;
                }
                .mobile-tab-pane {
                    display: block !important;
                    opacity: 1 !important;
                }
                #params-panel {
                    background-color: #f8fafc !important;
                    border-right: 1px solid #e2e8f0 !important;
                    min-height: calc(100vh - 120px);
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                }
                #results-panel {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                }
            }

            @media (max-width: 767.98px) {
                .mobile-tabs {
                    display: flex !important;
                    margin-bottom: 1.5rem;
                    border-bottom: 1px solid #cbd5e1;
                }
                .mobile-tabs .nav-link {
                    color: #475569;
                    font-weight: 600;
                    border: none;
                    border-bottom: 3px solid transparent;
                    border-radius: 0;
                    padding: 0.75rem 1rem;
                }
                .mobile-tabs .nav-link.active {
                    color: #0f766e !important;
                    border-bottom-color: #0f766e !important;
                    background: transparent !important;
                }
                #params-panel, #results-panel {
                    padding-left: 0.5rem !important;
                    padding-right: 0.5rem !important;
                    border-right: none !important;
                }
                .tab-content > .mobile-tab-pane:not(.active) {
                    display: none !important;
                }
            }

            /* Plot Scrolling & Min-Width Styles */
            .plot-scroll-container {
                width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                border-radius: 4px;
            }
            .plot-min-width-wrapper {
                min-width: 750px;
            }
        """)
    ),
    
    # Header Area
    ui.div(
        ui.h2("SpecParam Knee Simulation & Fitting Dashboard", class_="mt-3 mb-1 font-weight-bold", style="color: #2c3e50;"),
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
                    "input.add_peak",
                    ui.input_slider("peak_freq", "Peak Frequency (Hz)", min=2, max=50, value=10, step=1),
                    ui.input_slider("peak_amp", "Peak Amplitude (log units)", min=0.05, max=14.5, value=0.35, step=0.05),
                    ui.input_slider("peak_width", "Peak Bandwidth (Hz)", min=0.5, max=5.0, value=1.8, step=0.1)
                ),
                
                ui.h5("Noise Configuration", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("noise_level", "Spectral Noise Level (SD)", min=0.0, max=1.6, value=0.04, step=0.01),
                
                ui.h5("Fitting Boundaries", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("min_f", "Min Fit Frequency (Hz)", min=1, max=50, value=1, step=1),
                ui.input_slider("max_f", "Max Fit Frequency (Hz)", min=50, max=100, value=100, step=1),
                
                ui.h5("specparam Fit Settings", class_="mt-4 mb-3 border-bottom pb-2 font-weight-bold", style="color: #2c3e50; font-size: 1.05rem;"),
                ui.input_slider("max_n_peaks", "Max Number of Peaks", min=0, max=5, value=2, step=1),
                ui.input_slider("peak_threshold", "Peak Threshold (SD)", min=1.0, max=5.0, value=2.0, step=0.1),
                ui.input_slider("min_peak_height", "Min Peak Height (log units)", min=0.0, max=1.0, value=0.0, step=0.05),
                ui.input_slider("peak_width_limits", "Peak Bandwidth Limits (Hz)", min=0.5, max=15.0, value=[0.5, 12.0], step=0.5),
                ui.input_slider("gauss_overlap_thresh", "Overlap Threshold (SD)", min=0.0, max=2.0, value=0.75, step=0.05),
                class_="p-3 rounded",
                style="background-color: #f8fafc; border: 1px solid #e2e8f0;"
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
                    ui.h6("Semilog Representation (Physical Hz)", class_="card-header bg-transparent text-center font-weight-bold text-secondary"),
                    ui.div(
                        ui.div(
                            ui.output_plot("plot_semilog", height="480px"),
                            class_="plot-min-width-wrapper"
                        ),
                        class_="plot-scroll-container"
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
    
    # 1. Prevent overlap between min_f and max_f
    @reactive.Effect
    def _():
        min_f = input.min_f()
        max_f = input.max_f()
        if min_f >= max_f - 4:
            ui.update_slider("min_f", value=max_f - 5)
            
    @reactive.Effect
    def _():
        min_f = input.min_f()
        max_f = input.max_f()
        if max_f <= min_f + 4:
            ui.update_slider("max_f", value=min_f + 5)

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
        # Get inputs
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
        
        # specparam fitting parameters
        max_n_peaks_val = input.max_n_peaks()
        peak_threshold_val = input.peak_threshold()
        min_peak_height_val = input.min_peak_height()
        peak_width_limits_val = list(input.peak_width_limits())
        gauss_overlap_thresh_val = input.gauss_overlap_thresh()
        
        if model_type == "knee":
            fk_val = input.fk()
            k_val = input.k()
            if fk_val is None:
                fk_val = 10.0
            if k_val is None:
                k_val = fk_val ** chi_val
        else:
            fk_val = 0.0
            k_val = 0.0
            
        # Generate raw clean aperiodic spectrum
        sim_freqs = np.linspace(1, 100, 200)
        if model_type == "knee":
            true_aperiodic = b_val - np.log10(k_val + sim_freqs**chi_val)
        else:
            true_aperiodic = b_val - np.log10(sim_freqs**chi_val)
            
        true_power = true_aperiodic.copy()
        if add_peak_val:
            # Inject realistic peak (Gaussian shape centered on custom parameters)
            # Standard deviation scaled to width/2.0 to match peak bandwidth definition
            peak_signal = peak_amp_val * norm.pdf(sim_freqs, loc=peak_freq_val, scale=peak_width_val / 2.0)
            true_power += peak_signal
            
        # Add realistic experimental noise for fitting
        np.random.seed(42)
        noise = np.random.normal(0, noise_level_val, len(sim_freqs)) if noise_level_val > 0 else np.zeros(len(sim_freqs))
        sim_power = true_power + noise
        
        # Fit Knee model on the noisy signal
        fm_k = SpectralModel(
            aperiodic_mode='knee',
            max_n_peaks=max_n_peaks_val,
            peak_threshold=peak_threshold_val,
            min_peak_height=min_peak_height_val,
            peak_width_limits=peak_width_limits_val,
            gauss_overlap_thresh=gauss_overlap_thresh_val,
            verbose=False
        )
        fm_k.fit(sim_freqs, 10**sim_power, [min_f_val, max_f_val])
        
        # Fit Fixed model on the noisy signal
        fm_f = SpectralModel(
            aperiodic_mode='fixed',
            max_n_peaks=max_n_peaks_val,
            peak_threshold=peak_threshold_val,
            min_peak_height=min_peak_height_val,
            peak_width_limits=peak_width_limits_val,
            gauss_overlap_thresh=gauss_overlap_thresh_val,
            verbose=False
        )
        fm_f.fit(sim_freqs, 10**sim_power, [min_f_val, max_f_val])
        
        # Get fit results coordinates over the fitting range
        mask = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        fit_freqs = sim_freqs[mask]
        
        k_res = np.array([])
        f_res = np.array([])
        
        if fm_k.results.model:
            k_off = fm_k.get_params('aperiodic', 'offset')
            k_kn = fm_k.get_params('aperiodic', 'knee')
            k_exp = fm_k.get_params('aperiodic', 'exponent')
            # Avoid log of zero/negative
            k_res = k_off - np.log10(max(1e-5, k_kn) + fit_freqs**k_exp)
            
        if fm_f.results.model:
            f_off = fm_f.get_params('aperiodic', 'offset')
            f_exp = fm_f.get_params('aperiodic', 'exponent')
            f_res = f_off - np.log10(fit_freqs**f_exp)
        
        # Extracts params safely
        fit_k_offset = float(fm_k.get_params('aperiodic', 'offset')) if fm_k.results.model else 0.0
        fit_k_knee = float(fm_k.get_params('aperiodic', 'knee')) if fm_k.results.model else 0.0
        fit_k_exponent = float(fm_k.get_params('aperiodic', 'exponent')) if fm_k.results.model else 0.0
        
        fit_f_offset = float(fm_f.get_params('aperiodic', 'offset')) if fm_f.results.model else 0.0
        fit_f_exponent = float(fm_f.get_params('aperiodic', 'exponent')) if fm_f.results.model else 0.0
        
        r2_k = float(fm_k.get_metrics('gof_rsquared')) if fm_k.results.model else 0.0
        r2_f = float(fm_f.get_metrics('gof_rsquared')) if fm_f.results.model else 0.0
        
        # Calculate Mean Absolute Error (MAE)
        mae_k = float(np.mean(np.abs(sim_power[mask] - k_res))) if len(k_res) > 0 else 0.0
        mae_f = float(np.mean(np.abs(sim_power[mask] - f_res))) if len(f_res) > 0 else 0.0
        
        # Calculate true knee parameter metrics for knee model
        if model_type == "knee":
            true_fk = fk_val
        else:
            true_fk = 0.0
            
        # Calculate fitted knee frequency
        if fit_k_knee > 0 and fit_k_exponent > 0:
            fit_k_fk = fit_k_knee**(1 / fit_k_exponent)
        else:
            fit_k_fk = 0.0

        # Extract estimated periodic peaks (CF, PW, BW rows) from each fit
        def extract_peaks(fm):
            if not fm.results.model:
                return []
            try:
                pk = np.atleast_2d(np.asarray(fm.get_params('peak'), dtype=float))
            except Exception:
                return []
            return [
                (float(r[0]), float(r[1]), float(r[2]))
                for r in pk
                if r.size >= 3 and not np.any(np.isnan(r[:3]))
            ]

        k_peaks = extract_peaks(fm_k)
        f_peaks = extract_peaks(fm_f)

        return {
            "sim_freqs": sim_freqs,
            "sim_power": sim_power,
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
        ax.plot(sim_freqs, sim_power, color='#cbd5e1', linewidth=1.0, alpha=0.7, label='Raw Spectrum (outside fit)')
        ax.plot(sim_freqs[mask_in], sim_power[mask_in], color='#94a3b8', linewidth=1.2, alpha=0.9, label='Raw Spectrum (inside fit)')
        
        ax.plot(sim_freqs, true_aperiodic, color='#475569', linestyle=':', linewidth=1.2, alpha=0.8, label='True Aperiodic')
        if add_peak:
            ax.plot(sim_freqs, true_power, color='#475569', linestyle='-', linewidth=1.2, alpha=0.8, label='True Power Spectrum')
        
        if len(k_res) > 0:
            ax.plot(fit_freqs, k_res, color='#0f766e', linewidth=2.5, label='Knee Model Fit')
        if len(f_res) > 0:
            ax.plot(fit_freqs, f_res, color='#ea580c', linestyle='--', linewidth=2.0, label='Fixed Model Fit')
            
        # Vertical True Knee line (only if simulated signal has a knee)
        if model_type == "knee":
            true_fk = res["true_fk"]
            y_true_fk = b_val - np.log10(res["true_k"] + true_fk**chi_val)
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
                    
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz, log scale)', color='#2c3e50', fontsize=10, fontweight='medium')
        ax.set_ylabel('log10(Power)', color='#2c3e50', fontsize=10, fontweight='medium')
        ax.set_xlim(1, 100)
        ax.set_ylim(ymin, ymax)
        
        # Format x-axis nicely for log ticks
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95, facecolor='#ffffff', edgecolor='#e2e8f0')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e2e8f0', alpha=0.7)
        plt.tight_layout()
        return fig

    # 4. Output: Semilog Plot
    @output
    @render.plot
    def plot_semilog():
        return build_plot()

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
