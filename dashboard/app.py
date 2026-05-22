import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from specparam import SpectralModel
from shiny import App, ui, render, reactive
import shinyswatch

# Define modern, responsive Bootstrap page_sidebar layout
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("🔧 Ground-Truth Parameters", class_="mb-3 border-bottom pb-1 text-primary"),
        ui.input_slider("b", "Offset (b)", min=0.5, max=5.0, value=3.0, step=0.1),
        ui.input_slider("chi", "Exponent (\u03c7)", min=0.5, max=4.0, value=2.0, step=0.1),
        ui.input_slider("k", "Knee Parameter (k)", min=0.1, max=100.0, value=15.0, step=0.5),
        ui.p("Transitions knee plateau (restricted dynamically: fk \u2264 100 Hz)", class_="text-muted small mb-4"),
        
        ui.h5("🎯 Fitting Boundaries", class_="mb-3 border-bottom pb-1 text-primary"),
        ui.input_slider("min_f", "Min Fit Frequency (Hz)", min=1, max=50, value=1, step=1),
        ui.input_slider("max_f", "Max Fit Frequency (Hz)", min=50, max=100, value=100, step=1),
        ui.p("Adjust bounds to see how the Fixed model is biased by low frequencies, while the Knee model recovers true parameters.", class_="text-muted small"),
        width=320
    ),
    
    # Main panel content
    ui.div(
        ui.h2("🧠 SpecParam Knee Simulation & Fitting Dashboard", class_="text-center mt-2 mb-1 font-weight-bold"),
        ui.p("Explore aperiodic power spectra models and fitting bias in real-time with Python WebAssembly.", class_="text-center text-muted mb-4"),
        class_="container-fluid"
    ),
    
    # Top row: Quick Stat Cards
    ui.row(
        ui.column(
            4,
            ui.div(
                ui.div(
                    ui.h6("True Knee Frequency (fk)", class_="card-title text-uppercase text-muted mb-1 small"),
                    ui.h3(ui.output_text("val_fk"), class_="card-text font-weight-bold text-teal"),
                    class_="card-body p-3"
                ),
                class_="card shadow-sm border-light mb-3"
            )
        ),
        ui.column(
            4,
            ui.div(
                ui.div(
                    ui.h6("True Time Constant (\u03c4)", class_="card-title text-uppercase text-muted mb-1 small"),
                    ui.h3(ui.output_text("val_tau"), class_="card-text font-weight-bold text-teal"),
                    class_="card-body p-3"
                ),
                class_="card shadow-sm border-light mb-3"
            )
        ),
        ui.column(
            4,
            ui.div(
                ui.div(
                    ui.h6("True Plateau Height", class_="card-title text-uppercase text-muted mb-1 small"),
                    ui.h3(ui.output_text("val_plateau"), class_="card-text font-weight-bold text-teal"),
                    class_="card-body p-3"
                ),
                class_="card shadow-sm border-light mb-3"
            )
        )
    ),
    
    # Middle row: Plots side-by-side
    ui.row(
        ui.column(
            6,
            ui.div(
                ui.div(
                    ui.h6("Log-Log Representation", class_="card-header bg-transparent text-center font-weight-bold text-secondary"),
                    ui.output_plot("plot_loglog", height="350px"),
                    class_="card shadow-sm border-light mb-3"
                )
            )
        ),
        ui.column(
            6,
            ui.div(
                ui.div(
                    ui.h6("Semilog Representation (Physical Hz)", class_="card-header bg-transparent text-center font-weight-bold text-secondary"),
                    ui.output_plot("plot_semilog", height="350px"),
                    class_="card shadow-sm border-light mb-3"
                )
            )
        )
    ),
    
    # Bottom row: Comparison table
    ui.row(
        ui.column(
            12,
            ui.div(
                ui.div(
                    ui.h6("📊 Dynamic Parameter Fitting Comparison Table (SpecParam v2 Python)", class_="card-header bg-dark text-white font-weight-bold"),
                    ui.div(
                        ui.output_text_verbatim("table_text"),
                        class_="card-body bg-black text-success p-3 m-0",
                        style="font-family: 'JetBrains Mono', monospace; white-space: pre-wrap; font-size: 0.85rem;"
                    ),
                    class_="card shadow-sm border-dark mb-4"
                        )
                    )
                ),
    
    ui.div(
        ui.hr(),
        ui.p("© 2026 Daniel Borek. PhD Research Tool.", class_="text-center text-muted small"),
        class_="container-fluid mt-4"
    ),
    
    title="🧠 SpecParam Knee Simulation & Fitting Dashboard",
    theme=shinyswatch.theme.minty()
)

def server(input, output, session):
    
    # 1. Reactive value constraint for Knee k
    @reactive.Effect
    def _():
        chi = input.chi()
        # Enforce fk = k^(1/chi) <= 100 Hz -> k <= 100^chi
        max_k = min(1000.0, 100.0**chi)
        current_k = input.k()
        
        # Update slider constraints dynamically
        ui.update_slider("k", max=max_k, value=min(current_k, max_k))
        
    # 2. Prevent overlap between min_f and max_f
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

    # 3. Reactive simulation and fit runner
    @reactive.calc
    def run_simulation_and_fit():
        # Get inputs
        b_val = input.b()
        chi_val = input.chi()
        k_val = input.k()
        min_f_val = input.min_f()
        max_f_val = input.max_f()
        
        # Clamp knee value to the exact mathematical limit
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        
        # Generate raw aperiodic spectrum
        np.random.seed(42)
        sim_freqs = np.linspace(1, 100, 200)
        true_aperiodic = b_val - np.log10(k_val + sim_freqs**chi_val)
        noise = np.random.normal(0, 0.04, len(sim_freqs))
        sim_power = true_aperiodic + noise
        
        # Inject realistic alpha peak at 10Hz
        alpha_peak = 0.3 * norm.pdf(sim_freqs, loc=10, scale=1.5)
        sim_power += alpha_peak
        
        # Fit Knee model
        fm_k = SpectralModel(aperiodic_mode='knee', verbose=False)
        fm_k.fit(sim_freqs, 10**sim_power, [min_f_val, max_f_val])
        
        # Fit Fixed model
        fm_f = SpectralModel(aperiodic_mode='fixed', max_n_peaks=0, verbose=False)
        fm_f.fit(sim_freqs, 10**sim_power, [min_f_val, max_f_val])
        
        # Get fit results coordinates
        mask = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        fit_freqs = sim_freqs[mask]
        
        # Extract individual fit line parts over the fitting range
        k_res = np.array([])
        f_res = np.array([])
        
        if fm_k.results.model:
            k_off = fm_k.get_params('aperiodic', 'offset')
            k_kn = fm_k.get_params('aperiodic', 'knee')
            k_exp = fm_k.get_params('aperiodic', 'exponent')
            k_res = k_off - np.log10(k_kn + fit_freqs**k_exp)
            
        if fm_f.results.model:
            f_off = fm_f.get_params('aperiodic', 'offset')
            f_exp = fm_f.get_params('aperiodic', 'exponent')
            f_res = f_off - np.log10(fit_freqs**f_exp)
        
        # Parameters
        fit_k_offset = fm_k.get_params('aperiodic', 'offset') if fm_k.results.model else 0.0
        fit_k_knee = fm_k.get_params('aperiodic', 'knee') if fm_k.results.model else 0.0
        fit_k_exponent = fm_k.get_params('aperiodic', 'exponent') if fm_k.results.model else 0.0
        
        fit_f_offset = fm_f.get_params('aperiodic', 'offset') if fm_f.results.model else 0.0
        fit_f_exponent = fm_f.get_params('aperiodic', 'exponent') if fm_f.results.model else 0.0
        
        r2_k = fm_k.get_metrics('gof_rsquared') if fm_k.results.model else 0.0
        r2_f = fm_f.get_metrics('gof_rsquared') if fm_f.results.model else 0.0
        
        return {
            "sim_freqs": sim_freqs,
            "true_aperiodic": true_aperiodic,
            "sim_power": sim_power,
            "fit_freqs": fit_freqs,
            "k_res": k_res,
            "f_res": f_res,
            "fit_k_offset": float(fit_k_offset),
            "fit_k_knee": float(fit_k_knee),
            "fit_k_exponent": float(fit_k_exponent),
            "fit_f_offset": float(fit_f_offset),
            "fit_f_exponent": float(fit_f_exponent),
            "r2_k": float(r2_k),
            "r2_f": float(r2_f)
        }

    # 4. Outputs: Quick Stats
    @output
    @render.text
    def val_fk():
        chi_val = input.chi()
        k_val = input.k()
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        f_k = k_val**(1 / chi_val)
        return f"{f_k:.2f} Hz"
        
    @output
    @render.text
    def val_tau():
        chi_val = input.chi()
        k_val = input.k()
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        f_k = k_val**(1 / chi_val)
        tau_ms = (1 / (2 * np.pi * f_k)) * 1000
        return f"{tau_ms:.1f} ms"
        
    @output
    @render.text
    def val_plateau():
        b_val = input.b()
        chi_val = input.chi()
        k_val = input.k()
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        plateau = b_val - np.log10(k_val)
        return f"{plateau:.2f} log unit"

    # Helper function to plot
    def build_plot(is_loglog):
        res = run_simulation_and_fit()
        sim_freqs = res["sim_freqs"]
        sim_power = res["sim_power"]
        true_aperiodic = res["true_aperiodic"]
        fit_freqs = res["fit_freqs"]
        k_res = res["k_res"]
        f_res = res["f_res"]
        
        b_val = input.b()
        chi_val = input.chi()
        k_val = input.k()
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        true_fk = k_val**(1 / chi_val)
        
        min_f_val = input.min_f()
        max_f_val = input.max_f()
        
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
        
        # Divide into inside/outside fit range
        mask_in = (sim_freqs >= min_f_val) & (sim_freqs <= max_f_val)
        mask_out = ~mask_in
        
        if is_loglog:
            x_data_in = np.log10(sim_freqs[mask_in])
            x_data_out = np.log10(sim_freqs[mask_out])
            x_true = np.log10(sim_freqs)
            x_fit = np.log10(fit_freqs)
            
            # Plot raw data points
            ax.scatter(x_data_out, sim_power[mask_out], color='#cbd5e1', s=15, alpha=0.7, label='Data (outside fit)')
            ax.scatter(x_data_in, sim_power[mask_in], color='#94a3b8', s=25, alpha=0.9, label='Data (inside fit)')
            
            # Ground truth
            ax.plot(x_true, true_aperiodic, color='#64748b', linestyle=':', linewidth=1.5, label='True Aperiodic (Ground Truth)')
            
            # Fits
            if len(k_res) > 0:
                ax.plot(x_fit, k_res, color='#0f766e', linewidth=3, label='Knee Fit')
            if len(f_res) > 0:
                ax.plot(x_fit, f_res, color='#ea580c', linestyle='--', linewidth=2.5, label='Fixed Fit')
                
            # Vertical True Knee line
            ax.axvline(np.log10(true_fk), color='#e11d48', linestyle='--', alpha=0.8, label=f'True Knee ({true_fk:.2f} Hz)')
            
            # Fitted Knee Line
            if len(k_res) > 0:
                fit_k_k = res["fit_k_knee"]
                fit_k_chi = res["fit_k_exponent"]
                if fit_k_k > 0 and fit_k_chi > 0:
                    fit_fk = fit_k_k**(1 / fit_k_chi)
                    if min_f_val <= fit_fk <= max_f_val:
                        ax.axvline(np.log10(fit_fk), color='#0f766e', linestyle=':', alpha=0.8, label=f'Fitted Knee ({fit_fk:.2f} Hz)')
                        
            ax.set_xlabel('log10(Frequency)')
            ax.set_ylabel('log10(Power)')
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 4.5)
        else:
            # Semilog plot
            ax.scatter(sim_freqs[mask_out], sim_power[mask_out], color='#cbd5e1', s=15, alpha=0.7, label='Data (outside fit)')
            ax.scatter(sim_freqs[mask_in], sim_power[mask_in], color='#94a3b8', s=25, alpha=0.9, label='Data (inside fit)')
            
            ax.plot(sim_freqs, true_aperiodic, color='#64748b', linestyle=':', linewidth=1.5, label='True Aperiodic (Ground Truth)')
            
            if len(k_res) > 0:
                ax.plot(fit_freqs, k_res, color='#0f766e', linewidth=3, label='Knee Fit')
            if len(f_res) > 0:
                ax.plot(fit_freqs, f_res, color='#ea580c', linestyle='--', linewidth=2.5, label='Fixed Fit')
                
            ax.axvline(true_fk, color='#e11d48', linestyle='--', alpha=0.8, label=f'True Knee ({true_fk:.2f} Hz)')
            
            if len(k_res) > 0:
                fit_k_k = res["fit_k_knee"]
                fit_k_chi = res["fit_k_exponent"]
                if fit_k_k > 0 and fit_k_chi > 0:
                    fit_fk = fit_k_k**(1 / fit_k_chi)
                    if min_f_val <= fit_fk <= max_f_val:
                        ax.axvline(fit_fk, color='#0f766e', linestyle=':', alpha=0.8, label=f'Fitted Knee ({fit_fk:.2f} Hz)')
                        
            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz, log scale)')
            ax.set_ylabel('log10(Power)')
            ax.set_xlim(1, 100)
            ax.set_ylim(0, 4.5)
            
            # Format x-axis nicely for log ticks
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.tight_layout()
        return fig

    # 5. Output: Log-Log Plot
    @output
    @render.plot
    def plot_loglog():
        return build_plot(is_loglog=True)

    # 6. Output: Semilog Plot
    @output
    @render.plot
    def plot_semilog():
        return build_plot(is_loglog=False)

    # 7. Output: Text comparison table
    @output
    @render.text
    def table_text():
        res = run_simulation_and_fit()
        b_val = input.b()
        chi_val = input.chi()
        k_val = input.k()
        max_k = min(1000.0, 100.0**chi_val)
        k_val = min(k_val, max_k)
        true_fk = k_val**(1 / chi_val)
        tau_ms = (1 / (2 * np.pi * true_fk)) * 1000
        
        k_err_b = res["fit_k_offset"] - b_val
        k_err_k = res["fit_k_knee"] - k_val
        k_err_chi = res["fit_k_exponent"] - chi_val
        
        f_err_b = res["fit_f_offset"] - b_val
        f_err_chi = res["fit_f_exponent"] - chi_val
        
        def format_err(val):
            return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
            
        return (
            f"【 GROUND TRUTH 】  Offset (b): {b_val:.2f} | Knee (k): {k_val:.1f} | Exponent (\u03c7): {chi_val:.2f} | fk: {true_fk:.2f} Hz | \u03c4: {tau_ms:.1f} ms\n"
            f"【 KNEE FIT 】      Offset (b): {res['fit_k_offset']:.2f} (err: {format_err(k_err_b)}) | Knee (k): {res['fit_k_knee']:.1f} (err: {format_err(k_err_k)}) | Exponent (\u03c7): {res['fit_k_exponent']:.2f} (err: {format_err(k_err_chi)}) | R\u00b2: {res['r2_k']:.3f}\n"
            f"【 FIXED FIT 】     Offset (b): {res['fit_f_offset']:.2f} (err: {format_err(f_err_b)}) | Exponent (\u03c7): {res['fit_f_exponent']:.2f} (err: {format_err(f_err_chi)}) | R\u00b2: {res['r2_f']:.3f}"
        )

app = App(app_ui, server)
