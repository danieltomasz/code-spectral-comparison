# Analysis of Butterworth Filter Bias in Armonaite et al. (2026)

This document provides a detailed summary of our analysis regarding the paper **"Scale-Free Neurodynamics as Functional Fingerprint of Brain Regions" (Armonaite et al., 2026)** and whether their preprocessing pipeline introduces a systematic bias in their Power Spectral Density (PSD) scale-free exponent ($\beta$) estimates.

---

## 1. Context and Paper Methodology

The paper investigates the aperiodic, scale-free component ($1/f^\beta$) of resting-state stereotactic intracranial EEG (sEEG) recordings across 37 brain regions as a possible functional "fingerprint."

To estimate the power-law exponent $\beta$, they processed the sEEG signals into two separate bands before applying Welch's method for PSD estimation and fitting a straight line on a double-logarithmic scale:
1. **Low-Frequency Range (0.5–4.0 Hz):** Signals were preprocessed using a **7th-order lowpass Butterworth filter with an 8 Hz cutoff**.
2. **High-Frequency Range (34–80 Hz):** Signals were preprocessed using a **7th-order highpass Butterworth filter with a 33 Hz cutoff**.

> [!NOTE]
> The authors stated that the filters were applied to attenuate alpha-band oscillations (8–12 Hz) before estimating the scale-free parameters, with the aim of reducing spectral distortions from periodic peaks.

---

## 2. Key Findings & Mathematical Proof of Bias

### 🔴 High-Frequency Band (34–80 Hz): Severe Underestimation Bias
The application of a 7th-order highpass Butterworth filter with a 33 Hz cutoff introduces a **severe systematic underestimation bias** on the high-frequency $\beta$ exponent. 

#### Mathematical Derivation:
The power transfer response of a 7th-order highpass Butterworth filter is:
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f_c}{f}\right)^{2n}} = \frac{1}{1 + \left(\frac{33}{f}\right)^{14}}$$

Because the cutoff frequency ($33\text{ Hz}$) is extremely close to the lower boundary of the fitting range ($34\text{ Hz}$), the filter's transition band heavily overlaps with the regression interval:
* **At 34 Hz:** $|H(34)|^2 \approx 0.603$ (attenuation of **$-2.2\text{ dB}$**)
* **At 80 Hz:** $|H(80)|^2 \approx 0.999996$ (practically **$0\text{ dB}$**)

In the log-log fitting domain, the filtered PSD is:
$$\ln(\text{PSD}_{\text{filtered}}(f)) = \ln(\text{PSD}_{\text{raw}}(f)) + \ln(|H(f)|^2)$$

Applying linear regression (which is a linear operator) yields:
$$\text{Slope}_{\text{filtered}} = \text{Slope}_{\text{raw}} + \text{Slope}_{\ln(|H(f)|^2)}$$

Since $\beta \equiv -\text{Slope}$, we get:
$$\beta_{\text{estimated}} = \beta_{\text{true}} - \text{Slope}_{\ln(|H(f)|^2)}$$

A least-squares linear regression on the filter response term $\ln(|H(f)|^2)$ across the 34–80 Hz interval (with 2048-point FFT resolution) yields a positive slope of approximately **$+0.216$**. Because this bias is additive in the log domain, it is **completely independent of the true value of $\beta$**.

> [!IMPORTANT]
> The Butterworth highpass filter systematically biases the high-frequency scale-free exponent by **$\approx -0.216$**, making the estimated aperiodic slope flatter than it actually is:
> $$\beta_{\text{estimated}} \approx \beta_{\text{true}} - 0.216$$

---

### 🟢 Low-Frequency Band (0.5–4.0 Hz): No Meaningful Bias
The 7th-order lowpass Butterworth filter with an 8 Hz cutoff has **virtually zero effect** on the low-frequency $\beta$ estimate.

#### Mathematical Derivation:
The power transfer response is:
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}} = \frac{1}{1 + \left(\frac{f}{8}\right)^{14}}$$

At the upper boundary of the fitting range ($f = 4\text{ Hz}$):
$$|H(4)|^2 = \frac{1}{1 + (0.5)^{14}} = \frac{1}{1 + 0.000061} \approx 0.99994 \text{ (approx. } -0.00026\text{ dB)}$$

The spectral shape in the 0.5–4.0 Hz interval is completely unaffected, resulting in **zero meaningful bias** on the low-frequency exponent.

---

## 3. Simulation Scripts Prepared

To verify these results, two Python simulation scripts have been saved in your local scratch directory:
📁 **Path:** `/Users/daniel/.gemini/antigravity/brain/c17048bb-c958-4e37-8161-112249324b24/scratch/`

### 1. Pure Python Zero-Dependency Version (`simulate_filter_bias_pure.py`)
This script uses only built-in standard library components to calculate the exact theoretical PSD response of the Butterworth filters and run log-log linear fits across multiple target $\beta$ values:

```python
import math

def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def simulate_bias_theoretical(true_beta=2.0):
    fs = 200.0
    nfft = 2048
    df = fs / nfft
    
    # High-Frequency Range (34 - 80 Hz)
    fc_hp = 33.0
    order_hp = 7
    x_hf, y_hf_raw, y_hf_filtered = [], [], []
    
    f = 0.0
    while f <= fs / 2.0:
        if 34.0 <= f <= 80.0:
            if f > 0:
                x_hf.append(math.log(f))
                psd_raw = 1.0 / (f ** true_beta)
                y_hf_raw.append(math.log(psd_raw))
                h_sq = 1.0 / (1.0 + (fc_hp / f) ** (2 * order_hp))
                y_hf_filtered.append(math.log(psd_raw * h_sq))
        f += df
        
    slope_raw_hf, _ = linear_regression(x_hf, y_hf_raw)
    slope_filt_hf, _ = linear_regression(x_hf, y_hf_filtered)
    
    print(f"True beta: {true_beta:.4f}")
    print(f"Estimated beta (Unfiltered): {-slope_raw_hf:.4f}")
    print(f"Estimated beta (Filtered):   {-slope_filt_hf:.4f}")
    print(f"High-frequency bias:         {(-slope_filt_hf - (-slope_raw_hf)):.4f}\n")

if __name__ == '__main__':
    simulate_bias_theoretical(2.0)
```

### 2. Time-Domain Signal Simulation (`simulate_filter_bias.py`)
This script generates synthetic $1/f^\beta$ noise, filters it in the time domain using `scipy.signal.lfilter`, and applies Welch's method to replicate the exact numerical pipeline described in the paper. It requires `numpy` and `scipy` to be installed.

---

## 4. Methodological Recommendations

> [!TIP]
> **Why high-pass filtering is unnecessary:**
> High-pass filtering at 33 Hz to attenuate alpha peaks (8–12 Hz) before fitting in the 34–80 Hz range is statistically redundant. Under standard Fourier transforms (like Welch's method with the Hamming window applied by the authors), power in the 8–12 Hz range is isolated to its respective bins and does not leak into the 34–80 Hz region.

### Recommended Alternatives:
1. **Raw PSD Fitting:** Simply compute the PSD of the raw, unfiltered signal and restrict the linear regression fit to the desired window (e.g., 34–80 Hz). This completely avoids filter transition band artifacts.
2. **Spectral Parameterization (FOOOF / specparam):** Use modern toolboxes (like `FOOOF`) which are designed to mathematically separate periodic (oscillatory) peaks from the aperiodic $1/f$ background without modifying the time-domain signal.
