import numpy as np
import scipy.signal as signal

def simulate_bias():
    # Sampling frequency and duration (similar to sEEG in the paper)
    fs = 200.0  # Hz
    duration = 60.0  # seconds
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Let's generate a synthetic 1/f^beta signal with a true beta = 2.0 (Brownian noise-like)
    # We do this in the frequency domain to get an exact 1/f^2.0 relationship
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    freqs[0] = freqs[1]  # avoid division by zero
    
    true_beta = 2.0
    # Amplitude spectrum: A(f) = 1 / f^(beta/2)
    amplitude_spectrum = 1.0 / (freqs ** (true_beta / 2.0))
    # Random phases
    phases = np.random.uniform(0, 2*np.pi, len(freqs))
    
    # Complex spectrum
    complex_spectrum = amplitude_spectrum * np.exp(1j * phases)
    # Inverse FFT to get time-domain signal
    x = np.fft.irfft(complex_spectrum, n=n_samples)
    
    # 1. Low-frequency analysis: 7th-order lowpass Butterworth at 8 Hz
    # Fitting range: 0.5 - 4 Hz
    b_lp, a_lp = signal.butter(7, 8.0, btype='low', fs=fs)
    x_lp = signal.lfilter(b_lp, a_lp, x)
    
    # 2. High-frequency analysis: 7th-order highpass Butterworth at 33 Hz
    # Fitting range: 34 - 80 Hz
    b_hp, a_hp = signal.butter(7, 33.0, btype='high', fs=fs)
    x_hp = signal.lfilter(b_hp, a_hp, x)
    
    # Compute PSD using Welch's method (same as paper: 2048-point FFT, Hamming window)
    f_welch, psd_raw = signal.welch(x, fs=fs, window='hamming', nperseg=2048)
    f_welch_lp, psd_lp = signal.welch(x_lp, fs=fs, window='hamming', nperseg=2048)
    f_welch_hp, psd_hp = signal.welch(x_hp, fs=fs, window='hamming', nperseg=2048)
    
    # Fit in Low-Frequency Range (0.5 - 4 Hz)
    idx_lf = (f_welch >= 0.5) & (f_welch <= 4.0)
    slope_raw_lf = np.polyfit(np.log(f_welch[idx_lf]), np.log(psd_raw[idx_lf]), 1)[0]
    slope_lp_lf = np.polyfit(np.log(f_welch_lp[idx_lf]), np.log(psd_lp[idx_lf]), 1)[0]
    
    # Fit in High-Frequency Range (34 - 80 Hz)
    idx_hf = (f_welch >= 34.0) & (f_welch <= 80.0)
    slope_raw_hf = np.polyfit(np.log(f_welch[idx_hf]), np.log(psd_raw[idx_hf]), 1)[0]
    slope_hp_hf = np.polyfit(np.log(f_welch_hp[idx_hf]), np.log(psd_hp[idx_hf]), 1)[0]
    
    # Exponents are -slope
    beta_raw_lf = -slope_raw_lf
    beta_lp_lf = -slope_lp_lf
    
    beta_raw_hf = -slope_raw_hf
    beta_hp_hf = -slope_hp_hf
    
    print(f"True beta: {true_beta:.4f}")
    print("\n--- Low-Frequency Band (0.5 - 4 Hz) with 8 Hz Low-Pass Filter ---")
    print(f"Estimated beta (Unfiltered): {beta_raw_lf:.4f}")
    print(f"Estimated beta (Filtered):   {beta_lp_lf:.4f}")
    print(f"Low-frequency bias:          {beta_lp_lf - beta_raw_lf:.4f}")
    
    print("\n--- High-Frequency Band (34 - 80 Hz) with 33 Hz High-Pass Filter ---")
    print(f"Estimated beta (Unfiltered): {beta_raw_hf:.4f}")
    print(f"Estimated beta (Filtered):   {beta_hp_hf:.4f}")
    print(f"High-frequency bias:         {beta_hp_hf - beta_raw_hf:.4f}")

if __name__ == '__main__':
    simulate_bias()