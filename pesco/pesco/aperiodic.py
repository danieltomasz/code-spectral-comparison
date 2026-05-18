"""Bush et al. Lorentzian aperiodic model for specparam 2.0 rc syntax.

This module keeps only the aperiodic model used in Bush et al. 2024:

    L(f) = A * (fk**chi + fmin**chi) / (fk**chi + f**chi)

The functions below operate in specparam's fit-function convention:
linear frequency inputs, log10-power outputs. The fitted parameters are:

    offset, log_knee, exponent

where ``offset = log10(A)``, ``log_knee = log10(fk)``, and
``exponent = chi``.

"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import inspect

import numpy as np


@dataclass
class AperiodicFit:
    """Container returned by :func:`fit_aperiodic`."""

    params: np.ndarray
    freqs: np.ndarray
    power_spectrum: np.ndarray
    aperiodic_fit: np.ndarray
    fmin: float


def _flatten_params(params: Sequence[Any], expected: Optional[int] = None) -> np.ndarray:
    """Collect function parameters into a flat float array."""

    if len(params) == 1 and np.ndim(params[0]) > 0:
        params = tuple(np.asarray(params[0], dtype=float).ravel())

    arr = np.asarray(params, dtype=float).ravel()
    if expected is not None and arr.size != expected:
        raise ValueError("Expected {} parameters, received {}.".format(expected, arr.size))

    return arr


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert input to a finite, 1D float array."""

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("{} must not be empty.".format(name))
    if not np.all(np.isfinite(arr)):
        raise ValueError("{} contains NaN or Inf values.".format(name))
    return arr


def _check_freqs(freqs: Sequence[float]) -> np.ndarray:
    """Validate and return a 1D frequency array."""

    freqs = _as_1d_float_array(freqs, "freqs")
    if np.any(freqs <= 0):
        raise ValueError("All frequency values must be positive.")
    return freqs


def resolve_fmin(freqs: Sequence[float], fmin: Optional[float] = None) -> float:
    """Return explicit ``fmin`` or infer it from positive frequency values."""

    if fmin is not None:
        fmin = float(fmin)
        if not np.isfinite(fmin) or fmin <= 0:
            raise ValueError("fmin must be a positive finite value.")
        return fmin

    freqs = _check_freqs(freqs)
    return float(np.min(freqs[freqs > 0]))


def trim_spectrum(
    freqs: Sequence[float],
    power_spectrum: Sequence[float],
    freq_range: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim a spectrum to a frequency range."""

    freqs = _check_freqs(freqs)
    power_spectrum = _as_1d_float_array(power_spectrum, "power_spectrum")
    if freqs.shape != power_spectrum.shape:
        raise ValueError("freqs and power_spectrum must be the same length.")

    if freq_range is None:
        return freqs, power_spectrum

    if len(freq_range) != 2:
        raise ValueError("freq_range must be a [low, high] pair.")
    low, high = float(freq_range[0]), float(freq_range[1])
    if low >= high:
        raise ValueError("freq_range lower bound must be less than upper bound.")

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        raise ValueError("freq_range does not overlap the supplied frequency values.")

    return freqs[mask], power_spectrum[mask]


def as_log_power(power_spectrum: Sequence[float], power_is_log: bool = False) -> np.ndarray:
    """Return log10 power, validating linear-power inputs when needed."""

    power_spectrum = _as_1d_float_array(power_spectrum, "power_spectrum")
    if power_is_log:
        return power_spectrum
    if np.any(power_spectrum <= 0):
        raise ValueError("Linear power_spectrum values must be positive.")
    return np.log10(power_spectrum)


def lorentzian_function(
    xs: Sequence[float],
    *params: float,
    fmin: Optional[float] = None,
) -> np.ndarray:
    """Evaluate the Bush et al. Lorentzian aperiodic function.

    Parameters
    ----------
    xs
        Linear frequency values.
    *params
        ``offset, log_knee, exponent``.
    fmin
        Minimal positive frequency of interest. If None, uses ``min(xs)``.

    Returns
    -------
    np.ndarray
        Aperiodic fit values in log10-power units.
    """

    xs = _check_freqs(xs)
    fmin = resolve_fmin(xs, fmin)
    offset, log_knee, exponent = _flatten_params(params, expected=3)

    knee_term = np.power(10.0, log_knee * exponent)
    fmin_term = np.power(fmin, exponent)

    return (
        offset
        + np.log10(knee_term + fmin_term)
        - np.log10(knee_term + np.power(xs, exponent))
    )


def make_lorentzian_function(fmin: Optional[float] = None) -> Callable[..., np.ndarray]:
    """Return a Lorentzian function with ``fmin`` captured for specparam."""

    def _lorentzian(xs: Sequence[float], *params: float) -> np.ndarray:
        return lorentzian_function(xs, *params, fmin=fmin)

    label = "inferred" if fmin is None else "{:g}".format(float(fmin))
    _lorentzian.__name__ = "lorentzian_function_fmin_{}".format(label)
    _lorentzian.__doc__ = lorentzian_function.__doc__

    return _lorentzian


def generate_aperiodic(
    freqs: Sequence[float],
    aperiodic_params: Sequence[float],
    fmin: Optional[float] = None,
) -> np.ndarray:
    """Generate the Lorentzian aperiodic component in log10-power units."""

    return lorentzian_function(freqs, *aperiodic_params, fmin=fmin)


def default_lorentzian_bounds(
    freqs: Optional[Sequence[float]] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return data-aware bounds for ``offset, log_knee, exponent``.

    The paper bounded ``fk`` from ``fmin / 10`` to the upper fitted frequency.
    Since this module optimizes ``log_knee = log10(fk)``, these bounds are
    applied in log10-frequency space when enough information is available.
    """

    if freqs is not None:
        freqs = _check_freqs(freqs)
        fmin = resolve_fmin(freqs, fmin)
        fmax = float(np.max(freqs)) if fmax is None else float(fmax)

    if fmin is None or fmax is None:
        log_knee_low, log_knee_high = -np.inf, np.inf
    else:
        if fmax <= 0 or fmin <= 0:
            raise ValueError("fmin and fmax must be positive.")
        if fmax <= fmin / 10.0:
            raise ValueError("fmax must be greater than fmin / 10.")
        log_knee_low = np.log10(fmin / 10.0)
        log_knee_high = np.log10(fmax)

    return (
        np.array([-np.inf, log_knee_low, -np.inf], dtype=float),
        np.array([np.inf, log_knee_high, np.inf], dtype=float),
    )


def guess_lorentzian_params(
    freqs: Sequence[float],
    power_spectrum: Sequence[float],
    freq_range: Optional[Sequence[float]] = None,
    power_is_log: bool = False,
    fmin: Optional[float] = None,
    log_knee_guess: Optional[float] = None,
) -> np.ndarray:
    """Create a data-driven initial guess for ``offset, log_knee, exponent``."""

    freqs, powers = trim_spectrum(freqs, power_spectrum, freq_range)
    powers = as_log_power(powers, power_is_log=power_is_log)
    fmin = resolve_fmin(freqs, fmin)

    offset = powers[0]
    exponent = abs((powers[-1] - powers[0]) / (np.log10(freqs[-1]) - np.log10(freqs[0])))

    if log_knee_guess is None:
        # Preserves the old 10 Hz starting point when fmin is 1 Hz, while scaling
        # for recordings whose lowest reliable frequency is not 1 Hz.
        log_knee_guess = np.log10(10.0 * fmin)

    return np.array([offset, log_knee_guess, exponent], dtype=float)


def make_lorentzian_mode(fmin: Optional[float] = None) -> Any:
    """Create a specparam rc6-compatible custom aperiodic ``Mode`` object."""

    try:
        from specparam.modes.mode import Mode
        from specparam.modes.params import ParamDefinition
    except ImportError as exc:
        raise ImportError(
            "specparam is required for make_lorentzian_mode(). "
            "Install specparam 2.0.0rc6 or newer, or use lorentzian_function directly."
        ) from exc

    params = OrderedDict(
        [
            ("offset", "Log10 aperiodic offset at fmin."),
            ("log_knee", "Log10 knee frequency."),
            ("exponent", "Aperiodic exponent."),
        ]
    )

    mode_kwargs = {
        "name": "lorentzian",
        "component": "aperiodic",
        "description": "Bush et al. Lorentzian aperiodic function.",
        "func": make_lorentzian_function(fmin=fmin),
        "jacobian": None,
        "params": ParamDefinition(params),
        "ndim": 1,
        "freq_space": "linear",
        "powers_space": "log10",
    }

    # specparam 2.0.0rc6 has no formula argument; newer dev builds do.
    if "formula" in inspect.signature(Mode.__init__).parameters:
        mode_kwargs["formula"] = "L(f) = A * (fk^chi + fmin^chi) / (fk^chi + f^chi)"

    return Mode(**mode_kwargs)


def specparam_model_kwargs(
    freqs: Sequence[float],
    power_spectrum: Sequence[float],
    freq_range: Optional[Sequence[float]] = None,
    power_is_log: bool = False,
    fmin: Optional[float] = None,
    algorithm_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for ``specparam.SpectralModel`` rc6+."""

    fit_freqs, fit_powers = trim_spectrum(freqs, power_spectrum, freq_range)
    fit_log_powers = as_log_power(fit_powers, power_is_log=power_is_log)
    fmin = resolve_fmin(fit_freqs, fmin)

    settings = {} if algorithm_settings is None else dict(algorithm_settings)
    settings.setdefault(
        "ap_guess",
        guess_lorentzian_params(
            fit_freqs,
            fit_log_powers,
            power_is_log=True,
            fmin=fmin,
        ),
    )
    settings.setdefault("ap_bounds", default_lorentzian_bounds(fit_freqs, fmin=fmin))

    return {
        "aperiodic_mode": make_lorentzian_mode(fmin=fmin),
        "algorithm_settings": settings,
    }


def fit_spectral_model(
    freqs: Sequence[float],
    power_spectrum: Sequence[float],
    freq_range: Optional[Sequence[float]] = None,
    power_is_log: bool = False,
    fmin: Optional[float] = None,
    algorithm_settings: Optional[Dict[str, Any]] = None,
    **model_kwargs: Any,
) -> Any:
    """Fit a specparam ``SpectralModel`` with the Bush Lorentzian mode."""

    try:
        from specparam import SpectralModel
    except ImportError as exc:
        raise ImportError(
            "specparam is required for fit_spectral_model(). "
            "Use fit_aperiodic() for the standalone scipy-based aperiodic fit."
        ) from exc

    kwargs = specparam_model_kwargs(
        freqs,
        power_spectrum,
        freq_range=freq_range,
        power_is_log=power_is_log,
        fmin=fmin,
        algorithm_settings=algorithm_settings,
    )
    model = SpectralModel(**kwargs, **model_kwargs)

    fit_powers = (
        np.power(10.0, _as_1d_float_array(power_spectrum, "power_spectrum"))
        if power_is_log
        else power_spectrum
    )
    model.fit(freqs, fit_powers, freq_range=freq_range)

    return model


def _require_curve_fit() -> Callable[..., Any]:
    """Import scipy's curve_fit lazily."""

    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise ImportError(
            "fit_aperiodic() requires scipy. The model function and specparam "
            "Mode helper do not import scipy."
        ) from exc

    return curve_fit


def fit_aperiodic(
    freqs: Sequence[float],
    power_spectrum: Sequence[float],
    freq_range: Optional[Sequence[float]] = None,
    power_is_log: bool = False,
    fmin: Optional[float] = None,
    robust: bool = True,
    percentile_thresh: float = 0.025,
    ap_guess: Optional[Sequence[float]] = None,
    ap_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    maxfev: int = 5000,
) -> AperiodicFit:
    """Fit only the Bush Lorentzian aperiodic component with scipy."""

    freqs, powers = trim_spectrum(freqs, power_spectrum, freq_range)
    log_powers = as_log_power(powers, power_is_log=power_is_log)
    fmin = resolve_fmin(freqs, fmin)
    func = make_lorentzian_function(fmin=fmin)
    curve_fit = _require_curve_fit()

    guess = (
        guess_lorentzian_params(freqs, log_powers, power_is_log=True, fmin=fmin)
        if ap_guess is None
        else _as_1d_float_array(ap_guess, "ap_guess")
    )
    bounds = default_lorentzian_bounds(freqs, fmin=fmin) if ap_bounds is None else ap_bounds

    params, _ = curve_fit(
        func,
        freqs,
        log_powers,
        p0=guess,
        bounds=bounds,
        maxfev=maxfev,
        check_finite=False,
    )

    if robust:
        initial_fit = func(freqs, *params)
        flat = log_powers - initial_fit
        flat[flat < 0] = 0

        threshold = np.percentile(flat, percentile_thresh)
        mask = flat <= threshold
        if np.count_nonzero(mask) >= params.size:
            params, _ = curve_fit(
                func,
                freqs[mask],
                log_powers[mask],
                p0=params,
                bounds=bounds,
                maxfev=maxfev,
                check_finite=False,
            )

    return AperiodicFit(
        params=np.asarray(params, dtype=float),
        freqs=freqs,
        power_spectrum=log_powers,
        aperiodic_fit=func(freqs, *params),
        fmin=fmin,
    )


__all__ = [
    "AperiodicFit",
    "as_log_power",
    "default_lorentzian_bounds",
    "fit_aperiodic",
    "fit_spectral_model",
    "generate_aperiodic",
    "guess_lorentzian_params",
    "lorentzian_function",
    "make_lorentzian_function",
    "make_lorentzian_mode",
    "resolve_fmin",
    "specparam_model_kwargs",
    "trim_spectrum",
]
