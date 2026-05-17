# specparam 2.0rc6 vs fooof/specparam v1 — API changes

## Result access

`get_all_data()` is gone entirely. `get_params()` still exists but throws `AttributeError` when called with v1 key names (e.g. `'aperiodic_params'`). The reliable approach is to access `fg.results` directly — a list of `FitResults` namedtuples.

| v1 | v2 (rc6) |
|---|---|
| `fg.get_all_data('background_params', 'slope')` | `[r.aperiodic_fit[-1] for r in fg.results]` |
| `fg.get_all_data('peak_params')` | `[r.peak_fit for r in fg.results]` |
| `fg.get_all_data('error')` | `[r.metrics['error_mae'] for r in fg.results]` |
| `fg.get_all_data('r_squared')` | `[r.metrics['gof_rsquared'] for r in fg.results]` |

## `FitResults` field names

| v1 | v2 |
|---|---|
| `r.aperiodic_params` | `r.aperiodic_fit` — array `[offset, exponent]` |
| `r.peak_params` | `r.peak_fit` — array of `[CF, Amp, BW]` rows |
| `r.error` | `r.metrics['error_mae']` |
| `r.r_squared` | `r.metrics['gof_rsquared']` |

## Constructor

| v1 | v2 |
|---|---|
| `min_peak_amplitude=` | `min_peak_height=` |
| `background_params` terminology | `aperiodic` terminology throughout |

## Imports

```python
# v1
from fooof import FOOOFGroup

# v2
from specparam import SpectralGroupModel
```

## Notes

- Pylance/type checker will complain that `fg.results` is `list[Unknown]` — the package lacks type stubs. Safe to ignore; the attributes work at runtime.
- `r.aperiodic_fit[-1]` is the exponent, `r.aperiodic_fit[0]` is the offset, `r.aperiodic_fit[1]` is the knee (only if knee mode is used).
- `r.peak_fit` is a flat array — reshape with `.reshape(-1, 3)` to get one `[CF, Amp, BW]` row per peak.
