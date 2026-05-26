# Oscillatory peak and 1/f^β shoulders from a single process

A stochastic process of randomly timed, randomly damped oscillatory pulses
can be decomposed into the envelope-and-carrier form
$X(t) = A(t)\cos(2\pi f_0 t + \varphi(t))$, where the envelope $A(t)$
inherits a spectral structure that can produce $1/f^\beta$-like shoulders
around the carrier frequency. One distribution, one mechanism, two
spectral features.

## Generative process

$$
X(t) = \sum_k A_k\, e^{-\gamma_k(t-t_k)}\cos\bigl(\omega_0(t-t_k)\bigr)\,
       \Theta(t-t_k)
$$

- $t_k$: Poisson arrival times
- $A_k$: random pulse amplitudes (i.i.d., zero mean)
- $\gamma_k \sim P_\gamma$: random damping rates
- $\omega_0 = 2\pi f_0$: fixed carrier frequency

A filtered Poisson process in the sense of [@korzeniowska2023ApparentUniversality]
but with a damped-resonant pulse shape — equivalent at the level of the
second-order spectrum to the damped-oscillator mixture of
[@evertz2022AlphaBlocking1].

## Rice decomposition

Expanding $\cos(\omega_0(t-t_k))$:

$$
X(t) = X_c(t)\cos(\omega_0 t) + X_s(t)\sin(\omega_0 t)
$$

with

$$
X_c(t) = \sum_k A_k\cos(\omega_0 t_k)\,e^{-\gamma_k(t-t_k)}\Theta(t-t_k),
\qquad
X_s(t) = \sum_k A_k\sin(\omega_0 t_k)\,e^{-\gamma_k(t-t_k)}\Theta(t-t_k).
$$

$X_c$ and $X_s$ are themselves filtered Poisson processes with *monotonic
exponential* pulses and zero-mean random amplitudes — the exact form
studied by [@korzeniowska2023ApparentUniversality]. The oscillatory
character of $X$ is carried entirely by the carrier; the envelope dynamics
live in the low-pass shot noises $X_c, X_s$.

The envelope-phase identity [@rice1944MathematicalAnalysisRandom] gives

$$
X(t) = A(t)\cos(\omega_0 t + \varphi(t)),
\qquad
A(t) = \sqrt{X_c^2(t) + X_s^2(t)}.
$$

## Spectrum

In the narrowband regime $\gamma_k \ll \omega_0$, $X_c$ and $X_s$ are
uncorrelated, and the modulation theorem gives

$$
S_X(f) = \tfrac{1}{4}\bigl[S_{X_c}(f - f_0) + S_{X_c}(f + f_0)\bigr].
$$

By Campbell's theorem,

$$
S_{X_c}(f) \;\propto\; \int_0^\infty
P_\gamma(\gamma)\,\frac{2\gamma}{\gamma^2 + (2\pi f)^2}\,d\gamma.
$$

This is a sum of Lorentzians centred at zero — the canonical form for
broadband spectral shapes built from distributed relaxation rates
[@milotti20021NoisePedagogical]. Shifted to $\pm f_0$ by the modulation,
it produces broadband shoulders around the carrier, while the peak at
$f_0$ arises from the integrated power of $X_c$.

## What the construction shows

A single damping distribution $P_\gamma$ can simultaneously generate

- a narrowband peak at $f_0$, dominated by small-$\gamma$ contributions
  (narrow Lorentzians);
- broadband $1/f^\beta$-like shoulders around $f_0$, contributed by the
  broader-$\gamma$ part of $P_\gamma$.

Observing both features in a recorded PSD therefore does not, on its own,
require two distinct mechanisms. The periodic/aperiodic split is one
possible decomposition of the spectrum, not a feature the spectrum
compels.

## Validity and scope

- The envelope/carrier representation requires $\gamma_k \ll \omega_0$.
  For heavily damped pulses the algebra still goes through but
  $X_c, X_s$ are no longer slowly varying, and "$A(t)$" stops being
  meaningfully an envelope.
- This is a *possibility* argument, not an identification. Other
  generative processes (point-process models, dendritic filtering, generic
  stable linear stochastic dynamics) can produce equivalent second-order
  spectra. Discriminating between them requires going beyond the PSD.
- The construction adopts Korzeniowska's shot-noise formalism but uses a
  resonant pulse shape rather than their monotonic exponential. The
  finite-size scaling-bias result derived in their paper is for the
  exponential-pulse case and is not invoked here.

## References (Better BibTeX keys)

- `@rice1944MathematicalAnalysisRandom`
- `@evertz2022AlphaBlocking1`
- `@milotti20021NoisePedagogical`
- `@korzeniowska2023ApparentUniversality`