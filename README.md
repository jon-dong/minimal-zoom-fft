# minimal-zoom-fft

The Fourier transform of a signal on any frequency band, at any sampling, in PyTorch: a zoomed FFT, computed by the chirp Z-transform. One implementation file, one dependency.

`torch.fft.fft` evaluates the spectrum of an `N`-sample signal at `N` equispaced frequencies covering the whole circle `[0, 2π)`. Often you only care about a narrow band, or you want it sampled more finely than `2π/N`, or you want `M ≠ N` output samples. Bluestein's algorithm does exactly that with three FFTs, at `O((N+M) log(N+M))` cost, on any device, with autograd.

Typical uses: pupil-to-PSF computation in microscopy, Fourier ptychography, diffraction onto a rescaled output grid, fine spectral analysis of a narrow band.

## Install

```bash
pip install minimal-zoom-fft
pip install git+https://github.com/jon-dong/minimal-zoom-fft     # the development version
pip install -e ".[test]" && pytest                               # from a checkout
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick start

```python
import torch
from minimal_zoom_fft import zoom_fft, zoom_ifft, zoom_freq, czt

x = torch.randn(256, dtype=torch.complex64)

# Spectrum on the band [-0.2, 0.2] rad/sample, sampled at 1024 points.
X = zoom_fft(x, n_out=1024, k_start=-0.2, k_end=0.2)
w = zoom_freq(1024, -0.2, 0.2)          # the 1024 frequencies, for plotting

# The adjoint (and, on a full band, the inverse).
x_back = zoom_ifft(zoom_fft(x), n_out=256)   # x again, to round-off

# Images: transform the last two axes, batch axes are untouched.
pupil = torch.randn(8, 64, 64, dtype=torch.complex64)
psf = zoom_fft(pupil, n_out=(256, 256), k_start=-0.5, k_end=0.5,
               dim=(-2, -1), center=True)

# The raw chirp Z-transform, if you need it.
X = czt(x, n_out=100, w_phase=-0.01, a_phase=0.3)
```

## API

Along one transformed axis with `N` input and `M` output samples:

| Function | Computes |
|---|---|
| `czt(x, n_out, w_phase, a_phase, dim)` | `X[k] = Σₙ x[n] e^{-i a n} e^{i w n k}`, `k = 0..M-1` |
| `zoom_fft(x, n_out, k_start, k_end, dim, norm, center, include_end)` | `X[m] = Σₙ x[n] e^{-i ωₘ (n - c)}` |
| `zoom_ifft(X, n_out, k_start, k_end, dim, norm, center, include_end)` | `x[n] = Σₘ X[m] e^{+i ωₘ (n - c)}` |
| `zoom_freq(n, k_start, k_end, include_end)` | the band samples `ωₘ = k_start + step · m` |

`ωₘ` samples `[k_start, k_end]` in radians per sample; `c` is the origin index set by `center`. Every function:

- acts on the last axis by default; `dim` takes an int or a tuple (the transform is separable, so a 2-D transform is `dim=(-2, -1)`);
- leaves all other axes alone (batch axes);
- accepts a scalar or one value per transformed axis for `n_out`, `k_start`, `k_end`, `center`, `w_phase`, `a_phase`;
- keeps the input's precision (complex64 in, complex64 out; real input is promoted to complex);
- runs on CPU, CUDA or MPS and is differentiable with respect to `x`.

`czt_plain` is the readable twin of `czt`, the same transform with Bluestein's algorithm written out one step per line: multiply by the chirp, convolve with the conjugate chirp by a zero-padded FFT, multiply by the chirp again. It comes first in [`core.py`](src/minimal_zoom_fft/core.py) and the tests pin it to `czt`. What differs is where the cast to the working precision happens: `czt_plain` evaluates each phasor in float64 and casts the result, while `czt` reduces the phase modulo 2π and then takes the cosine and the sine in the working precision, on the device, which is cheaper and, thanks to the reduction, as accurate. `czt` also skips the trivial factors and pads the convolution to a power of two. Read the plain one, call the fast one.

## Conventions

The defaults reproduce `torch.fft` to round-off: `zoom_fft(x)` is `torch.fft.fft(x, norm="ortho")`, `zoom_ifft` the inverse, and `czt(x)` the unnormalised DFT.

The band is sampled like FFT bins when `include_end=False`, the default: `step = (k_end - k_start) / M`, right end excluded, so `[0, 2π)` with `M = N` is the DFT. The `fftshift`-ed spectrum is the full band that starts at `k_start = -2π (N // 2) / N`: that is `[-π, π)` for even `N`, and half a bin above it for odd `N`. `include_end=True` puts the last sample on `k_end` (`step = span / (M - 1)`).

Normalisation follows `torch.fft`: `"backward"` leaves the forward transform unnormalised and divides the inverse by the number of band samples; `"forward"` does the opposite; `"ortho"` splits the factor. `zoom_ifft` is the conjugate transpose of `zoom_fft` over the same band under mirrored norms (`"backward"` ↔ `"forward"`, `"ortho"` ↔ `"ortho"`), for any band. On a full band, and with the same norm in both directions, it is also its inverse, to round-off.

The origin of the transform is set by `center`. With `False`, sample 0 is the origin, as in `torch.fft`. With `True`, the origin is at index `N // 2`, the `torch.fft.fftshift` convention, so on the full band `zoom_fft(x, center=True)` is `fft(ifftshift(x), norm="ortho")` and `zoom_ifft(X, center=True)` is `fftshift(ifft(X, norm="ortho"))`, to round-off. A float pins the origin at any index, possibly fractional: `(N - 1) / 2` is the geometric centre of the grid, which for even `N` lies between two samples (the convention of `psf_generator`). The correction is a phase ramp, so it is exact for any band and never wraps indices. In `zoom_ifft`, `center` refers to the output grid.

For accuracy, the Bluestein chirps `e^{i w k² / 2}` are evaluated in float64, reduced modulo `2π`, and only then cast to the working precision, so a float32 transform of thousands of samples stays at float32 round-off instead of losing digits to large phases.

## Relation to other implementations

- `scipy.signal.czt` / `scipy.signal.zoom_fft`: same algorithm. SciPy's contour is a general spiral (complex `w`, `a`); here it is restricted to the unit circle (phases `w_phase`, `a_phase`), which is the case needed for zooming and is numerically stable. This version is N-D, batched, GPU-capable and differentiable.
- `torch.fft`: recovered by the defaults, to round-off, see above.
- Origin: extracted from [psf_generator](https://github.com/Biomedical-Imaging-Group/psf_generator) (`custom_fft2`, `custom_ifft2`) and the `ciel` linear-operator library. The `ciel` names map as `custom_fft2(x, shape_out, k_start, k_end, norm, fftshift_input, include_end)` → `zoom_fft(x, shape_out, k_start, k_end, dim=(-2, -1), norm, center, include_end)`, and `custom_ifft2` → `zoom_ifft` in the same way, with `fftshift_input=True` becoming `center=True`. `psf_generator` follows another convention: its `custom_ifft2` takes samples on the grid, returns the band and uses the kernel `e^{+i ωₘ (n - c)}`, so `custom_ifft2(x, shape_out, k_start, k_end, norm, fftshift_input, include_end)` is `zoom_fft(x, shape_out, -k_start, -k_end, dim=(-2, -1), norm, center, include_end)` with the norm mirrored (`"backward"` ↔ `"forward"`) and `fftshift_input=True` becoming `center=(N - 1) / 2` (checked on bands symmetric about zero, the ones `psf_generator` uses).

## Tutorials

Three notebooks in [`notebooks/`](notebooks/), runnable after `pip install -e ".[notebooks]"`:

1. [Why a zoomed FFT](notebooks/01_why_zoom_fft.ipynb): the PSF of a disk pupil is an Airy disk that a plain FFT undersamples; zero-padding fixes that at a cost `zoom_fft` avoids.
2. [What it computes](notebooks/02_what_it_computes.ipynb): one figure and one block of checks per topic: the definition, why zero-padding gives the same samples, Bluestein's algorithm, the band, origin and normalisation conventions, the adjoint, images and autograd.
3. [Benchmark](notebooks/03_benchmark.ipynb): the arrays each route allocates, speed against padded FFTs and SciPy, CPU against MPS, precision against float64 summation, memory and edge cases.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests pin every convention above against explicit float64 summation and against `torch.fft`.

## License

MIT

## Manifest

- Purpose: the Fourier transform of a signal on any frequency band, at any sampling, in PyTorch: a zoomed FFT, computed by the chirp Z-transform. One implementation file, one dependency.
- Dependencies: `torch`.
- Size: about 350 lines of implementation in one module, 5 public functions; about 690 lines of tests; 3 tutorial notebooks.
- Origin: the `psf_generator` library and the `ciel` computational-imaging library, EPFL; Bluestein's chirp Z-transform. Used by `minimal-linop`'s `fft` extra.
- Provenance: written with Claude (Anthropic) from a brief; read and checked in full by Jonathan Dong.
- Version: 0.1.0, MIT.
