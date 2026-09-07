"""Zoomed FFT and chirp Z-transform in PyTorch.

Along one axis, with ``N`` input and ``M`` output samples::

    zoom_fft(x)[m]  = sum_n x[n] exp(-i w_m (n - c))          m = 0..M-1
    zoom_ifft(X)[n] = sum_m X[m] exp(+i w_m (n - c))          n = 0..N-1
    czt(x)[k]       = sum_n x[n] exp(-i a n) exp(i w n k)      k = 0..M-1

``w_m = zoom_freq(M, k_start, k_end, include_end)`` samples the band in
radians per sample and ``c`` is the index of the origin (``center``).  With
default arguments ``zoom_fft`` is ``torch.fft.fft(x, norm="ortho")``,
``zoom_ifft`` its inverse and ``czt`` the unnormalised DFT.

Every function transforms the last axis by default; ``dim`` selects one or
several axes (the transform is separable) and the other axes are batch axes.
The input precision is kept (complex64 in, complex64 out; real input is
promoted), any device works, and everything is differentiable in ``x``.

The chirp Z-transform is Bluestein's algorithm: three FFTs of length
``next_pow2(N + M - 1)``.  Chirp phases are reduced modulo 2*pi in float64
before the cast to the working precision, so float32 stays accurate even
when the phases reach thousands of radians.
"""

from __future__ import annotations

import math

import torch
from torch.fft import fft, ifft

__all__ = ["czt", "czt_plain", "zoom_fft", "zoom_ifft", "zoom_freq"]

TWO_PI = 2 * math.pi

# Exponent p of the scaling 1 / M**p, per norm and per direction.
_NORM = {"backward": {"forward": 0.0, "inverse": 1.0},
         "forward": {"forward": 1.0, "inverse": 0.0},
         "ortho": {"forward": 0.5, "inverse": 0.5}}


# --- helpers ---------------------------------------------------------------

def _complex(x):
    """Promote real input to complex; only float32/64 and complex64/128 are accepted."""
    if x.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError(f"unsupported dtype {x.dtype}; expected float32/64 or complex64/128")
    return x.to(torch.promote_types(x.dtype, torch.complex64))


def _phasor(phase, like):
    """``exp(i phase)`` on the device and in the precision of ``like``.

    ``phase`` is float64 and is reduced modulo 2*pi *before* the cast, so
    ``exp(i w k^2 / 2)`` keeps full accuracy in float32 for large phases.
    """
    phase = torch.remainder(phase, TWO_PI).to(device=like.device, dtype=like.real.dtype)
    return torch.complex(phase.cos(), phase.sin())


def _axes(x, dim, **params):
    """Yield ``(axis, {name: value})`` per transformed axis.

    A scalar parameter applies to every axis; a sequence gives one value per
    axis, in the order of ``dim``.
    """
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    if any(not -x.ndim <= d < x.ndim for d in dims):
        raise ValueError(f"dim={dim!r} is out of range for a {x.ndim}-D input")
    dims = tuple(d % x.ndim for d in dims)
    if len(set(dims)) < len(dims):
        raise ValueError(f"dim={dim!r} names the same axis twice")
    for name, value in params.items():
        if not isinstance(value, (tuple, list)):
            params[name] = (value,) * len(dims)
        elif len(value) != len(dims):
            raise ValueError(f"{name} needs one value per transformed axis ({len(dims)})")
    for i, d in enumerate(dims):
        yield d, {name: value[i] for name, value in params.items()}


def _band(n, k_start, k_end, include_end):
    """The spacing and the ``n`` band samples over ``[k_start, k_end]``."""
    step = (k_end - k_start) / (max(n - 1, 1) if include_end else n)
    return step, k_start + step * torch.arange(n, dtype=torch.float64)


# --- readable twin ---------------------------------------------------------
# czt_plain and czt compute the same transform.  czt_plain is Bluestein's
# algorithm written out one step per line, every phasor evaluated in float64
# and cast to the working precision afterwards; czt is the same steps with the
# phases reduced modulo 2*pi first, so that the cosine and the sine can be
# taken in the working precision on the device, with the trivial factors
# skipped and the convolution padded to a power of two.  Both stay at the
# round-off of the working precision; the tests pin the two together.

def czt_plain(x, n_out=None, w_phase=None, a_phase=0.0, dim=-1):
    """``czt``, written out.

    Bluestein's identity ``n k = (n² + k² − (k − n)²) / 2`` turns the sum
    ``X[k] = Σ_n x[n] e^{-i a n} e^{i w n k}`` into a convolution::

        X[k] = e^{i w k²/2} · Σ_n [ x[n] e^{-i a n} e^{i w n²/2} ] · e^{-i w (k − n)²/2}

    so the transform is three steps: multiply the input by a chirp, convolve
    with the conjugate chirp (by FFT, zero-padded so nothing wraps around),
    multiply the result by the chirp again.  Every phasor is evaluated in
    float64 and cast to the working precision afterwards, which is accurate
    for any phase but builds a complex128 array on the CPU.  ``czt`` reduces the
    phase modulo 2*pi instead and takes the cosine and the sine in the working
    precision, on the device; that reduction is what keeps the cheaper route
    accurate, and the two agree to float32 round-off on transforms whose
    phases reach thousands of radians.
    """
    x = _complex(x)
    for d, p in _axes(x, dim, n_out=n_out, w_phase=w_phase, a_phase=a_phase):
        x = _czt_plain_last(x.movedim(d, -1), **p).movedim(-1, d)
    return x


def _czt_plain_last(x, n_out, w_phase, a_phase):
    """Bluestein's algorithm along the last axis, step by step."""
    n = x.shape[-1]
    m = n if n_out is None else int(n_out)
    w = -TWO_PI / m if w_phase is None else float(w_phase)
    a = float(a_phase)

    def phasor(phase):                                    # e^{i phase}, in the precision of x
        return torch.exp(1j * phase).to(device=x.device, dtype=x.dtype)

    k = torch.arange(max(n, m), dtype=torch.float64)
    chirp = phasor(w * k * k / 2)                         # e^{i w k²/2}

    # 1. multiply the input by the start phase and the chirp
    xn = x * phasor(-a * k[:n]) * chirp[:n]

    # 2. the convolution kernel e^{-i w j²/2} for every lag j = k - n, from -(n-1) to m-1
    j = torch.arange(-(n - 1), m, dtype=torch.float64)
    kernel = phasor(-w * j * j / 2)

    # 3. linear convolution by FFT: pad to the full length so the circular one is linear
    length = n + len(j) - 1
    conv = ifft(fft(xn, n=length) * fft(kernel, n=length))

    # 4. the lag j = k - n sits at position k + n - 1; multiply by the chirp again
    return conv[..., n - 1 : n - 1 + m] * chirp[:m]


# --- chirp Z-transform -----------------------------------------------------

def _czt_last(x, n_out, w_phase, a_phase):
    """Bluestein's chirp Z-transform along the last axis of a complex ``x``."""
    n = x.shape[-1]
    m = n if n_out is None else int(n_out)
    w = -TWO_PI / m if w_phase is None else float(w_phase)
    a = float(a_phase)
    # Power of two >= n + m - 1.  That is shorter than the full linear length
    # 2n + m - 2 that czt_plain pads to, so the end of the convolution wraps
    # around, but it wraps below index n - 1, outside the window kept below.
    n_fft = 1 << (n + m - 2).bit_length()
    k = torch.arange(max(n, m), dtype=torch.float64)
    chirp = _phasor(w * k * k / 2, x)                    # exp(i w k^2 / 2)
    # n k = (n^2 + k^2 - (k - n)^2) / 2 turns the sum over n into a linear
    # convolution of x[n] exp(-i a n) exp(i w n^2 / 2) with exp(-i w j^2 / 2),
    # j = k - n in [-(n - 1), m - 1], which three FFTs compute.
    pre = chirp[:n] if a == 0 else chirp[:n] * _phasor(-a * k[:n], x)
    kernel = torch.cat([chirp[1:n].flip(0), chirp[:m]]).conj()
    y = ifft(fft(x * pre, n=n_fft) * fft(kernel, n=n_fft))
    return y[..., n - 1 : n - 1 + m] * chirp[:m]


def czt(x, n_out=None, w_phase=None, a_phase=0.0, dim=-1):
    """Chirp Z-transform along ``dim``.

    Along each transformed axis, with ``N`` input samples::

        X[k] = sum_{n=0}^{N-1} x[n] exp(-i a n) exp(i w n k),   k = 0, ..., M-1

    Parameters
    ----------
    x : Tensor
        Real or complex input; the axes not in ``dim`` are batch axes.
    n_out : int or sequence of int, optional
        Output samples ``M`` per axis.  Default: the input length.
    w_phase : float or sequence of float, optional
        Phase increment ``w`` in radians.  Default ``-2*pi/M``, which with
        ``M = N`` is the unnormalised DFT, ``torch.fft.fft(x)``.
    a_phase : float or sequence of float
        Starting phase ``a`` in radians.  Default 0.
    dim : int or sequence of int
        Axes to transform.  Default: the last axis.
    """
    x = _complex(x)
    for d, p in _axes(x, dim, n_out=n_out, w_phase=w_phase, a_phase=a_phase):
        x = _czt_last(x.movedim(d, -1), **p).movedim(-1, d)
    return x


# --- zoomed FFT ------------------------------------------------------------

def zoom_freq(n, k_start=0.0, k_end=TWO_PI, include_end=False):
    """The ``n`` frequencies (radians per sample) at which ``zoom_fft``
    samples the band: ``k_start + step * m``, as a float64 tensor.

    ``include_end=False`` samples like FFT bins (``step = span / n``, right end
    excluded); ``include_end=True`` puts the last sample on ``k_end``
    (``step = span / (n - 1)``).  The analogue of ``torch.fft.fftfreq``.
    """
    return _band(n, k_start, k_end, include_end)[1]


def _zoom_last(x, n_out, k_start, k_end, center, include_end, inverse):
    """One axis of ``zoom_fft`` (or of ``zoom_ifft`` if ``inverse``), unnormalised."""
    n_out = x.shape[-1] if n_out is None else int(n_out)
    m = x.shape[-1] if inverse else n_out                # band samples
    n = n_out if inverse else x.shape[-1]                # grid samples
    c = float(n // 2) if center is True else float(center)
    step, freq = _band(m, k_start, k_end, include_end)    # step and zoom_freq(m, ...)
    ramp = None if c == 0 else _phasor(c * freq, x)
    if inverse:
        # sum_m X[m] e^{+i w_m (n - c)} = e^{+i k_start n} sum_m [X[m] e^{-i c w_m}] e^{+i step m n}
        y = _czt_last(x if ramp is None else x * ramp.conj(), n, step, 0.0)
        return y if k_start == 0 else y * _phasor(k_start * torch.arange(n, dtype=torch.float64), x)
    # sum_n x[n] e^{-i w_m (n - c)} = e^{+i c w_m} sum_n x[n] e^{-i k_start n} e^{-i step n m}
    y = _czt_last(x, m, -step, k_start)
    return y if ramp is None else y * ramp


def _zoom(x, n_out, k_start, k_end, dim, norm, center, include_end, inverse):
    """``zoom_fft`` / ``zoom_ifft`` over every axis in ``dim``, then the ``norm`` factor."""
    norm = norm or "backward"
    if norm not in _NORM:
        raise ValueError(f"unknown norm {norm!r}; expected 'ortho', 'forward' or 'backward'")
    x = _complex(x)
    n_band = 1
    for d, p in _axes(x, dim, n_out=n_out, k_start=k_start, k_end=k_end, center=center):
        y = _zoom_last(x.movedim(d, -1), include_end=include_end, inverse=inverse, **p)
        y = y.movedim(-1, d)
        n_band *= (x if inverse else y).shape[d]
        x = y
    p = _NORM[norm]["inverse" if inverse else "forward"]
    return x if p == 0 else x * n_band ** -p


def zoom_fft(x, n_out=None, k_start=0.0, k_end=TWO_PI, dim=-1, norm="ortho",
             center=False, include_end=False):
    """Zoomed FFT: the spectrum of ``x`` on the band ``[k_start, k_end]``.

    Along each transformed axis, with ``N`` input samples::

        X[m] = sum_{n=0}^{N-1} x[n] exp(-i w_m (n - c)),   m = 0, ..., M-1

    with ``w_m = zoom_freq(M, k_start, k_end, include_end)`` and ``c`` the
    origin index set by ``center``.  With default arguments this is exactly
    ``torch.fft.fft(x, norm="ortho")``.

    Parameters
    ----------
    x : Tensor
        Real or complex input; the axes not in ``dim`` are batch axes.
    n_out : int or sequence of int, optional
        Band samples ``M`` per axis.  Default: the input length.
    k_start, k_end : float or sequence of float
        Band limits in radians per sample.  Default: the full circle
        ``[0, 2*pi)``; ``[-pi, pi)`` gives the ``fftshift``-ed spectrum.
    dim : int or sequence of int
        Axes to transform.  Default: the last axis.
    norm : {"ortho", "forward", "backward"}
        As in ``torch.fft``: ``"backward"`` leaves the forward transform
        unnormalised, ``"forward"`` divides it by the number of band samples
        ``prod(M)``, ``"ortho"`` by its square root.
    center : bool, float or sequence
        Origin of the input grid.  ``False``: sample 0 (``torch.fft``).
        ``True``: index ``N // 2`` (``fftshift``; on the full band this is
        ``fft(ifftshift(x))``).  A float pins the origin at that index, which
        may be fractional: ``(N - 1) / 2`` is the centre of the grid.
    include_end : bool
        Sample ``k_end`` exactly (``M - 1`` steps) instead of excluding it.
    """
    return _zoom(x, n_out, k_start, k_end, dim, norm, center, include_end, inverse=False)


def zoom_ifft(x, n_out=None, k_start=0.0, k_end=TWO_PI, dim=-1, norm="ortho",
              center=False, include_end=False):
    """Zoomed inverse FFT: the adjoint of ``zoom_fft`` over the same band.

    The input holds ``M`` band samples along each transformed axis (the
    output grid of ``zoom_fft``) and the output has ``N = n_out`` samples::

        x[n] = sum_{m=0}^{M-1} X[m] exp(+i w_m (n - c)),   n = 0, ..., N-1

    This is the conjugate transpose of ``zoom_fft`` with the same
    ``k_start``, ``k_end``, ``center`` and ``include_end`` under mirrored
    ``norm`` (``"backward"`` <-> ``"forward"``, ``"ortho"`` <-> ``"ortho"``),
    as for ``torch.fft.fft`` / ``ifft``.  On a full band (``k_end = k_start
    + 2*pi``, ``include_end=False``, ``n_out = M``) it is also the exact
    inverse, and with default arguments it is ``torch.fft.ifft(x, norm="ortho")``.

    Parameters
    ----------
    x : Tensor
        Band samples along ``dim``; the other axes are batch axes.
    n_out : int or sequence of int, optional
        Output (grid) samples ``N`` per axis.  Default: the input length.
    k_start, k_end, dim, include_end
        As passed to ``zoom_fft``.
    norm : {"ortho", "forward", "backward"}
        As in ``torch.fft``: ``"backward"`` divides this inverse transform by
        the number of band samples ``prod(M)``, ``"forward"`` leaves it
        unnormalised, ``"ortho"`` divides by the square root.
    center : bool, float or sequence
        Origin of the *output* grid, with the meaning it has in ``zoom_fft``
        (``True`` is ``N // 2``; on the full band this is ``fftshift(ifft(X))``).
    """
    return _zoom(x, n_out, k_start, k_end, dim, norm, center, include_end, inverse=True)
