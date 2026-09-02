"""Chirp Z-transform (CZT) and zoomed FFT in PyTorch.

Every function acts on the last axis by default, accepts ``dim`` (an int or a
tuple of ints) to act on other or several axes -- the transforms are
separable, so an N-D transform is the 1-D transform applied axis by axis --
leaves the remaining (batch) axes untouched, runs on any device, keeps the
input's precision (complex64 in, complex64 out) and is differentiable with
respect to ``x``.

Definitions along one axis, with ``N`` input and ``M`` output samples::

    czt(x, M, w, a)[k]   = sum_n x[n] exp(-i a n) exp(i w n k)      k = 0..M-1
    zoom_fft(x)[m]       = sum_n x[n] exp(-i w_m (n - c))           m = 0..M-1
    zoom_ifft(X)[n]      = sum_m X[m] exp(+i w_m (n - c))           n = 0..N-1

where ``w_m = k_start + step * m`` samples the band ``[k_start, k_end]`` in
radians per sample and ``c`` is the index of the origin (see ``center``).
``zoom_fft`` with default arguments is exactly ``torch.fft.fft``; ``zoom_ifft``
is the conjugate transpose of ``zoom_fft`` over the same band (under mirrored
``norm``) and its inverse on a full band.

The CZT is computed with Bluestein's algorithm: three FFTs of length
``next_pow2(N + M - 1)``.  Chirp phases are evaluated in float64 and reduced
modulo 2*pi before the cast to the working precision, so float32 stays
accurate even for large transforms.
"""

from __future__ import annotations

import math
import numbers

import torch
from torch.fft import fft, ifft

__all__ = ["czt", "zoom_fft", "zoom_ifft", "zoom_freq"]

TWO_PI = 2.0 * math.pi

_REAL_DTYPE = {
    torch.float32: torch.float32,
    torch.complex64: torch.float32,
    torch.float64: torch.float64,
    torch.complex128: torch.float64,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    try:
        return _REAL_DTYPE[dtype]
    except KeyError:
        raise TypeError(
            f"unsupported dtype {dtype}; expected float32/float64 or "
            "complex64/complex128"
        ) from None


def _dims(dim, ndim: int) -> tuple[int, ...]:
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = tuple(d % ndim for d in dims)
    if len(set(dims)) != len(dims):
        raise ValueError(f"dim={dim!r} names the same axis twice")
    return dims


def _per_dim(value, n: int, name: str) -> tuple:
    """Broadcast a scalar (or None) to ``n`` axes; check the length otherwise."""
    if value is None or isinstance(value, numbers.Number):
        return (value,) * n
    value = tuple(value)
    if len(value) != n:
        raise ValueError(
            f"{name} must be a scalar or have one entry per transformed axis "
            f"({n}); got {len(value)}"
        )
    return value


def _phasor(phase: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """``exp(1j * phase)`` for a float64 CPU ``phase``.

    The phase is reduced modulo 2*pi *before* the cast to the working
    precision, so ``exp(i w k^2 / 2)`` keeps full accuracy in float32 even
    when ``w k^2`` reaches thousands of radians.
    """
    phase = torch.remainder(phase, TWO_PI).to(device=device, dtype=dtype)
    return torch.complex(torch.cos(phase), torch.sin(phase))


def _norm_factor(norm, n_band: int, inverse: bool) -> float:
    """Scaling that matches ``torch.fft`` conventions, ``n_band`` = number of
    band samples (output of the forward transform, input of the inverse)."""
    if norm == "ortho":
        return 1.0 / math.sqrt(n_band)
    if norm == "backward" or norm is None:
        return 1.0 / n_band if inverse else 1.0
    if norm == "forward":
        return 1.0 if inverse else 1.0 / n_band
    raise ValueError(
        f"unknown norm {norm!r}; expected 'ortho', 'forward' or 'backward'"
    )


def _step(k_start: float, k_end: float, n: int, include_end: bool) -> float:
    """Spacing of ``n`` band samples over ``[k_start, k_end]``."""
    if include_end:
        return (k_end - k_start) / (n - 1) if n > 1 else 0.0
    return (k_end - k_start) / n


def _origin(center, n: int) -> float:
    if center is True:
        return float(n // 2)
    if center is False or center is None:
        return 0.0
    return float(center)


# ---------------------------------------------------------------------------
# Chirp Z-transform
# ---------------------------------------------------------------------------

def _czt_last(x: torch.Tensor, n_out, w_phase, a_phase) -> torch.Tensor:
    """Bluestein CZT along the last axis."""
    n_in = x.shape[-1]
    m = n_in if n_out is None else int(n_out)
    w = -TWO_PI / m if w_phase is None else float(w_phase)
    a = float(a_phase)
    rdtype, device = _real_dtype(x.dtype), x.device
    n_fft = 1 << (n_in + m - 2).bit_length()        # power of two >= n_in + m - 1

    k = torch.arange(max(n_in, m), dtype=torch.float64)
    chirp = _phasor(w * k * k / 2, rdtype, device)   # exp(i w k^2 / 2)
    pre = chirp[:n_in]
    if a != 0.0:
        pre = pre * _phasor(-a * k[:n_in], rdtype, device)
    # n k = (n^2 + k^2 - (k - n)^2) / 2 turns the sum over n into a linear
    # convolution with exp(-i w j^2 / 2) for j = k - n in [-(n_in-1), m-1].
    kernel = torch.cat([chirp[1:n_in].flip(0), chirp[:m]]).conj()
    y = ifft(fft(x * pre, n=n_fft) * fft(kernel, n=n_fft))
    return y[..., n_in - 1 : n_in - 1 + m] * chirp[:m]


def czt(
    x: torch.Tensor,
    n_out=None,
    w_phase=None,
    a_phase=0.0,
    dim=-1,
) -> torch.Tensor:
    """Chirp Z-transform along ``dim``.

    Along each transformed axis, with ``N`` input samples::

        X[k] = sum_{n=0}^{N-1} x[n] exp(-i a n) exp(i w n k),   k = 0, ..., M-1

    Parameters
    ----------
    x : Tensor
        Real or complex input; other axes are batch axes.
    n_out : int or sequence of int, optional
        Number of output samples ``M`` per axis.  Default: input length.
    w_phase : float or sequence of float, optional
        Phase increment ``w`` in radians.  Default ``-2*pi/M``, which with
        ``M = N`` gives the unnormalised DFT (``torch.fft.fft(x, norm="backward")``).
    a_phase : float or sequence of float
        Starting phase ``a`` in radians.  Default 0.
    dim : int or sequence of int
        Axes to transform.  Default: last axis.  Scalar ``n_out``, ``w_phase``
        and ``a_phase`` apply to every axis; sequences are matched to ``dim``.

    Returns
    -------
    Tensor
        Complex tensor with the transformed axes resized to ``n_out``.
    """
    dims = _dims(dim, x.ndim)
    n = len(dims)
    for d, m, w, a in zip(dims, _per_dim(n_out, n, "n_out"),
                          _per_dim(w_phase, n, "w_phase"),
                          _per_dim(a_phase, n, "a_phase")):
        x = _czt_last(x.movedim(d, -1), m, w, a).movedim(-1, d)
    return x


# ---------------------------------------------------------------------------
# Zoomed FFT
# ---------------------------------------------------------------------------

def zoom_freq(
    n: int,
    k_start: float = 0.0,
    k_end: float = TWO_PI,
    include_end: bool = False,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """The ``n`` angular frequencies (radians per sample) that ``zoom_fft``
    evaluates over ``[k_start, k_end]``: ``k_start + step * m``.

    ``include_end=False`` samples like FFT bins (``step = span / n``, right
    end excluded); ``include_end=True`` puts the last sample exactly on
    ``k_end`` (``step = span / (n - 1)``).  The analogue of ``torch.fft.fftfreq``.
    """
    step = _step(k_start, k_end, n, include_end)
    return k_start + step * torch.arange(n, dtype=dtype, device=device)


def zoom_fft(
    x: torch.Tensor,
    n_out=None,
    k_start=0.0,
    k_end=TWO_PI,
    dim=-1,
    norm="ortho",
    center=False,
    include_end=False,
) -> torch.Tensor:
    """Zoomed FFT: the spectrum of ``x`` on the band ``[k_start, k_end]``.

    Along each transformed axis, with ``N`` input samples::

        X[m] = sum_{n=0}^{N-1} x[n] exp(-i w_m (n - c)),   m = 0, ..., M-1

    with ``w_m = zoom_freq(M, k_start, k_end, include_end)`` and ``c`` the
    origin index set by ``center``.  With default arguments this is exactly
    ``torch.fft.fft(x, norm="ortho")``.

    Parameters
    ----------
    x : Tensor
        Real or complex input; other axes are batch axes.
    n_out : int or sequence of int, optional
        Number of band samples ``M`` per axis.  Default: input length.
    k_start, k_end : float or sequence of float
        Band limits in radians per sample.  Default: the full circle
        ``[0, 2*pi)``.  A band ``[-pi, pi)`` gives the ``fftshift``-ed spectrum.
    dim : int or sequence of int
        Axes to transform.  Default: last axis.
    norm : {"ortho", "forward", "backward"}
        As in ``torch.fft``: ``"backward"`` leaves the forward transform
        unnormalised, ``"forward"`` divides it by the number of band samples
        ``prod(M)``, ``"ortho"`` by its square root.
    center : bool, float or sequence
        Where the origin of the input grid sits.  ``False``: sample 0 (the
        ``torch.fft`` convention).  ``True``: index ``N // 2`` (the
        ``torch.fft.fftshift`` convention; on the full band this equals
        ``fft(ifftshift(x))``).  A float pins the origin at that index, which
        may be fractional: ``(N - 1) / 2`` is the centre of a grid symmetric
        about zero.
    include_end : bool
        Sample ``k_end`` exactly (``M - 1`` steps) instead of excluding it.

    Returns
    -------
    Tensor
        Complex tensor with the transformed axes resized to ``n_out``.
    """
    _norm_factor(norm, 1, inverse=False)          # reject a bad norm early
    dims = _dims(dim, x.ndim)
    n = len(dims)
    rdtype, device = _real_dtype(x.dtype), x.device
    n_band = 1
    for d, m, k0, k1, c in zip(dims, _per_dim(n_out, n, "n_out"),
                               _per_dim(k_start, n, "k_start"),
                               _per_dim(k_end, n, "k_end"),
                               _per_dim(center, n, "center")):
        n_in = x.shape[d]
        m = n_in if m is None else int(m)
        step = _step(k0, k1, m, include_end)
        # sum_n x[n] exp(-i (k0 + step m) n)  ->  czt with a = k0, w = -step.
        y = _czt_last(x.movedim(d, -1), m, -step, k0)
        c = _origin(c, n_in)
        if c != 0.0:
            # exp(-i w_m (n - c)) = exp(+i c w_m) exp(-i w_m n)
            y = y * _phasor(c * zoom_freq(m, k0, k1, include_end), rdtype, device)
        x = y.movedim(-1, d)
        n_band *= m
    factor = _norm_factor(norm, n_band, inverse=False)
    return x if factor == 1.0 else x * factor


def zoom_ifft(
    x: torch.Tensor,
    n_out=None,
    k_start=0.0,
    k_end=TWO_PI,
    dim=-1,
    norm="ortho",
    center=False,
    include_end=False,
) -> torch.Tensor:
    """Zoomed inverse FFT: adjoint of ``zoom_fft`` over the same band.

    The input holds the ``M`` band samples along each transformed axis (the
    output grid of ``zoom_fft``, so the band step is set by the input length
    here) and the output has ``N = n_out`` samples::

        x[n] = sum_{m=0}^{M-1} X[m] exp(+i w_m (n - c)),   n = 0, ..., N-1

    This is the conjugate transpose of ``zoom_fft`` with the same ``k_start``,
    ``k_end``, ``center`` and ``include_end`` -- under mirrored ``norm``
    (``"backward"`` <-> ``"forward"``, ``"ortho"`` <-> ``"ortho"``), exactly as
    for ``torch.fft.fft``/``ifft``.  On a full band (``k_end = k_start + 2*pi``,
    ``include_end=False``, ``n_out = M``) it is also the exact inverse, and
    with default arguments it is ``torch.fft.ifft(x, norm="ortho")``.

    Parameters
    ----------
    x : Tensor
        Band samples along ``dim``; other axes are batch axes.
    n_out : int or sequence of int, optional
        Number of output (spatial) samples ``N`` per axis.  Default: input length.
    k_start, k_end, dim, include_end
        As passed to ``zoom_fft``.
    norm : {"ortho", "forward", "backward"}
        As in ``torch.fft``: ``"backward"`` divides this inverse transform by
        the number of band samples ``prod(M)``, ``"forward"`` leaves it
        unnormalised, ``"ortho"`` divides by the square root.
    center : bool, float or sequence
        Origin of the *output* grid, with the same meaning as in ``zoom_fft``
        (``True`` is ``N // 2``; on the full band this equals
        ``fftshift(ifft(X))``).
    """
    _norm_factor(norm, 1, inverse=True)           # reject a bad norm early
    dims = _dims(dim, x.ndim)
    n = len(dims)
    rdtype, device = _real_dtype(x.dtype), x.device
    n_band = 1
    for d, n_sp, k0, k1, c in zip(dims, _per_dim(n_out, n, "n_out"),
                                  _per_dim(k_start, n, "k_start"),
                                  _per_dim(k_end, n, "k_end"),
                                  _per_dim(center, n, "center")):
        m = x.shape[d]                              # band samples
        n_sp = m if n_sp is None else int(n_sp)
        step = _step(k0, k1, m, include_end)
        y = x.movedim(d, -1)
        c = _origin(c, n_sp)
        if c != 0.0:
            # exp(+i w_m (n - c)) = exp(-i c w_m) exp(+i w_m n): a ramp on the input.
            y = y * _phasor(-c * zoom_freq(m, k0, k1, include_end), rdtype, device)
        # sum_m X[m] exp(+i step m n)  ->  czt with w = +step; the k0 part of
        # w_m multiplies the *output* index n, so it is an output-side ramp.
        y = _czt_last(y, n_sp, step, 0.0)
        if k0 != 0.0:
            y = y * _phasor(k0 * torch.arange(n_sp, dtype=torch.float64), rdtype, device)
        x = y.movedim(-1, d)
        n_band *= m
    factor = _norm_factor(norm, n_band, inverse=True)
    return x if factor == 1.0 else x * factor
