"""Tests for ``zoom_fft`` / ``zoom_ifft`` / ``zoom_freq``.

References are written from the *definitions* (explicit summation in
float64), not from the implementation, so they can disagree with it.
"""
import math

import pytest
import torch

from minimal_zoom_fft import czt, czt_plain, zoom_fft, zoom_ifft, zoom_freq

torch.manual_seed(0)
PI = math.pi
TOL = 1e-10
C64, C128, F32, F64 = torch.complex64, torch.complex128, torch.float32, torch.float64
NORMS = ["ortho", "forward", "backward"]


# ---------------------------------------------------------------------------
# Brute-force references
# ---------------------------------------------------------------------------

def band(n, k0, k1, include_end):
    if include_end:
        step = (k1 - k0) / (n - 1) if n > 1 else 0.0
    else:
        step = (k1 - k0) / n
    return k0 + step * torch.arange(n, dtype=F64)


def fwd_matrix(n_in, n_out, k0, k1, include_end=False, c=0.0):
    """``E[m, n] = exp(-i w_m (n - c))`` so that ``X = E @ x``."""
    w = band(n_out, k0, k1, include_end)
    n = torch.arange(n_in, dtype=F64) - c
    return torch.exp(-1j * w[:, None] * n[None, :])


def inv_matrix(n_band, n_out, k0, k1, include_end=False, c=0.0):
    """``E[n, m] = exp(+i w_m (n - c))`` so that ``x = E @ X``."""
    w = band(n_band, k0, k1, include_end)
    n = torch.arange(n_out, dtype=F64) - c
    return torch.exp(1j * w[None, :] * n[:, None])


def fwd2(x, shape_out, k0, k1, include_end=False, c=(0.0, 0.0)):
    e1 = fwd_matrix(x.shape[-2], shape_out[0], k0, k1, include_end, c[0])
    e2 = fwd_matrix(x.shape[-1], shape_out[1], k0, k1, include_end, c[1])
    return e1 @ x.to(C128) @ e2.T


def inv2(X, shape_out, k0, k1, include_end=False, c=(0.0, 0.0)):
    e1 = inv_matrix(X.shape[-2], shape_out[0], k0, k1, include_end, c[0])
    e2 = inv_matrix(X.shape[-1], shape_out[1], k0, k1, include_end, c[1])
    return e1 @ X.to(C128) @ e2.T


def dot(a, b):
    return complex((a.conj() * b).sum())


# ---------------------------------------------------------------------------
# Full band == torch.fft
# ---------------------------------------------------------------------------

class TestFullBandIsTorchFft:

    @pytest.mark.parametrize("norm", NORMS + [None])
    def test_1d(self, norm):
        x = torch.randn(9, dtype=C128)
        assert torch.allclose(zoom_fft(x, norm=norm), torch.fft.fft(x, norm=norm), atol=TOL)
        assert torch.allclose(zoom_ifft(x, norm=norm), torch.fft.ifft(x, norm=norm), atol=TOL)

    @pytest.mark.parametrize("norm", NORMS)
    @pytest.mark.parametrize("shape", [(9, 9), (8, 12)])
    def test_2d(self, shape, norm):
        x = torch.randn(*shape, dtype=C128)
        d = (-2, -1)
        assert torch.allclose(zoom_fft(x, dim=d, norm=norm), torch.fft.fft2(x, norm=norm), atol=TOL)
        assert torch.allclose(zoom_ifft(x, dim=d, norm=norm), torch.fft.ifft2(x, norm=norm), atol=TOL)

    def test_default_norm_is_ortho(self):
        x = torch.randn(16, dtype=C128)
        assert torch.allclose(zoom_fft(x), torch.fft.fft(x, norm="ortho"), atol=TOL)

    @pytest.mark.parametrize("n", [8, 9])
    def test_shifted_full_band_is_fftshift(self, n):
        """The full band starting at ``-2 pi (N // 2) / N`` is the
        ``fftshift``-ed spectrum.  That is ``[-pi, pi)`` for even ``N``; for odd
        ``N`` the ``fftshift``-ed bins sit half a bin above ``[-pi, pi)``."""
        x = torch.randn(n, dtype=C128)
        ref = torch.fft.fftshift(torch.fft.fft(x, norm="ortho"))
        bins = 2 * PI * torch.fft.fftshift(torch.fft.fftfreq(n, dtype=F64))
        k0 = -2 * PI * (n // 2) / n
        assert torch.allclose(zoom_fft(x, k_start=k0, k_end=k0 + 2 * PI), ref, atol=TOL)
        assert torch.allclose(zoom_freq(n, k0, k0 + 2 * PI), bins, atol=TOL)
        half_bin = PI / n if n % 2 else 0.0
        assert torch.allclose(zoom_freq(n, -PI, PI) + half_bin, bins, atol=TOL)
        if n % 2 == 0:
            assert torch.allclose(zoom_fft(x, k_start=-PI, k_end=PI), ref, atol=TOL)

    def test_zoom_freq_default_is_fftfreq(self):
        freqs = torch.remainder(2 * PI * torch.fft.fftfreq(8, dtype=F64), 2 * PI)
        assert torch.allclose(zoom_freq(8), freqs, atol=TOL)

    def test_zoom_freq_include_end(self):
        assert torch.allclose(zoom_freq(5, -1.0, 1.0, include_end=True),
                              torch.linspace(-1.0, 1.0, 5, dtype=F64), atol=TOL)
        assert zoom_freq(1, 0.7, 2.1, include_end=True).item() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Zoomed bands against explicit summation
# ---------------------------------------------------------------------------

class TestZoomMatchesBruteForce:
    K0, K1 = -PI / 3, PI / 3

    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("n_out", [12, 7, 16])
    def test_1d_forward(self, n_out, include_end):
        x = torch.randn(12, dtype=C128)
        ref = fwd_matrix(12, n_out, self.K0, self.K1, include_end) @ x
        got = zoom_fft(x, n_out, self.K0, self.K1, norm="backward", include_end=include_end)
        assert torch.allclose(got, ref, atol=TOL)

    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("shape_out", [(12, 12), (12, 7), (5, 16)])
    def test_2d_forward(self, shape_out, include_end):
        """Non-square outputs pin that *each* axis spans the whole band."""
        x = torch.randn(12, 12, dtype=C128)
        ref = fwd2(x, shape_out, self.K0, self.K1, include_end)
        got = zoom_fft(x, shape_out, self.K0, self.K1, dim=(-2, -1), norm="backward",
                       include_end=include_end)
        assert torch.allclose(got, ref, atol=TOL)

    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("n_out", [12, 7, 16])
    def test_1d_inverse(self, n_out, include_end):
        X = torch.randn(9, dtype=C128)
        ref = inv_matrix(9, n_out, self.K0, self.K1, include_end) @ X
        got = zoom_ifft(X, n_out, self.K0, self.K1, norm="forward", include_end=include_end)
        assert torch.allclose(got, ref, atol=TOL)

    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("shape_out", [(12, 12), (12, 7), (5, 16)])
    def test_2d_inverse(self, shape_out, include_end):
        X = torch.randn(9, 11, dtype=C128)
        ref = inv2(X, shape_out, self.K0, self.K1, include_end)
        got = zoom_ifft(X, shape_out, self.K0, self.K1, dim=(-2, -1), norm="forward",
                        include_end=include_end)
        assert torch.allclose(got, ref, atol=TOL)

    def test_per_axis_bands(self):
        x = torch.randn(10, 12, dtype=C128)
        k0, k1 = (-1.0, 0.3), (1.0, 2.0)
        e1 = fwd_matrix(10, 8, k0[0], k1[0])
        e2 = fwd_matrix(12, 9, k0[1], k1[1])
        got = zoom_fft(x, (8, 9), k0, k1, dim=(-2, -1), norm="backward")
        assert torch.allclose(got, e1 @ x @ e2.T, atol=TOL)

    def test_single_sample_band_with_include_end(self):
        x = torch.randn(6, dtype=C128)
        got = zoom_fft(x, 1, 0.7, 2.1, norm="backward", include_end=True)
        assert torch.allclose(got, fwd_matrix(6, 1, 0.7, 0.7) @ x, atol=TOL)

    def test_non_last_axis_and_batch(self):
        x = torch.randn(3, 10, 5, dtype=C128)
        got = zoom_fft(x, 7, self.K0, self.K1, dim=1, norm="backward")
        assert got.shape == (3, 7, 5)
        ref = fwd_matrix(10, 7, self.K0, self.K1) @ x[2, :, 3]
        assert torch.allclose(got[2, :, 3], ref, atol=TOL)


# ---------------------------------------------------------------------------
# ``center``: where the origin of the spatial grid sits
# ---------------------------------------------------------------------------

class TestCenter:

    @pytest.mark.parametrize("k0", [0.0, -PI / 3])
    @pytest.mark.parametrize("shape", [(12, 12), (13, 13), (8, 16)])
    def test_true_means_n_half(self, shape, k0):
        """``center=True`` is the ``torch.fft.fftshift`` convention, ``n // 2``.
        Even sizes are the interesting case (half a pixel off the midpoint)."""
        x = torch.randn(*shape, dtype=C128)
        k1 = k0 + 2 * PI / 3
        c = (shape[0] // 2, shape[1] // 2)
        ref = fwd2(x, shape, k0, k1, include_end=True, c=c)
        got = zoom_fft(x, k_start=k0, k_end=k1, dim=(-2, -1), norm="backward",
                       center=True, include_end=True)
        assert torch.allclose(got, ref, atol=TOL)

    @pytest.mark.parametrize("n", [8, 9])
    def test_true_on_full_band_is_fft_of_ifftshift(self, n):
        x = torch.randn(n, n, dtype=C128)
        d = (-2, -1)
        assert torch.allclose(zoom_fft(x, dim=d, center=True),
                              torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho"), atol=TOL)

    @pytest.mark.parametrize("norm", NORMS)
    @pytest.mark.parametrize("n", [8, 9])
    def test_true_on_full_band_inverse_is_fftshift_of_ifft(self, n, norm):
        X = torch.randn(n, n, dtype=C128)
        d = (-2, -1)
        assert torch.allclose(zoom_ifft(X, dim=d, center=True, norm=norm),
                              torch.fft.fftshift(torch.fft.ifft2(X, norm=norm)), atol=TOL)

    @pytest.mark.parametrize("n_in,n_out,k0,k1", [
        (9, 7, -1.3, 1.3),   # odd input, symmetric range
        (8, 6, -0.9, 1.4),   # even input, asymmetric range
        (9, 9, 0.2, 2.5),    # one-sided range
    ])
    def test_float_origin_inverse(self, n_in, n_out, k0, k1):
        """A fractional origin in ``zoom_ifft``, in complex64, against the
        explicit sum."""
        X = torch.randn(n_in, n_in, dtype=C64)
        c = (n_in - 1) / 2
        ref = inv2(X, (n_out, n_out), k0, k1, include_end=True, c=(c, c))
        got = zoom_ifft(X, (n_out, n_out), k0, k1, dim=(-2, -1), norm="forward",
                        center=c, include_end=True)
        assert torch.allclose(got.to(C128), ref, atol=1e-4)

    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("n_in,n_out,k0,k1", [
        (9, 7, -1.3, 1.3),   # odd input, symmetric range
        (8, 6, -0.9, 1.4),   # even input, asymmetric range
        (9, 9, 0.2, 2.5),    # one-sided range
    ])
    def test_conjugate_kernel_is_the_negated_band(self, n_in, n_out, k0, k1, include_end):
        """``X[m] = sum_n x[n] exp(+i w_m (n - c))`` on the band ``[k0, k1]``
        with ``c = (N - 1) / 2``, the sum that ``psf_generator``'s
        ``custom_ifft2`` takes from a centred pupil, is ``zoom_fft`` on the
        band ``[-k0, -k1]`` with the norm mirrored."""
        x = torch.randn(n_in, n_in, dtype=C128)
        c = (n_in - 1) / 2
        e = fwd_matrix(n_in, n_out, k0, k1, include_end, c).conj()      # exp(+i w_m (n - c))
        ref = e @ x @ e.T
        kw = dict(n_out=(n_out, n_out), k_start=-k0, k_end=-k1, dim=(-2, -1),
                  center=c, include_end=include_end)
        assert torch.allclose(zoom_fft(x, norm="backward", **kw), ref, atol=TOL)
        assert torch.allclose(zoom_fft(x, norm="forward", **kw), ref / n_out ** 2, atol=TOL)

    def test_float_origin_forward(self):
        x = torch.randn(10, dtype=C128)
        c = 4.5
        ref = fwd_matrix(10, 7, -1.0, 1.0, include_end=True, c=c) @ x
        got = zoom_fft(x, 7, -1.0, 1.0, norm="backward", center=c, include_end=True)
        assert torch.allclose(got, ref, atol=TOL)

    def test_per_axis_center(self):
        x = torch.randn(8, 9, dtype=C128)
        ref = fwd2(x, (6, 6), -1.0, 1.0, include_end=True, c=(4, 2.5))
        got = zoom_fft(x, (6, 6), -1.0, 1.0, dim=(-2, -1), norm="backward",
                       center=(True, 2.5), include_end=True)
        assert torch.allclose(got, ref, atol=TOL)


# ---------------------------------------------------------------------------
# zoom_ifft is the adjoint of zoom_fft, and its inverse on a full band
# ---------------------------------------------------------------------------

class TestAdjointAndInverse:

    @pytest.mark.parametrize("norm", NORMS)
    @pytest.mark.parametrize("k0", [0.0, -PI, 0.7])
    def test_full_band_roundtrip_any_k_start(self, k0, norm):
        x = torch.randn(16, 16, dtype=C128)
        kw = dict(k_start=k0, k_end=k0 + 2 * PI, dim=(-2, -1), norm=norm)
        assert torch.allclose(zoom_ifft(zoom_fft(x, **kw), **kw), x, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize("k0", [0.0, -PI])
    @pytest.mark.parametrize("shape", [(8, 8), (9, 9), (8, 9), (13, 7)])
    def test_centered_roundtrip_full_band(self, shape, k0):
        x = torch.randn(*shape, dtype=C128)
        kw = dict(k_start=k0, k_end=k0 + 2 * PI, dim=(-2, -1), center=True)
        assert torch.allclose(zoom_ifft(zoom_fft(x, **kw), **kw), x, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize("center", [False, True, 3.5])
    @pytest.mark.parametrize("include_end", [False, True])
    @pytest.mark.parametrize("shape_in,shape_out", [((12, 12), (12, 7)), ((9, 13), (6, 11))])
    def test_dot_test_on_zoom_band(self, shape_in, shape_out, include_end, center):
        """``<F x, y> == <x, F^H y>`` on a band that is not a full circle."""
        x = torch.randn(*shape_in, dtype=C128)
        y = torch.randn(*shape_out, dtype=C128)
        kw = dict(k_start=-PI / 3, k_end=PI / 2, dim=(-2, -1),
                  include_end=include_end, center=center)
        lhs = dot(zoom_fft(x, shape_out, norm="backward", **kw), y)
        rhs = dot(x, zoom_ifft(y, shape_in, norm="forward", **kw))
        assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))

    @pytest.mark.parametrize("norm_fwd,norm_inv",
                             [("backward", "forward"), ("ortho", "ortho"), ("forward", "backward")])
    def test_dot_test_under_mirrored_norms(self, norm_fwd, norm_inv):
        x = torch.randn(10, 14, dtype=C128)
        y = torch.randn(7, 9, dtype=C128)
        kw = dict(k_start=0.3, k_end=1.9, dim=(-2, -1), include_end=True)
        lhs = dot(zoom_fft(x, (7, 9), norm=norm_fwd, **kw), y)
        rhs = dot(x, zoom_ifft(y, (10, 14), norm=norm_inv, **kw))
        assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))

    def test_1d_dot_test(self):
        x, y = torch.randn(11, dtype=C128), torch.randn(5, dtype=C128)
        kw = dict(k_start=-0.4, k_end=0.9, center=True)
        lhs = dot(zoom_fft(x, 5, norm="ortho", **kw), y)
        rhs = dot(x, zoom_ifft(y, 11, norm="ortho", **kw))
        assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))


# ---------------------------------------------------------------------------
# dtypes, autograd, errors
# ---------------------------------------------------------------------------

class TestDtypesAndErrors:

    @pytest.mark.parametrize("dtype,expected", [(C64, C64), (F32, C64), (C128, C128), (F64, C128)])
    def test_dtype_preserved(self, dtype, expected):
        x = torch.randn(8, dtype=dtype)
        assert zoom_fft(x, 5, -1.0, 1.0).dtype == expected
        assert zoom_ifft(x, 5, -1.0, 1.0).dtype == expected

    def test_float32_matches_float64(self):
        x = torch.randn(512, 512, dtype=C128)
        kw = dict(n_out=(300, 200), k_start=-0.5, k_end=0.8, dim=(-2, -1), center=True)
        ref, got = zoom_fft(x, **kw), zoom_fft(x.to(C64), **kw)
        assert (got.to(C128) - ref).norm() / ref.norm() < 1e-5

    def test_autograd(self):
        x = torch.randn(16, dtype=C64, requires_grad=True)
        (zoom_fft(x, 24, -1.0, 1.0, center=True).abs() ** 2).sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_unknown_norm_raises(self):
        x = torch.randn(8, dtype=C128)
        with pytest.raises(ValueError, match="norm"):
            zoom_fft(x, norm="nope")
        with pytest.raises(ValueError, match="norm"):
            zoom_ifft(x, norm="nope")

    def test_out_of_range_dim_raises(self):
        x = torch.randn(8, dtype=C128)
        with pytest.raises(ValueError, match="out of range"):
            zoom_fft(x, dim=1)
        with pytest.raises(ValueError, match="out of range"):
            zoom_ifft(x, dim=-2)

    @pytest.mark.parametrize("transform", [zoom_fft, zoom_ifft])
    def test_non_tensor_input_names_its_type(self, transform):
        with pytest.raises(TypeError, match="x must be a torch.Tensor, got list"):
            transform([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("transform", [zoom_fft, zoom_ifft])
    @pytest.mark.parametrize("n_out", [0, -3])
    def test_n_out_below_one_raises(self, n_out, transform):
        x = torch.randn(8, dtype=C128)
        with pytest.raises(ValueError, match="n_out.*at least 1"):
            transform(x, n_out)
        with pytest.raises(ValueError, match="n_out.*at least 1"):
            transform(x, n_out, include_end=True, norm="backward")

    @pytest.mark.parametrize("transform", [zoom_fft, zoom_ifft])
    @pytest.mark.parametrize("n_out", [5.7, "5"])
    def test_non_integer_n_out_raises(self, n_out, transform):
        with pytest.raises(TypeError, match="n_out.*not an integer"):
            transform(torch.randn(8, dtype=C128), n_out)

    @pytest.mark.parametrize("include_end", [(False, False), 1, 0, None, "no"])
    def test_include_end_must_be_a_bool(self, include_end):
        """A tuple would count as True on every axis, so anything but a bool is refused."""
        x = torch.randn(6, 5, dtype=C128)
        for transform in (zoom_fft, zoom_ifft):
            with pytest.raises(TypeError, match="include_end"):
                transform(x, (4, 7), -0.5, 0.5, dim=(-2, -1), include_end=include_end)
        with pytest.raises(TypeError, match="include_end"):
            zoom_freq(5, -1.0, 1.0, include_end=include_end)

    def test_zoom_freq_needs_a_positive_integer(self):
        with pytest.raises(ValueError, match="n=0 must be at least 1"):
            zoom_freq(0)
        with pytest.raises(ValueError, match="n=0 must be at least 1"):
            zoom_freq(0, include_end=True)
        with pytest.raises(TypeError, match="n=4.0 is not an integer"):
            zoom_freq(4.0)


# ---------------------------------------------------------------------------
# Default device
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_runs_under_an_mps_default_device():
    """The float64 tables are built on the CPU, since MPS has no float64, so
    ``torch.set_default_device("mps")`` leaves every function working."""
    x = torch.randn(16, dtype=C64)
    calls = [lambda t: zoom_fft(t, 24, -0.5, 0.5, center=True),
             lambda t: zoom_ifft(t, 24, 0.3, 1.5, center=True),
             lambda t: czt(t, 9, 0.3, -0.2),
             lambda t: czt_plain(t, 9, 0.3, -0.2)]
    refs = [call(x) for call in calls]
    previous = torch.empty(0).device
    try:
        torch.set_default_device("mps")
        outs = [call(x.to("mps")) for call in calls]
        freq = zoom_freq(8)
    finally:
        torch.set_default_device(previous)
    assert freq.device.type == "cpu" and freq.dtype == F64
    for got, ref in zip(outs, refs):
        assert got.device.type == "mps"
        assert (got.cpu() - ref).abs().max() < 1e-4 * ref.abs().max()
