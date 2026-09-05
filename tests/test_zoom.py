"""Tests for ``zoom_fft`` / ``zoom_ifft`` / ``zoom_freq``.

References are written from the *definitions* (explicit summation in
float64), not from the implementation, so they can disagree with it.
"""
import math

import pytest
import torch

from minimal_zoom_fft import zoom_fft, zoom_ifft, zoom_freq

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

    def test_shifted_full_band_is_fftshift(self):
        """The band ``[-pi, pi)`` is the ``fftshift``-ed spectrum."""
        x = torch.randn(8, dtype=C128)
        got = zoom_fft(x, k_start=-PI, k_end=PI)
        assert torch.allclose(got, torch.fft.fftshift(torch.fft.fft(x, norm="ortho")), atol=TOL)

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
    def test_half_pixel_origin_matches_psf_generator(self, n_in, n_out, k0, k1):
        """``center=(N-1)/2`` reproduces ``psf_generator``'s ``custom_ifft2``
        with ``fftshift_input=True``: origin at the geometric centre of the
        grid, between two samples for even ``N``."""
        X = torch.randn(n_in, n_in, dtype=C64)
        c = (n_in - 1) / 2
        ref = inv2(X, (n_out, n_out), k0, k1, include_end=True, c=(c, c))
        got = zoom_ifft(X, (n_out, n_out), k0, k1, dim=(-2, -1), norm="forward",
                        center=c, include_end=True)
        assert torch.allclose(got.to(C128), ref, atol=1e-4)

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
