"""Tests for ``minimal_fft.czt`` against torch.fft and explicit summation."""
import math

import pytest
import torch

from minimal_fft import czt

torch.manual_seed(0)
TOL = 1e-10          # float64 tests
C64, C128, F32, F64 = torch.complex64, torch.complex128, torch.float32, torch.float64


def brute_czt(x, n_out, w, a):
    """1-D reference along the last axis, by explicit summation in float64:
    ``X[k] = sum_n x[n] exp(-i a n) exp(i w n k)``."""
    n = torch.arange(x.shape[-1], dtype=F64)
    k = torch.arange(n_out, dtype=F64)
    kernel = torch.exp(-1j * a * n[None, :]) * torch.exp(1j * w * n[None, :] * k[:, None])
    return x.to(C128) @ kernel.T


class TestDefaultsAreTheDft:
    """``czt(x)`` with defaults is the unnormalised DFT along ``dim``."""

    def test_1d(self):
        x = torch.randn(32, dtype=C128)
        assert torch.allclose(czt(x), torch.fft.fft(x), atol=TOL)

    def test_2d(self):
        x = torch.randn(16, 16, dtype=C128)
        assert torch.allclose(czt(x, dim=(-2, -1)), torch.fft.fft2(x), atol=TOL)

    def test_2d_rectangular(self):
        x = torch.randn(12, 20, dtype=C128)
        assert torch.allclose(czt(x, dim=(-2, -1)), torch.fft.fft2(x), atol=TOL)

    def test_non_last_axis(self):
        x = torch.randn(7, 5, dtype=C128)
        assert torch.allclose(czt(x, dim=0), torch.fft.fft(x, dim=0), atol=TOL)

    def test_batch_axes_untouched(self):
        x = torch.randn(3, 4, 16, dtype=C128)
        out = czt(x, n_out=20)
        assert out.shape == (3, 4, 20)
        assert torch.allclose(out[1, 2], czt(x[1, 2], n_out=20), atol=TOL)

    def test_real_input_gives_complex_output(self):
        x = torch.randn(16, dtype=F64)
        assert torch.allclose(czt(x), torch.fft.fft(x), atol=TOL)


class TestGeneralParameters:

    @pytest.mark.parametrize("n_in,n_out", [(16, 40), (32, 10), (1, 5), (5, 1), (13, 13)])
    def test_matches_brute_force(self, n_in, n_out):
        x = torch.randn(n_in, dtype=C128)
        w, a = 0.37, -1.1
        assert torch.allclose(czt(x, n_out, w, a), brute_czt(x, n_out, w, a), atol=TOL)

    def test_per_axis_parameters(self):
        x = torch.randn(9, 11, dtype=C128)
        both = czt(x, n_out=(6, 14), w_phase=(-0.2, 0.3), a_phase=(0.5, -0.1), dim=(0, 1))
        rows = czt(x, n_out=6, w_phase=-0.2, a_phase=0.5, dim=0)
        cols = czt(rows, n_out=14, w_phase=0.3, a_phase=-0.1, dim=1)
        assert torch.allclose(both, cols, atol=TOL)

    def test_scalar_parameters_broadcast(self):
        x = torch.randn(8, 8, dtype=C128)
        w = -0.4
        assert torch.allclose(czt(x, w_phase=w, dim=(-2, -1)),
                              czt(x, w_phase=(w, w), dim=(-2, -1)), atol=TOL)

    def test_zoom_is_a_subset_of_dft_bins(self):
        """Half the circle sampled with half the points: the first N//2 bins."""
        n = 32
        x = torch.randn(n, dtype=C128)
        out = czt(x, n_out=n // 2, w_phase=-2 * math.pi / n)
        assert torch.allclose(out, torch.fft.fft(x)[: n // 2], atol=TOL)

    def test_linearity(self):
        x, y = torch.randn(20, dtype=C128), torch.randn(20, dtype=C128)
        a, b = 2.0 + 1j, -0.5 + 0.3j
        assert torch.allclose(czt(a * x + b * y, 25), a * czt(x, 25) + b * czt(y, 25), atol=TOL)


class TestDtypesAndAutograd:

    @pytest.mark.parametrize("dtype,expected", [(C64, C64), (F32, C64), (C128, C128), (F64, C128)])
    def test_dtype_preserved(self, dtype, expected):
        assert czt(torch.randn(8, dtype=dtype), 12).dtype == expected

    def test_float32_accuracy_for_large_transforms(self):
        """Chirp phases are reduced modulo 2*pi in float64, so float32 stays
        accurate even when ``w k^2`` reaches thousands of radians."""
        n_in, n_out = 4096, 3000
        x = torch.randn(n_in, dtype=C128)
        w, a = -2 * math.pi * 0.3 / n_out, 0.2
        ref = czt(x, n_out, w, a)
        got = czt(x.to(C64), n_out, w, a)
        rel = (got.to(C128) - ref).norm() / ref.norm()
        assert rel < 1e-5, f"relative error {rel:.2e}"

    def test_autograd(self):
        x = torch.randn(16, dtype=C64, requires_grad=True)
        (czt(x, 20).abs() ** 2).sum().backward()
        assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()

    def test_unsupported_dtype_raises(self):
        with pytest.raises(TypeError, match="dtype"):
            czt(torch.arange(8))


class TestArgumentErrors:

    def test_duplicate_dims_raise(self):
        with pytest.raises(ValueError, match="twice"):
            czt(torch.randn(4, 4, dtype=C64), dim=(0, -2))

    def test_wrong_parameter_length_raises(self):
        with pytest.raises(ValueError, match="n_out"):
            czt(torch.randn(4, 4, dtype=C64), n_out=(3, 3, 3), dim=(0, 1))
