"""``czt_plain`` is ``czt`` written out: same numbers, on every axis, for any band."""
import math

import pytest
import torch

from minimal_zoom_fft import czt, czt_plain

from test_czt import brute_czt

torch.manual_seed(0)
C64, C128 = torch.complex64, torch.complex128


@pytest.mark.parametrize("n,m", [(8, 8), (13, 7), (7, 13), (1, 5), (32, 1)])
@pytest.mark.parametrize("w,a", [(None, 0.0), (0.3, 0.0), (-0.05, 1.7), (2.0, -0.4)])
def test_plain_matches_fast_and_brute(n, m, w, a):
    x = torch.randn(3, n, dtype=C128)
    plain = czt_plain(x, m, w, a)
    assert torch.allclose(plain, czt(x, m, w, a), atol=1e-10)
    w_eff = -2 * math.pi / m if w is None else w
    assert torch.allclose(plain, brute_czt(x, m, w_eff, a), atol=1e-10)


def test_plain_on_two_axes():
    x = torch.randn(4, 6, 5, dtype=C128)
    assert torch.allclose(czt_plain(x, (9, 4), (0.2, -0.7), (0.1, 0.0), dim=(-2, -1)),
                          czt(x, (9, 4), (0.2, -0.7), (0.1, 0.0), dim=(-2, -1)), atol=1e-10)
    assert torch.allclose(czt_plain(x, dim=0), torch.fft.fft(x, dim=0), atol=1e-10)


def test_plain_keeps_precision_and_promotes_real():
    x = torch.randn(16, dtype=C64)
    assert czt_plain(x).dtype == C64
    assert torch.allclose(czt_plain(x), czt(x), atol=1e-5)
    assert czt_plain(torch.randn(16)).dtype == C64


@pytest.mark.parametrize("kwargs", [{"dim": 3}, {"n_out": (3, 3, 3), "dim": (0, 1)},
                                    {"dim": (0, -2)}])
def test_plain_refuses_what_fast_refuses(kwargs):
    x = torch.randn(4, 4, dtype=C64)
    with pytest.raises(ValueError) as fast:
        czt(x, **kwargs)
    with pytest.raises(ValueError) as plain:
        czt_plain(x, **kwargs)
    assert str(plain.value) == str(fast.value)


def test_plain_refuses_an_unsupported_dtype():
    x = torch.arange(8)
    with pytest.raises(TypeError, match="dtype"):
        czt_plain(x)
