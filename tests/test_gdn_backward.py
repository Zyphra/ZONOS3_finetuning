"""Backward parity of the FLA / TileLang GDN path against autograd through the plain recurrence, over sequence
lengths around the 64-row kernel chunk (CUDA + fla-core only)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GDN_HEADS, GDN_INNER, GDNMixer, Zonos3Args, fla_available  # noqa: E402

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and fla_available()), reason="needs CUDA and fla-core")


def _grads(mixer, x, cu, backend):
    x = x.detach().clone().requires_grad_(True)
    mixer.zero_grad(set_to_none=True)
    mixed = mixer.in_proj(x)
    qkv, z, a, b = mixed.split([3 * GDN_INNER, GDN_INNER, GDN_HEADS, GDN_HEADS], dim=-1)
    out = mixer._forward_fla(x, qkv, z, a, b, cu) if backend == "fla" else mixer._forward_torch(x, qkv, z, a, b, cu)
    (out.float() * torch.linspace(-1, 1, out.shape[-1], device=x.device)).sum().backward()
    params = {n: p.grad.detach().float().clone() for n, p in mixer.named_parameters()}
    return out.detach().float(), x.grad.detach().float().clone(), params


@pytest.mark.parametrize(
    "lengths",
    [[1], [5], [63], [64], [65], [200], [70, 130, 1, 64], [17, 3, 129, 250], [511, 1]],
)
def test_backward_matches_recurrence(lengths):
    torch.manual_seed(0)
    device = torch.device("cuda")
    mixer = GDNMixer(Zonos3Args(dim=96)).to(device, torch.float32)
    with torch.no_grad():
        mixer.reset_parameters()
        mixer.A_log.normal_(std=0.5)
        mixer.dt_bias.normal_(std=0.5)
    x = torch.randn(sum(lengths), 96, device=device)
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    out_fla, dx_fla, p_fla = _grads(mixer, x, cu, "fla")
    out_ref, dx_ref, p_ref = _grads(mixer, x, cu, "torch")
    torch.testing.assert_close(out_fla, out_ref, atol=2e-3, rtol=2e-3)
    def close(candidate, reference):  # relative to the largest reference entry, with an absolute floor
        return float((candidate - reference).abs().max()) <= 2e-2 * float(reference.abs().max()) + 1e-4

    assert close(dx_fla, dx_ref), f"input gradient mismatch for lengths {lengths}"
    for name in p_ref:
        assert close(p_fla[name], p_ref[name]), f"{name} gradient mismatch for {lengths}"
