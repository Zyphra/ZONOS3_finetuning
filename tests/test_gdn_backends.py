"""The FLA Triton path of the GDN mixer against the plain recurrence (CUDA + fla-core only)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GDN_HEADS, GDN_INNER, GDNMixer, Zonos3Args, fla_available  # noqa: E402

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and fla_available()), reason="needs CUDA and fla-core")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_matches_recurrence(dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    args = Zonos3Args(dim=96)
    mixer = GDNMixer(args).to(device, dtype)
    with torch.no_grad():
        mixer.reset_parameters()
        mixer.A_log.normal_(std=0.5)
        mixer.dt_bias.normal_(std=0.5)
    lengths = [70, 130, 1, 64]
    x = torch.randn(sum(lengths), 96, device=device, dtype=dtype)
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    mixed = mixer.in_proj(x)
    qkv, z, a, b = mixed.split([3 * GDN_INNER, GDN_INNER, GDN_HEADS, GDN_HEADS], dim=-1)
    fla_out = mixer._forward_fla(x, qkv, z, a, b, cu)
    torch_out = mixer._forward_torch(x, qkv, z, a, b, cu)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(fla_out.float(), torch_out.float(), atol=tolerance, rtol=tolerance)


def test_fla_backward_runs():
    torch.manual_seed(1)
    device = torch.device("cuda")
    mixer = GDNMixer(Zonos3Args(dim=64, gdn_backend="fla")).to(device, torch.bfloat16)
    with torch.no_grad():
        mixer.reset_parameters()
        mixer.A_log.normal_(std=0.5)
        mixer.dt_bias.normal_(std=0.5)
    x = torch.randn(200, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 90, 200], dtype=torch.int32, device=device)
    out = mixer(x, cu)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in mixer.parameters())
