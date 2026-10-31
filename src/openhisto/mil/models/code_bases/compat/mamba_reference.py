"""Differentiable Torch equivalents of Mamba's fused kernels.

Selective scan follows the reference implementation by Tri Dao and Albert Gu
(Copyright 2023, Apache-2.0, vendored MambaMIL/mamba). The causal convolution
and projections implement the same fused operation with ordinary Torch ops.
No sequence subsampling, state truncation, or approximate attention is used.
The SRMamba caller retains its native transpose/pad/inverse-transpose branch.
This path prioritizes portability over CUDA-kernel speed.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    variable_B, variable_C = B.ndim >= 3, C.ndim >= 3
    if A.is_complex():
        if variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (l two) -> ... l two", two=2).contiguous()
            )
        if variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (l two) -> ... l two", two=2).contiguous()
            )
    else:
        B, C = B.float(), C.float()
    if variable_B and B.ndim == 4:
        B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
    if variable_C and C.ndim == 4:
        C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])
    state = A.new_zeros((batch, dim, A.shape[1]))
    outputs = []
    # Compute one time step at a time, avoiding reference [B,D,L,state] buffers.
    for t in range(length):
        transition = torch.exp(delta[:, :, t, None] * A)
        bt = B if not variable_B else B[..., t]
        if variable_B and B.ndim == 3:
            bt = bt[:, None, :]
        state = transition * state + delta[:, :, t, None] * bt * u[:, :, t, None]
        ct = C if not variable_C else C[..., t]
        if variable_C and C.ndim == 3:
            ct = ct[:, None, :]
        y = (state * ct).sum(-1)
        if y.is_complex():
            y = y.real * 2
        outputs.append(y)
    result = torch.stack(outputs, dim=2)
    if D is not None:
        result = result + u * D[:, None]
    if z is not None:
        result = result * F.silu(z)
    result = result.to(dtype_in)
    return (result, state) if return_last_state else result


selective_scan_fn = selective_scan_ref


def mamba_inner_fn_no_out_proj(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    checkpoint_lvl=1,
):
    """Unfused causal-conv, input projection, scan and SiLU gate [B,D,L]."""
    x, z = xz.chunk(2, dim=1)
    length = x.shape[-1]
    weight = conv1d_weight if conv1d_weight.ndim == 3 else conv1d_weight[:, None, :]
    x = F.silu(
        F.conv1d(x, weight, conv1d_bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[
            ..., :length
        ]
    )
    projected = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
    rank = delta_proj_weight.shape[1]
    state_dim = A.shape[-1] * (2 if A.is_complex() else 1)
    delta = rearrange(
        F.linear(projected[:, :rank], delta_proj_weight), "(b l) d -> b d l", l=length
    )
    if B is None:
        B = projected[:, rank : rank + state_dim]
        if B_proj_bias is not None:
            B = B + B_proj_bias
        B = (
            rearrange(B, "(b l) (n two) -> b n (l two)", l=length, two=2)
            if A.is_complex()
            else rearrange(B, "(b l) n -> b n l", l=length)
        )
    if C is None:
        C = projected[:, -state_dim:]
        if C_proj_bias is not None:
            C = C + C_proj_bias
        C = (
            rearrange(C, "(b l) (n two) -> b n (l two)", l=length, two=2)
            if A.is_complex()
            else rearrange(C, "(b l) n -> b n l", l=length)
        )
    return selective_scan_ref(x, delta, A, B, C, D, z, delta_bias, delta_softplus)


def mamba_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    checkpoint_lvl=1,
):
    out = mamba_inner_fn_no_out_proj(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
        checkpoint_lvl,
    )
    return F.linear(out.transpose(1, 2), out_proj_weight, out_proj_bias)
