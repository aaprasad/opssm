# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable CAUSAL context encoders for the Zakai-filter DeepONet.

The operator conditions on a per-timestep context ctx_t summarizing the observations. For the FORWARD
filter that summary must be strictly CAUSAL (ctx_t depends only on y_{0:t}); the BACKWARD/smoother
operator is the anti-causal twin, obtained GENERICALLY by flip(0) -> causal encoder -> flip(0) (handled
in OperatorFilter.context via the `reverse` flag), so every encoder is implemented ONCE as a causal
module here.

Contract (every encoder): forward(inp: (T,B,in_dim)) -> (T,B,ctx_dim), strictly causal in T. No
normalization or pooling ACROSS the time axis (that would leak the future). The observation mask is just
the last input channel (the caller packs inp = [obs*mask, mask]).

Select via config: `model.encoder=<name>` with optional `model.encoder_kwargs={...}`. Default `gru` is
byte-identical to the original hard-coded nn.GRU encoder.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

ENCODERS = {}  # name -> CausalEncoder subclass


def register(name):
    def deco(cls):
        ENCODERS[name] = cls
        return cls
    return deco


def make_encoder(name, in_dim, ctx_dim, hidden=64, **kwargs):
    """Factory: build a registered causal encoder. Each operator (forward/backward) calls this itself so
    the two get INDEPENDENT weights. `hidden` defaults to gru_hidden but encoder_kwargs may override it
    (per-encoder width tuning for A/B param-fairness)."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; registered: {sorted(ENCODERS)}")
    kwargs.setdefault("hidden", hidden)
    return ENCODERS[name](in_dim, ctx_dim, **kwargs)


class CausalEncoder(nn.Module):
    """Base class: (T,B,in_dim) -> (T,B,ctx_dim), strictly causal in T."""

    def __init__(self, in_dim, ctx_dim, hidden=64):
        super().__init__()
        self.in_dim = in_dim
        self.ctx_dim = ctx_dim
        self.hidden = hidden

    def forward(self, inp):                                # (T,B,in_dim) -> (T,B,ctx_dim)
        raise NotImplementedError


@register("gru")
class GRUEncoder(CausalEncoder):
    """Baseline: 1-layer unidirectional GRU + linear projection to ctx_dim. Byte-identical to the
    original OperatorFilter encoder (gru then to_ctx, same construction order -> same init RNG)."""

    def __init__(self, in_dim, ctx_dim, hidden=64, layers=1):
        super().__init__(in_dim, ctx_dim, hidden)
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers)   # (T,B,in_dim) native seq-first
        self.to_ctx = nn.Linear(hidden, ctx_dim)

    def forward(self, inp):
        h, _ = self.gru(inp)                               # (T,B,hidden)
        return self.to_ctx(h)                              # (T,B,ctx_dim)


@register("tcn")
class TCNEncoder(CausalEncoder):
    """Causal dilated Temporal Conv Net: LEFT-padded dilated 1-D convs (out[t] depends only on inputs <=t)
    in residual blocks with dilations 1,2,4,... Receptive field RF = 1 + (kernel-1)*(2^n_layers - 1);
    n_layers=7, kernel=2 -> RF=128 >= T=100. Fully PARALLEL over time (no recurrence). For accuracy at
    larger T, grow n_layers so RF >= T."""

    def __init__(self, in_dim, ctx_dim, hidden=32, n_layers=7, kernel=2):
        super().__init__(in_dim, ctx_dim, hidden)
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel, dilation=2 ** i) for i in range(n_layers))
        self.pads = [(kernel - 1) * (2 ** i) for i in range(n_layers)]  # left-pad = causal
        self.act = nn.GELU()
        self.out = nn.Conv1d(hidden, ctx_dim, 1)

    def forward(self, inp):                                # (T,B,in_dim)
        x = self.inp(inp.permute(1, 2, 0))                 # (B,hidden,T)
        for conv, pad in zip(self.convs, self.pads):
            x = x + self.act(conv(F.pad(x, (pad, 0))))     # left-pad only -> strictly causal; residual
        return self.out(x).permute(2, 0, 1)                # (T,B,ctx_dim)


def _sinusoidal_pos(T, d, device):
    """Parameter-free sinusoidal positional encoding (T,d) -- works for ANY T (needed for the scaling study)."""
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : (d - d // 2)]) if d % 2 else torch.cos(ang)
    return pe


@register("transformer")
class TransformerEncoder(CausalEncoder):
    """Causal-masked self-attention: input proj + sinusoidal positional encoding + a causal
    TransformerEncoder (triangular -inf mask -> token t attends only to <=t). T~100 so O(T^2) is trivial.
    The observation-mask is a FEATURE (last input channel), NOT an attention padding mask."""

    def __init__(self, in_dim, ctx_dim, hidden=48, n_heads=4, n_layers=2, ffn=64):
        super().__init__(in_dim, ctx_dim, hidden)
        self.inp = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(hidden, n_heads, ffn, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(hidden, ctx_dim)

    def forward(self, inp):                                # (T,B,in_dim)
        T = inp.shape[0]
        x = self.inp(inp).transpose(0, 1)                  # (B,T,hidden)
        x = x + _sinusoidal_pos(T, x.shape[-1], x.device)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.enc(x, mask=mask, is_causal=True)         # causal self-attention
        return self.out(x).transpose(0, 1)                 # (T,B,ctx_dim)


@register("rno")
class RNOEncoder(CausalEncoder):
    """Recurrent Neural Operator (neuralop) as a causal context encoder. Our observations are per-timestep
    VECTORS with no spatial structure, so we give the RNO a size-1 spatial axis -- the spectral conv over a
    length-1 axis collapses to a channel-linear, so the RNO reduces to its GRU-like recurrence (selu +
    reset gate; the Fourier/operator part is inert here). We drive the RNO's lifting + recurrent blocks
    directly and apply our OWN per-timestep projection to ctx_dim, bypassing RNO.projection (whose
    return_sequences=True path has a time-axis shape bug). Causal (recurrence over time)."""

    def __init__(self, in_dim, ctx_dim, hidden=32, n_layers=2, n_modes=1):
        super().__init__(in_dim, ctx_dim, hidden)
        from neuralop.models import RNO
        self.rno = RNO(n_modes=(n_modes,), in_channels=in_dim, out_channels=1,   # out_channels unused (bypassed)
                       hidden_channels=hidden, n_layers=n_layers, return_sequences=True,
                       positional_embedding=None)
        self.to_ctx = nn.Linear(hidden, ctx_dim)

    def forward(self, inp):                                # (T,B,in_dim)
        T, B, _ = inp.shape
        x = inp.permute(1, 0, 2).reshape(B * T, self.in_dim, 1)   # (B*T, in_dim, spatial=1); row=(b,t)
        x = self.rno.lifting(x).reshape(B, T, -1, 1)             # (B, T, hidden, 1)
        for layer in self.rno.layers:                            # RNO recurrent blocks (return_sequences)
            x = layer(x, None)                                   # (B, T, hidden, 1)
        return self.to_ctx(x.squeeze(-1)).permute(1, 0, 2)       # (T, B, ctx_dim)


@torch.no_grad()
def _rand_input(enc, T, B, device):
    return torch.randn(T, B, enc.in_dim, device=device)


def assert_causal(enc, T=16, B=2, device="cpu"):
    """CORRECTNESS GATE: verify the encoder is strictly causal -- output at time t must not depend on any
    input at t' > t. Probes a few output times via autodiff (grad of out[t] w.r.t. future inputs == 0).
    A leak here silently 'cheats' the filter KL by letting ctx_t peek at future observations."""
    was_training = enc.training
    enc.eval()
    x = torch.randn(T, B, enc.in_dim, device=device, requires_grad=True)
    for t_probe in (0, T // 2, T - 1):
        if x.grad is not None:
            x.grad = None
        enc(x)[t_probe].sum().backward()
        fut = x.grad.abs().sum(dim=(1, 2))[t_probe + 1:]   # sensitivity to inputs strictly after t_probe
        leak = 0.0 if fut.numel() == 0 else float(fut.max())
        assert leak == 0.0, f"{type(enc).__name__}: output[{t_probe}] leaks from future inputs (leak={leak:.3e})"
    enc.train(was_training)
    return True


def assert_anticausal(enc, T=16, B=2, device="cpu"):
    """Mirror check for the reverse-wrapped (backward) encoder: flip(0)->enc->flip(0) must be anti-causal,
    i.e. output at t must not depend on inputs t' < t. (Guaranteed by the flip identity if enc is causal,
    but checked explicitly.)"""
    was_training = enc.training
    enc.eval()

    def wrapped(x):
        return enc(x.flip(0)).flip(0)

    x = torch.randn(T, B, enc.in_dim, device=device, requires_grad=True)
    for t_probe in (0, T // 2, T - 1):
        if x.grad is not None:
            x.grad = None
        wrapped(x)[t_probe].sum().backward()
        past = x.grad.abs().sum(dim=(1, 2))[:t_probe]      # sensitivity to inputs strictly before t_probe
        leak = 0.0 if past.numel() == 0 else float(past.max())
        assert leak == 0.0, f"{type(enc).__name__}(reverse): output[{t_probe}] leaks from past inputs (leak={leak:.3e})"
    enc.train(was_training)
    return True
