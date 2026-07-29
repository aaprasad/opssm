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

"""High-D linear-sensor helpers: pseudo-inverse latent estimate + decode factory."""

import torch


def zhat_from_obs(y, C, d):
    """Pseudo-inverse latent estimate z_hat = (C^T C)^-1 C^T (y - d) -> (...,d_lat). C (D,d_lat),
    y (...,D). On the Stiefel manifold C^T C = I so this is C^T(y-d); the solve is kept for robustness
    at random init. The collocation / M-step center."""
    r = torch.einsum("od,...o->...d", C, y - d)                    # C^T (y-d)  (...,d_lat)
    gram = C.t() @ C                                               # (d_lat,d_lat)
    return torch.linalg.solve(gram, r.unsqueeze(-1)).squeeze(-1)   # (C^T C)^-1 C^T (y-d)


def make_decode(C, d):
    """h(z) = C z + d, C (D,d_lat), z (...,d_lat) -> (...,D). Obs are standardized to ~unit scale at the
    dataloader level, so the decode carries no separate obs-scale factor (s_scale removed)."""
    return lambda z: torch.einsum("od,...d->...o", C, z) + d
