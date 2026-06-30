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
    """Pseudo-inverse latent estimate z_hat = C^+ (y - d) per (T,B) -- the collocation center."""
    return ((y - d) * C).sum(-1) / (C * C).sum().clamp_min(1e-8)


def make_decode(C, d, s_scale):
    return lambda z: s_scale * C * z[..., None] + d
