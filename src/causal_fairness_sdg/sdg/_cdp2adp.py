"""
Copyright 2020 (https://github.com/IBM/discrete-gaussian-differential-privacy)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Vendored (cdp_rho only) from private-pgm/reprosyn's cdp2adp.py, itself from
Thomas Steinke's discrete-gaussian-differential-privacy repo
(https://arxiv.org/abs/2004.00010), so this project doesn't need to depend on
reprosyn as a package for one small utility function.
"""

import math


def cdp_delta(rho: float, eps: float) -> float:
    assert rho >= 0
    assert eps >= 0
    if rho == 0:
        return 0.0

    amin = 1.01
    amax = (eps + 1) / (2 * rho) + 2
    alpha = amin
    for _ in range(1000):
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    delta = math.exp(
        (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
    ) / (alpha - 1.0)
    return min(delta, 1.0)


def cdp_rho(eps: float, delta: float) -> float:
    """Given (eps, delta), find the smallest rho such that rho-CDP implies
    (eps, delta)-DP."""
    assert eps >= 0
    assert delta > 0
    if delta >= 1:
        return 0.0
    rhomin = 0.0
    rhomax = eps + 1
    for _ in range(1000):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin
