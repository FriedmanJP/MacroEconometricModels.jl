# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Critical value tables for unit root tests.
"""

# =============================================================================
# Critical Value Tables
# =============================================================================

# MacKinnon (2010) response surface coefficients for ADF/PP p-values
# Format: (β∞, β₁, β₂) for τ = β∞ + β₁/T + β₂/T²
const MACKINNON_ADF_COEFS = Dict(
    # No constant, no trend (nc)
    :none => Dict(
        1  => (-2.5658, -1.960, -10.04),   # 1%
        5  => (-1.9393, -0.398,  -0.0),    # 5%
        10 => (-1.6156, -0.181,  -0.0)     # 10%
    ),
    # Constant only (c)
    :constant => Dict(
        1  => (-3.4336, -5.999, -29.25),
        5  => (-2.8621, -2.738,  -8.36),
        10 => (-2.5671, -1.438,  -4.48)
    ),
    # Constant and trend (ct)
    :trend => Dict(
        1  => (-3.9638, -8.353, -47.44),
        5  => (-3.4126, -4.039, -17.83),
        10 => (-3.1279, -2.418,  -7.58)
    )
)

# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
const KPSS_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => 0.739, 5 => 0.463, 10 => 0.347),
    :trend    => Dict(1 => 0.216, 5 => 0.146, 10 => 0.119)
)

# Zivot-Andrews critical values (Zivot & Andrews 1992, Table 4)
const ZA_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => -5.34, 5 => -4.80, 10 => -4.58),
    :trend    => Dict(1 => -4.80, 5 => -4.42, 10 => -4.11),
    :both     => Dict(1 => -5.57, 5 => -5.08, 10 => -4.82)
)

# Ng-Perron critical values (Ng & Perron 2001, Table 1)
const NGPERRON_CRITICAL_VALUES = Dict(
    :constant => Dict(
        :MZa => Dict(1 => -13.8, 5 => -8.1, 10 => -5.7),
        :MZt => Dict(1 => -2.58, 5 => -1.98, 10 => -1.62),
        :MSB => Dict(1 => 0.174, 5 => 0.233, 10 => 0.275),
        :MPT => Dict(1 => 1.78, 5 => 3.17, 10 => 4.45)
    ),
    :trend => Dict(
        :MZa => Dict(1 => -23.8, 5 => -17.3, 10 => -14.2),
        :MZt => Dict(1 => -3.42, 5 => -2.91, 10 => -2.62),
        :MSB => Dict(1 => 0.143, 5 => 0.168, 10 => 0.185),
        :MPT => Dict(1 => 4.03, 5 => 5.48, 10 => 6.67)
    )
)

# Johansen critical values (Osterwald-Lenum 1992)
# Format: [10%, 5%, 1%] for each n-r (number of common trends)
# Trace test critical values (constant in cointegrating relation)
const JOHANSEN_TRACE_CV_CONSTANT = Dict(
    1 => [6.50, 8.18, 11.65],
    2 => [15.66, 17.95, 23.52],
    3 => [28.71, 31.52, 37.22],
    4 => [45.23, 48.28, 55.43],
    5 => [66.49, 70.60, 78.87],
    6 => [85.18, 90.39, 104.20],
    7 => [118.99, 124.25, 136.06],
    8 => [151.38, 157.11, 168.92],
    9 => [186.54, 192.89, 206.95],
    10 => [224.63, 231.26, 247.18]
)

# Max eigenvalue test critical values
const JOHANSEN_MAX_CV_CONSTANT = Dict(
    1 => [6.50, 8.18, 11.65],
    2 => [12.91, 14.90, 19.19],
    3 => [18.90, 21.07, 25.75],
    4 => [24.78, 27.14, 32.14],
    5 => [30.84, 33.32, 38.78],
    6 => [36.25, 39.43, 44.59],
    7 => [42.06, 45.28, 51.30],
    8 => [48.43, 51.42, 57.07],
    9 => [54.01, 57.12, 62.80],
    10 => [59.00, 62.81, 68.83]
)
