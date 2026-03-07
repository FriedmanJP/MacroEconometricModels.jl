# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Panel regression display — report() dispatches
# =============================================================================
# Note: Base.show(io, m) methods are defined in types.jl
# This file adds report() convenience wrappers.

"""
    report(m::PanelRegModel)

Print a Stata-style regression table for a panel linear model to stdout.
"""
report(m::PanelRegModel) = show(stdout, m)

"""
    report(m::PanelIVModel)

Print a Stata-style regression table for a panel IV model to stdout.
"""
report(m::PanelIVModel) = show(stdout, m)

"""
    report(m::PanelLogitModel)

Print a Stata-style regression table for a panel logit model to stdout.
"""
report(m::PanelLogitModel) = show(stdout, m)

"""
    report(m::PanelProbitModel)

Print a Stata-style regression table for a panel probit model to stdout.
"""
report(m::PanelProbitModel) = show(stdout, m)

"""
    report(t::PanelTestResult)

Print a specification test result to stdout.
"""
report(t::PanelTestResult) = show(stdout, t)
