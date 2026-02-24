"""
################################################################################
#  [Module 01/10]  config.py  –  Global Constants & ARMAX Defaults
#
#  Contains:
#    - TRADING_PARTNERS   : congestion-dummy mapping per zone
#    - ARMAX_*            : global ARMAX solver/convergence defaults
#      (all overridden inside the __main__ block in main.py at runtime)
#
#  Dependencies: none
################################################################################
"""

# Trading partners for congestion dummies (must match regional_data_combiner.py)
TRADING_PARTNERS = {
    'SE1': ['FI', 'NO4', 'SE2'],
    'SE2': ['NO3', 'NO4', 'SE1', 'SE3'],
    'SE3': ['DK1', 'FI', 'NO1', 'SE2', 'SE4'],
    'SE4': []   # To be defined
}

# ARMAX fitting defaults (overridden in __main__ config block if needed)
ARMAX_ALLOW_NONCONVERGED = False
ARMAX_MAXITER = 300
ARMAX_SOLVER = 'statespace'
ARMAX_USE_WARM_START = True
ARMAX_ENABLE_FALLBACK_ORDERS = True
ARMAX_FALLBACK_ORDERS = [(1, 0, 1), (2, 0, 2), (3, 0, 3)]
ARMAX_BASELINE_SPEC = {
    'enabled': True,
    'order': (1, 0, 1),
    'extra_ar_lags': [],
    'drop_initial_nan': True,
    'label': 'ARMAX(1,0,1)'
}
