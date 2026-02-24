"""
full_regression  –  Modular Electricity Price Analysis Package

Module order (pipeline sequence):
  01  config              : global constants (TRADING_PARTNERS, ARMAX defaults)
  02  data_loading        : load_data()
  03  preprocessing       : outlier handling, log transform, deseasonalization
  04  diagnostics         : Ljung-Box, ARCH, ADF, DF-GLS tests
  05  utils               : TVP Kalman filter, get_regression_variable_names()
  06  regression_models   : ARMAX fitting, GARCH, lag selection
  07  regression_analysis : perform_multivariate_analysis() orchestrator
  08  structural_analysis : rolling windows, structural breaks, quantile regression
  09  visualization       : all plotting functions + export_data_for_R()
  10  main                : entry point and configuration

Run:
  python -m full_regression.main
"""
