import re

# Define which functions go into which modules
# Format: (module_name, [(start_line, end_line, function_name), ...])
MODULES = {
    'config': [
        # Lines 24-45: configuration constants
        (24, 45, 'TRADING_PARTNERS and ARMAX config')
    ],
    'data_loading': [
        (50, 275, 'load_data'),
        (279, 370, 'handle_negative_prices'),
    ],
    'preprocessing': [
        (390, 530, 'handle_outliers_fredriksson'),
        (529, 680, 'handle_outliers_gianfreda'),
        (677, 800, 'deseasonalize_logged_variables'),
        (797, 880, 'apply_log_transform'),
    ],
    'diagnostics': [
        (879, 940, 'run_ljungbox_test'),
        (933, 995, 'run_heteroskedasticity_tests'),
        (989, 1050, 'run_stationarity_tests'),
    ],
    'utils': [
        (1044, 1260, 'run_tvp_wind_kalman_analysis'),
        (1262, 1280, 'get_regression_variable_names'),
        (1283, 1325, 'preprocess_data_for_regression'),
        (1545, 1575, '_get_break_model_tag, _get_break_model_label'),
        (1570, 1600, '_extract_armax_wind_coef, _attach_inferred_frequency'),
        (1592, 1620, 'frequency utilities'),
    ],
    'regression_models': [
        (1615, 1730, 'ARMAX utilities and fitting'),
        (1725, 1830, '_fit_armax_with_controls, _fit_armax_with_fallback'),
        (1821, 1900, '_diagnose_nonconvergence_simple'),
        (4145, 4320, 'ARMAX candidate evaluation'),
        (4290, 4330, 'ARMAX search grid'),
        (4332, 4420, 'select_armax_lags_aic functions'),
        (4534, 4600, 'fit_garchx_model'),
    ],
    'regression_analysis': [
        (4612, 4800, 'perform_multivariate_analysis'),
    ],
    'structural_analysis': [
        (1327, 1450, 'run_rolling_window_analysis'),
        (1937, 2000, '_estimate_rolling_wind_coefficients'),
        (2065, 2100, '_run_dynamic_break_lr_tests'),
        (2197, 2850, 'run_structural_break_analysis'),
        (2840, 3400, 'run_trend_break_analysis functions'),
        (3898, 3950, 'run_quantile_regression_analysis'),
    ],
    'visualization': [
        (4964, 5100, 'plot_zone_comparisons'),
        (5096, 5150, 'plot_time_series'),
        (5133, 5220, 'plot_distributions, plot_boxplots'),
        (5263, 5360, 'detect_outliers, plot_outliers_timeline'),
        (5420, 5460, 'plot_scatter_matrix, run_visualizations'),
        (5520, 5560, 'export_data_for_R'),
    ]
}

# Read the original file
with open("c:\Users\patri\VSCode\MSc-Econ-Thesis\full_regression.py", 'r') as f:
    lines = f.readlines()

# Extract imports (lines 1-21)
imports = ''.join(lines[0:21])

# Extract configs (lines 24-45, index 23-44)
config_section = ''.join(lines[23:45])

print("File read successfully!")
print(f"Total lines: {len(lines)}")
print(f"Imports extracted: {len(imports)} chars")
print(f"Config extracted: {len(config_section)} chars")
print("Now creating individual module files...")

