"""
################################################################################
#  [Module 10/10]  main.py  –  Configuration and Entry Point
#
#  This is the single file you edit to configure and run the analysis.
#  All toggles, paths, and method choices live in the if __name__ == '__main__'
#  block below (lines ~60 onwards).
#
#  Key configuration sections:
#    ACTIVE_ZONE                    : 'SE1' / 'SE2' / 'SE3' / 'SE4'
#    NEGATIVE_PRICE_HANDLING        : 'clip' or 'shift'
#    OUTLIER_METHOD                 : 'fredriksson' or 'gianfreda'
#    HANDLE_OUTLIERS_BEFORE_LOG     : timing of outlier removal in pipeline
#    ARMAX_BASELINE_SPEC            : order, extra AR lags, label
#    OPTIMIZE_ARMAX_LAGS            : grid-search ARMAX(p,q)
#    RUN_ROLLING_WINDOW             : overlapping rolling OLS
#    RUN_TVP_WIND_KALMAN            : time-varying coefficient (Kalman)
#    RUN_STRUCTURAL_BREAK           : Bai-Perron level or trend breaks
#    RUN_QUANTILE_REGRESSION        : quantile regression across price quantiles
#    EXPORT_DATA_FOR_R              : export to CSV for R strucchange
#    PATHS                          : file paths to data sources
#
#  Usage:
#    python -m full_regression.main      (from MSc-Econ-Thesis directory)
#    python main.py                      (from this directory)
#
#  Dependencies: all modules (01–09)
################################################################################
"""

# Standard library + third-party imports (needed for the __main__ block)
import os
import sys

# Add parent directory to path if running as standalone script
if __name__ == "__main__" and __package__ is None:
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)

from full_regression.config import TRADING_PARTNERS, ARMAX_BASELINE_SPEC
from full_regression.data_loading import load_data
from full_regression.preprocessing import (
    handle_negative_prices,
    handle_outliers_fredriksson,
    handle_outliers_gianfreda,
    apply_log_transform,
    deseasonalize_logged_variables
)
from full_regression.regression_analysis import perform_multivariate_analysis
from full_regression.structural_analysis import run_rolling_window_analysis
from full_regression.visualization import (
    plot_zone_comparisons,
    run_visualizations,
    export_data_for_R
)

# --- 6. EXECUTION BLOCK ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ACTIVE_ZONE = 'SE1'

    # --- ZONE COMPARISON PLOTS ---
    # When True: generates comparison plots overlaying SE1-SE4 for price, log price,
    # volatility (24h rolling std), and wind production share
    RUN_ZONE_COMPARISONS = False

    # --- VISUALIZATION TOGGLE ---
    # Toggle for data visualization and outlier detection
    # When True: generates comprehensive visualizations of raw data and outlier detection
    # When False: skips visualization and proceeds directly to regression analysis
    RUN_VISUALIZATIONS = False

    # --- TRANSFORMATION SETTINGS (ALWAYS APPLIED) ---
    # Log transformation and deseasonalization are ALWAYS applied (Fredriksson 2016 methodology)
    # - Log: applies log() to Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    # - Net_Exchange is NOT logged (can contain negative values)
    # - Deseasonalization: applied to LOGGED series using dummy variable regression
    # These transformations are hardcoded and cannot be toggled off.

    # --- NEGATIVE VALUE HANDLING ---
    # Method for handling negative price values (before log transformation)
    # 'clip': Replace values below 0.01 with 0.01 (current default, affects only negative/zero values)
    # 'shift': Shift entire price series upward so minimum becomes 0.01 (preserves relative differences)
    # Note: Net_Exchange is never modified (expected to have negative values)
    NEGATIVE_PRICE_HANDLING = 'shift'  # Options: 'clip' or 'shift'

    # --- OUTLIER HANDLING SETTINGS ---
    # Outlier handling is ALWAYS applied (cannot be toggled off)
    #
    # Outlier handling method selection
    # 'fredriksson': Fredriksson (2016) methodology
    #   - Threshold: +6σ / -3.7σ (asymmetric)
    #   - Replacement: Mean of 24h and 48h before/after outlier
    # 'gianfreda': Gianfreda (2010) / Mugele et al. (2005) methodology
    #   - Threshold: ±3σ (symmetric)
    #   - Replacement: Capped at ±3σ for respective weekday
    OUTLIER_METHOD = 'fredriksson'  # Options: 'fredriksson' or 'gianfreda'

    # METHODOLOGICAL NOTE:
    # Fredriksson (2016) applies outlier filter TWICE:
    #   1st: On original price series (found 31 outliers)
    #   2nd: On deseasonalized price series (found 42 outliers)
    #
    # OUR DEFAULT APPROACH: Apply outlier filter ONCE, on logged-deseasonalized series
    # Rationale:
    #   - Seasonal patterns mask true outliers (e.g., high winter prices vs low summer)
    #   - Log transformation stabilizes variance
    #   - Deseasonalized mean ≈ 0 makes threshold more meaningful
    #   - Cleaner single-pass approach with stronger statistical justification
    #   - Fredriksson provides no theoretical justification for double application
    #
    # ALTERNATIVE APPROACH (via HANDLE_OUTLIERS_BEFORE_LOG=True):
    #   Apply outlier filter on raw price series before log transformation
    #   - Suitable when outliers are data quality issues rather than market events
    #   - Prevents near-zero prices from creating excessive negative outliers in log space
    #
    # TODO: Future sensitivity analysis could compare single vs. double application

    # Toggle for linear interpolation of missing values
    # When True: fills missing values by linear interpolation between surrounding values
    # When False: drops all rows with missing values (original behavior)
    USE_LINEAR_INTERPOLATION = True

    # --- OUTLIER TIMING CONFIGURATION ---
    # Control WHEN outlier handling is applied in the transformation pipeline
    #
    # When True: Apply outlier detection/replacement BEFORE log transformation
    #   - Applied to raw Price series (after negative value handling)
    #   - Statistical rationale: Remove extreme values that could distort log transformation
    #   - Suitable when outliers are data quality issues (recording errors, system failures)
    #
    # When False: Apply outlier detection/replacement AFTER log + deseasonalization (DEFAULT)
    #   - Applied to logged-deseasonalized Price series (current behavior)
    #   - Statistical rationale: Remove outliers in transformed space with stabilized variance
    #   - Suitable when outliers are legitimate extreme market events
    #
    # Note: Cannot apply BOTH early and late outlier handling in single run
    HANDLE_OUTLIERS_BEFORE_LOG = False  # Default: False (preserves current behavior)

    # --- COMMODITY PRICE LAGGING ---
    # Commodity prices (oil & gas) are ALWAYS lagged by 24 hours (hardcoded in pipeline)
    # Rationale: Day-ahead electricity market uses commodity prices from bidding time (D-1)
    # This aligns with standard literature (Weron, Huisman, etc.)
    LAG_COMMODITY_HOURS = 24  # Applied automatically in load_data()

    # --- DIAGNOSTIC TEST TOGGLES (Fredriksson 2016 methodology) ---
    # Toggle for Ljung-Box test for autocorrelation
    # Tests whether residuals exhibit autocorrelation at various lag lengths
    RUN_LJUNGBOX_TEST = True

    # Toggle for heteroskedasticity and ARCH effects tests
    # Includes Engle's ARCH test and Ljung-Box Q test on squared residuals
    # If ARCH effects detected, consider implementing GARCHX model
    RUN_HETEROSKEDASTICITY_TESTS = True

    # Toggle for stationarity tests (ADF and DF-GLS)
    # Tests whether price series has a unit root (non-stationary)
    RUN_STATIONARITY_TESTS = True

    # --- MODEL SPECIFICATION TOGGLES ---
    # Toggle for automated ARMAX lag selection via AIC minimization
    # When True: Tests AR lags 1-10 and MA lags 1-10, selects optimal model
    # When False: Uses default ARMAX(1,1) specification
    # WARNING: This can take several minutes to run (tests 100 model combinations)
    OPTIMIZE_ARMAX_LAGS = False

    # Toggle for checkpointed lag selection with Ljung-Box diagnostics
    # When True: Uses checkpointed version that saves progress and includes diagnostics
    # When False: Uses original version (no checkpointing)
    # Only applies if OPTIMIZE_ARMAX_LAGS = True
    USE_CHECKPOINTED_LAG_SELECTION = False

    # --- ARMAX SEARCH GRID & SELECTION POLICY ---
    # Search ranges for p and q (inclusive)
    ARMAX_SEARCH_P_MIN = 1
    ARMAX_SEARCH_P_MAX = 1
    ARMAX_SEARCH_Q_MIN = 1
    ARMAX_SEARCH_Q_MAX = 1
    ARMAX_SEARCH_EXCLUDE_00 = True
    # Eligibility rules for model selection in grid search
    ARMAX_SEARCH_REQUIRE_CONVERGENCE = True
    ARMAX_SEARCH_REQUIRE_LJUNGBOX_PASS = True
    # Ranking criterion among eligible models: 'bic' (recommended) or 'aic'
    ARMAX_SEARCH_SELECTION_CRITERION = 'bic'
    # Number of top eligible models to save to results/armax_search_top_<ZONE>.csv
    ARMAX_SEARCH_SAVE_TOP_N = 20

    # --- ARMAX CONVERGENCE POLICY ---
    # When False (recommended): reject non-converged ARMAX fits and fail loudly
    # When True: allow using non-converged coefficients (can mimic OLS with zero AR/MA)
    ARMAX_ALLOW_NONCONVERGED = False
    # Optimizer controls
    ARMAX_MAXITER = 300
    ARMAX_SOLVER = 'statespace'
    ARMAX_USE_WARM_START = True
    # Fallback ladder if primary order does not converge
    ARMAX_ENABLE_FALLBACK_ORDERS = True
    ARMAX_FALLBACK_ORDERS = [(1, 0, 1), (2, 0, 2), (3, 0, 3)]
    # Baseline ARMAX spec (used when OPTIMIZE_ARMAX_LAGS=False):
    # - order: contiguous ARMA(p,q) component
    # - extra_ar_lags: sparse lagged dependent terms added to exogenous set for baseline run only
    ARMAX_BASELINE_SPEC = {
        'enabled': True,
        'order': (3, 0, 3),
        'extra_ar_lags': [23, 24, 25],  # Example: [23, 24, 25]
        'drop_initial_nan': True,
        'label': 'ARMAX(3,0,3)'
    }

    # --- GARCH CONFIGURATION ---
    # Fit GARCH only if ARCH effects detected (p < 0.05)
    FIT_GARCH_IF_ARCH = True
    # GARCH order: (p, q) for GARCH(p,q)
    GARCH_ORDER = (1, 1)
    # Note: Variance equation uses Wind_Forecast_Log (hardcoded, following Fredriksson 2016)

    # --- TVP KALMAN FILTER TOGGLE ---
    # When True: estimates time-varying wind coefficient using state-space model
    # When False: runs standard OLS + ARMAX analysis
    RUN_TVP_WIND_KALMAN = False

    # --- ROLLING-WINDOW ESTIMATION TOGGLE ---
    # When True: estimates wind coefficient using overlapping rolling windows (skips OLS/ARMAX)
    # When False: runs standard full-sample analysis
    RUN_ROLLING_WINDOW = False

    # Rolling window configuration
    ROLLING_WINDOW_YEARS = 1          # Window size in years
    ROLLING_STEP_YEARS = 1            # Step size between windows in years
    ROLLING_MIN_OBS = 24 * 365 - 24 * 30        # ~3 years minus 1 month tolerance, 24 * 365 * 3 - 24 * 30

    # --- QUANTILE REGRESSION TOGGLE ---
    # When True: estimates wind coefficient across quantiles of price distribution (skips OLS/ARMAX)
    # When False: runs standard analysis
    RUN_QUANTILE_REGRESSION = False

    # --- STRUCTURAL BREAK ANALYSIS TOGGLE ---
    # When True: detects structural breaks in wind coefficient
    # When False: runs standard analysis
    # NOTE: Requires 'ruptures' package for level break analysis (pip install ruptures)
    RUN_STRUCTURAL_BREAK = False

    # Structural break TYPE:
    # 'level' - Tests for step changes in coefficient mean (Bai-Perron methodology)
    # 'trend' - Tests for changes in coefficient slope over time (segmented linear regression)
    STRUCTURAL_BREAK_TYPE = 'trend'  # 'level' or 'trend'
    # Break estimation model:
    # 'ols' - Current baseline rolling OLS coefficient estimation
    # 'dynamic_armax' - Rolling ARMAX(3,0,3) coefficient estimation
    STRUCTURAL_BREAK_ESTIMATION_MODEL = 'ols'  # 'ols' or 'dynamic_armax'

    # Structural break configuration
    STRUCTURAL_BREAK_MAX_BREAKS = 3           # Maximum number of breaks to test (for both 'level' and 'trend')
    STRUCTURAL_BREAK_TRIMMING = 0.1          # Fraction of data to trim from endpoints (0.15 = 15%)
    # Known event dates to test with Chow test (list of 'YYYY-MM-DD' strings) - only for 'level' type
    # Examples: Russia-Ukraine invasion, COVID lockdowns, policy changes
    STRUCTURAL_BREAK_KNOWN_DATES = None #['2022-02-24', '2020-03-11'] # Russia invades Ukraine, # WHO declares COVID-19 pandemic

    # Structural break rolling window configuration (independent from standalone rolling window)
    STRUCTURAL_BREAK_WINDOW_YEARS = 1       # Window size in years for coefficient estimation
    STRUCTURAL_BREAK_STEP_YEARS = 3/12      # Step size between windows in years (2 months)
    STRUCTURAL_BREAK_MIN_OBS = 24 * 365 - 24 * 30  # Minimum observations per window

    # --- TREND BREAK TEST METHOD (trend mode only) ---
    # 'legacy': current implementation (BIC + standard sequential F-tests)
    # 'bp_supf': Bai-Perron style sequential supF(l+1|l) with bootstrap/table inference
    # Default keeps existing behavior for immediate rollback safety.
    TREND_BREAK_TEST_METHOD = 'bp_supf'  # 'legacy' or 'bp_supf'

    # BP trend inference configuration (used when TREND_BREAK_TEST_METHOD='bp_supf')
    BP_INFERENCE_MODE = 'tables'           # 'both', 'tables', or 'bootstrap'
    BP_SIGNIFICANCE_LEVEL = 0.05         # One of: 0.10, 0.05, 0.025, 0.01
    BP_BOOTSTRAP_REPS = 999
    BP_BOOTSTRAP_BLOCK_LENGTH = 8
    BP_RANDOM_SEED = 42
    BP_USE_HAC_SE = True

    # --- R DATA EXPORT TOGGLE ---
    # When True: exports fully processed data to CSV for R's strucchange package (Bai-Perron)
    # When False: skips data export
    # Output: data_for_R/regression_data_{zone}_for_R.csv + metadata file
    EXPORT_DATA_FOR_R = False

    # Data file paths - dynamically set based on ACTIVE_ZONE
    PATHS = {
        'combined': f'master data files/2015-2025/Combined_{ACTIVE_ZONE}_Data_2015_2025.xlsx',
        'hydro': 'master data files/Master_Hydro_Reservoir.xlsx',
        'crude_oil': 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
        'commodities': 'master data files/Master_Commodities.xlsx'  # Still used for TTF Gas
    }

    # --- ZONE COMPARISON PLOTS (runs before zone-specific pipeline) ---
    if RUN_ZONE_COMPARISONS:
        plot_zone_comparisons(start_date='2015-01-01', end_date='2025-12-31', plots_dir='plots')

    try:
        # Load and clean full dataset (filtered to 2021-2024 for hydro/commodity availability)
        # Note: Commodity prices are automatically lagged within load_data()
        data = load_data(
            PATHS,
            target_region=ACTIVE_ZONE,
            zone_hydro=ACTIVE_ZONE,
            use_interpolation=USE_LINEAR_INTERPOLATION,
            start_date='2015-01-01',
            end_date='2017-12-31',
            lag_commodity_hours=LAG_COMMODITY_HOURS
        )
        print(f"Merge successful. Total hourly observations: {len(data)}")

        # --- ROLLING WINDOW MODE (WINDOW-LOCAL PREPROCESSING) ---
        # Run rolling-window analysis directly on raw data so each window is transformed locally.
        # This avoids full-sample transformation leakage into local coefficient estimates.
        if RUN_ROLLING_WINDOW:
            run_rolling_window_analysis(
                data,
                ACTIVE_ZONE,
                window_years=ROLLING_WINDOW_YEARS,
                step_years=ROLLING_STEP_YEARS,
                min_obs=ROLLING_MIN_OBS,
                plots_dir="plots",
                results_dir="results",
                use_window_local_preprocessing=True,
                target_region=ACTIVE_ZONE,
                negative_price_handling=NEGATIVE_PRICE_HANDLING,
                outlier_method=OUTLIER_METHOD,
                handle_outliers_before_log=HANDLE_OUTLIERS_BEFORE_LOG
            )
            raise SystemExit(0)

        # --- STEP 1: VISUALIZATION OF RAW DATA (if enabled) ---
        # Visualize pure, untouched raw data BEFORE any transformations
        # This shows the true state of the data including any negative values or quality issues
        if RUN_VISUALIZATIONS:
            run_visualizations(data, ACTIVE_ZONE, method=OUTLIER_METHOD, stage='raw')

        # --- STEP 2: CHECK NEGATIVE VALUES & HANDLE PRICE ---
        # Check all variables for negative values and handle Price if needed
        # Net_Exchange is NOT checked or modified (expected to have negative values)
        # NOTE: This happens AFTER raw visualization so we can see the original data quality issues
        data = handle_negative_prices(data, method=NEGATIVE_PRICE_HANDLING)

        # --- STEP 2.5: EARLY OUTLIER HANDLING (if configured) ---
        # Applies if HANDLE_OUTLIERS_BEFORE_LOG=True
        #
        # EARLY OUTLIER DETECTION RATIONALE:
        #   - Applied to RAW price series (after negative value handling)
        #   - Suitable when outliers are data quality issues (recording errors, sensor failures)
        #   - Prevents extreme values from distorting log transformation
        #   - Trade-off: Less statistically rigorous (non-stabilized variance, seasonal patterns present)
        if HANDLE_OUTLIERS_BEFORE_LOG:
            print("\n" + "="*80)
            print("STEP 2.5: EARLY OUTLIER HANDLING (BEFORE LOG TRANSFORMATION)")
            print("="*80)
            print("Applying outlier detection and replacement to raw Price series")
            print("This occurs AFTER negative value handling but BEFORE log transformation\n")

            if OUTLIER_METHOD == 'fredriksson':
                data, outlier_stats_early = handle_outliers_fredriksson(data, apply_to_raw=True)
            elif OUTLIER_METHOD == 'gianfreda':
                data, outlier_stats_early = handle_outliers_gianfreda(data, apply_to_raw=True)
            else:
                raise ValueError(f"Unknown outlier method: {OUTLIER_METHOD}. Choose 'fredriksson' or 'gianfreda'.")

        # --- STEP 3: LOG TRANSFORMATION (always applied) ---
        # STANDARD APPROACH: Apply log transformation FIRST, before deseasonalization
        # This is the standard econometric approach for handling multiplicative seasonality
        # Note: Commodity prices are already lagged at this point
        # Note: Price negative values handled in STEP 2 (before log transformation)
        data = apply_log_transform(data)

        # Visualize logged data (if enabled)
        # This shows data AFTER negative handling and log transformation
        if RUN_VISUALIZATIONS:
            # Create temporary dataframe with logged variables mapped to base names for visualization
            data_logged_viz = data.copy()
            data_logged_viz['Price'] = data_logged_viz['Price_Log']
            data_logged_viz['Wind_Forecast'] = data_logged_viz['Wind_Forecast_Log']
            data_logged_viz['Hydro_Reserves'] = data_logged_viz['Hydro_Reserves_Log']
            data_logged_viz['Consumption'] = data_logged_viz['Consumption_Log']
            data_logged_viz['Oil_Price'] = data_logged_viz['Oil_Price_Log']
            # Net_Exchange stays the same (not logged)
            run_visualizations(data_logged_viz, ACTIVE_ZONE, method=OUTLIER_METHOD, stage='logged')

        # --- STEP 4: DESEASONALIZATION (always applied) ---
        # STANDARD APPROACH: Deseasonalize the LOGGED variables (after log transformation)
        # Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL deseasonalization)
        # Hydro, Oil, Gas: Year + Month ONLY (PARTIAL - no intraday patterns)
        data = deseasonalize_logged_variables(data)

        # --- STEP 5: LATE OUTLIER HANDLING (if not applied early) ---
        # Applies if HANDLE_OUTLIERS_BEFORE_LOG=False
        #
        # LATE OUTLIER DETECTION RATIONALE (RECOMMENDED APPROACH):
        #   - Applied to LOGGED-DESEASONALIZED series
        #   - More statistically rigorous:
        #       * Log transformation stabilizes variance
        #       * Deseasonalization removes seasonal patterns (high winter vs low summer)
        #       * Threshold more meaningful on zero-centered deseasonalized data
        #   - Suitable when outliers are extreme market events (not recording errors)
        #
        # MUTUALLY EXCLUSIVE WITH STEP 2.5:
        #   - Cannot apply both early and late outlier handling in same run
        #   - Prevents double-replacement of outliers
        if not HANDLE_OUTLIERS_BEFORE_LOG:
            print("\n" + "="*80)
            print("STEP 5: LATE OUTLIER HANDLING (AFTER TRANSFORMATIONS)")
            print("="*80)
            print("Applying outlier detection and replacement to transformed Price series")
            print("This occurs AFTER log transformation and deseasonalization\n")

            if OUTLIER_METHOD == 'fredriksson':
                data, outlier_stats = handle_outliers_fredriksson(data, apply_to_raw=False)
            elif OUTLIER_METHOD == 'gianfreda':
                data, outlier_stats = handle_outliers_gianfreda(data, apply_to_raw=False)
            else:
                raise ValueError(f"Unknown outlier method: {OUTLIER_METHOD}. Choose 'fredriksson' or 'gianfreda'.")

        # --- EXPORT DATA FOR R ANALYSIS (BAI-PERRON) ---
        # Export fully processed data for structural break testing in R
        # Note: Always exports logged and deseasonalized data (standard pipeline)
        if EXPORT_DATA_FOR_R:
            export_data_for_R(
                data=data,
                zone=ACTIVE_ZONE,
                use_log_transform=True,  # Always applied in standard pipeline
                use_deseasonalized=True,  # Always applied in standard pipeline
                handle_outliers=True,     # Always applied in standard pipeline
                outlier_method=OUTLIER_METHOD,
                negative_price_handling=NEGATIVE_PRICE_HANDLING,
                use_interpolation=USE_LINEAR_INTERPOLATION
            )

        # --- STEP 6: REGRESSION ANALYSIS ---
        # Run regression models with optional diagnostic tests
        # Commodity prices used in regression are lagged by 24h (from load_data)
        ols_model, armax_res, garch_res = perform_multivariate_analysis(data, ACTIVE_ZONE,
                                      target_region=ACTIVE_ZONE,
                                      run_ljungbox=RUN_LJUNGBOX_TEST,
                                      run_hetero_tests=RUN_HETEROSKEDASTICITY_TESTS,
                                      run_stationarity=RUN_STATIONARITY_TESTS,
                                      optimize_armax_lags=OPTIMIZE_ARMAX_LAGS,
                                      use_checkpointed_lag_selection=USE_CHECKPOINTED_LAG_SELECTION,
                                      armax_search_p_min=ARMAX_SEARCH_P_MIN,
                                      armax_search_p_max=ARMAX_SEARCH_P_MAX,
                                      armax_search_q_min=ARMAX_SEARCH_Q_MIN,
                                      armax_search_q_max=ARMAX_SEARCH_Q_MAX,
                                      armax_search_exclude_00=ARMAX_SEARCH_EXCLUDE_00,
                                      armax_search_require_convergence=ARMAX_SEARCH_REQUIRE_CONVERGENCE,
                                      armax_search_require_ljungbox_pass=ARMAX_SEARCH_REQUIRE_LJUNGBOX_PASS,
                                      armax_search_selection_criterion=ARMAX_SEARCH_SELECTION_CRITERION,
                                      armax_search_save_top_n=ARMAX_SEARCH_SAVE_TOP_N,
                                      run_tvp_wind_kalman=RUN_TVP_WIND_KALMAN,
                                      run_rolling_window=RUN_ROLLING_WINDOW,
                                      rolling_window_years=ROLLING_WINDOW_YEARS,
                                      rolling_step_years=ROLLING_STEP_YEARS,
                                      rolling_min_obs=ROLLING_MIN_OBS,
                                      run_quantile_regression=RUN_QUANTILE_REGRESSION,
                                      run_structural_break=RUN_STRUCTURAL_BREAK,
                                      structural_break_type=STRUCTURAL_BREAK_TYPE,
                                      structural_break_max_breaks=STRUCTURAL_BREAK_MAX_BREAKS,
                                      structural_break_trimming=STRUCTURAL_BREAK_TRIMMING,
                                      structural_break_known_dates=STRUCTURAL_BREAK_KNOWN_DATES,
                                      structural_break_window_years=STRUCTURAL_BREAK_WINDOW_YEARS,
                                      structural_break_step_years=STRUCTURAL_BREAK_STEP_YEARS,
                                      structural_break_min_obs=STRUCTURAL_BREAK_MIN_OBS,
                                      trend_break_test_method=TREND_BREAK_TEST_METHOD,
                                      bp_inference_mode=BP_INFERENCE_MODE,
                                      bp_significance_level=BP_SIGNIFICANCE_LEVEL,
                                      bp_bootstrap_reps=BP_BOOTSTRAP_REPS,
                                      bp_bootstrap_block_length=BP_BOOTSTRAP_BLOCK_LENGTH,
                                      bp_random_seed=BP_RANDOM_SEED,
                                      bp_use_hac_se=BP_USE_HAC_SE,
                                      structural_break_estimation_model=STRUCTURAL_BREAK_ESTIMATION_MODEL,
                                      armax_baseline_spec=ARMAX_BASELINE_SPEC)

    except Exception as e:
        print(f"Critical error during execution: {e}")
