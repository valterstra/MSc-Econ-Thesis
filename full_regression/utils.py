"""
################################################################################
#  [Module 05/10]  utils.py  –  Utility Functions & TVP Kalman Filter
#
#  Contains:
#    1. run_tvp_wind_kalman_analysis()  : state-space Kalman filter for
#                                         time-varying wind coefficient
#    2. get_regression_variable_names() : returns Y name + exog list for zone
#
#  Dependencies: config
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import TRADING_PARTNERS


# --- 4. MODELING FUNCTIONS (utilities) ---

def run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, plots_dir="plots"):
    """
    Estimate time-varying parameter (TVP) model for the wind coefficient using state-space Kalman filter.

    Model specification:
        Observation: y_t = beta_t * w_t + controls_t' * gamma + e_t
        State:       beta_t = beta_{t-1} + u_t

    Uses Frisch-Waugh-Lovell partialling out to control for other regressors,
    then estimates a random-walk state-space model for the wind coefficient.

    Note: Always uses Wind_Forecast_Log (logged wind variable).

    **IMPORTANT - NEEDS FURTHER INVESTIGATION:**
    Current implementation on hourly data shows excessive high-frequency volatility,
    suggesting the model is overfitting to noise rather than capturing genuine
    structural variation in the wind coefficient. The random walk state process
    with unconstrained variance allows β_t to change hour-to-hour, which is
    economically implausible.

    Potential improvements to explore:
    1. Aggregate to daily data (daily averages or peak prices) to reduce noise
    2. Constrain state variance to force smoother evolution
    3. Use AR(1) state process instead of random walk for mean reversion
    4. Consider fixed-coefficient models with structural breaks instead
    5. Estimate on rolling windows rather than full state-space approach

    NOTE: Kalman filters may not be the optimal approach for analyzing coefficient
    evolution over time in this context. Alternative methods (rolling regressions,
    regime-switching models, or dummy variable interactions) may be more appropriate.

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier (e.g., 'SE1')
    - Y: Dependent variable (price series)
    - exog_vars: List of exogenous variable names
    - plots_dir: Directory to save output plots
    """
    print("\n" + "="*80)
    print(f"TVP KALMAN FILTER ANALYSIS - TIME-VARYING WIND COEFFICIENT ({zone})")
    print("="*80)
    print("\nModel: y_t = beta_t * w_t + controls' * gamma + e_t")
    print("State: beta_t = beta_{t-1} + u_t (random walk)")

    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Always use logged wind variable
    wind_col = 'Wind_Forecast_Log'

    print(f"\nWind variable: {wind_col}")

    # Step 2: Create control columns (exog_vars minus wind)
    control_cols = [col for col in exog_vars if col != wind_col]
    print(f"Control variables: {control_cols}")

    # Extract data
    y = Y.copy()
    w = df[wind_col].copy()
    controls = df[control_cols].copy()

    # Align indices and drop any NaN
    combined = pd.concat([y, w, controls], axis=1).dropna()
    y = combined.iloc[:, 0]
    w = combined.iloc[:, 1]
    controls = combined.iloc[:, 2:]

    print(f"\nObservations after alignment: {len(y)}")

    # Step 3: Frisch-Waugh-Lovell partialling out
    print("\n--- Frisch-Waugh-Lovell Partialling Out ---")
    print("Removing control variable effects from both Y and Wind...")

    # Add constant to controls
    controls_with_const = sm.add_constant(controls)

    # Regress Y on controls and get residuals
    y_on_controls = sm.OLS(y, controls_with_const).fit()
    y_star = y_on_controls.resid
    print(f"  y* = residuals from OLS(Y ~ const + controls), R²={y_on_controls.rsquared:.4f}")

    # Regress wind on controls and get residuals
    w_on_controls = sm.OLS(w, controls_with_const).fit()
    w_star = w_on_controls.resid
    print(f"  w* = residuals from OLS(Wind ~ const + controls), R²={w_on_controls.rsquared:.4f}")

    # Step 4: Define custom TVP state-space model
    print("\n--- Fitting State-Space Model ---")
    print("Estimating time-varying wind coefficient via Kalman filter...")

    class TVPWind(sm.tsa.statespace.MLEModel):
        """
        Time-varying parameter model for wind coefficient.
        State equation: beta_t = beta_{t-1} + u_t (random walk)
        Observation equation: y*_t = beta_t * w*_t + e_t
        """
        def __init__(self, y_star, w_star):
            super().__init__(y_star, k_states=1)
            # Store w_star for use in design matrix
            self._w_star = w_star.values.reshape(1, 1, -1)
            # Design matrix: (1, k_states, nobs) - contains w_star
            self.ssm['design'] = self._w_star
            # Transition matrix: [[1.0]] (random walk)
            self.ssm['transition'] = np.array([[1.0]])
            # Selection matrix: [[1.0]]
            self.ssm['selection'] = np.array([[1.0]])
            # Initialize with approximate diffuse prior
            self.initialize_approximate_diffuse()

        @property
        def param_names(self):
            return ['log_obs_var', 'log_state_var']

        @property
        def start_params(self):
            # Starting values: r=1 (log=0), q≈0.14 (log=-2)
            return np.array([0.0, -2.0])

        def update(self, params, **kwargs):
            # Observation variance (r)
            r = np.exp(params[0])
            # State variance (q)
            q = np.exp(params[1])
            self.ssm['obs_cov'] = np.array([[r]])
            self.ssm['state_cov'] = np.array([[q]])

    # Fit the model
    tvp_model = TVPWind(y_star, w_star)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        tvp_results = tvp_model.fit(disp=False)

    # Step 5: Extract results
    beta_t = tvp_results.smoothed_state[0]
    se_t = np.sqrt(tvp_results.smoothed_state_cov[0, 0, :])
    upper_95 = beta_t + 1.96 * se_t
    lower_95 = beta_t - 1.96 * se_t

    # Get estimated variances
    r_hat = np.exp(tvp_results.params.iloc[0])  # Observation variance
    q_hat = np.exp(tvp_results.params.iloc[1])  # State variance

    # Step 6: Print summary statistics
    print("\n" + "="*80)
    print("TVP WIND COEFFICIENT - SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTime-varying beta_t (wind coefficient):")
    print(f"  Mean:   {beta_t.mean():.6f}")
    print(f"  Std:    {beta_t.std():.6f}")
    print(f"  Min:    {beta_t.min():.6f}")
    print(f"  Max:    {beta_t.max():.6f}")
    print(f"  Final:  {beta_t[-1]:.6f}")

    print(f"\nEstimated variances:")
    print(f"  Observation variance (r): {r_hat:.6f}")
    print(f"  State variance (q):       {q_hat:.6f}")
    print(f"  Signal-to-noise ratio:    {q_hat/r_hat:.6f}")

    print(f"\nModel fit:")
    print(f"  Log-likelihood: {tvp_results.llf:.2f}")
    print(f"  AIC:            {tvp_results.aic:.2f}")
    print(f"  BIC:            {tvp_results.bic:.2f}")

    # Step 7: Create and save plot
    print(f"\n--- Creating TVP Wind Coefficient Plot ---")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create time index
    time_index = y.index

    # Plot beta_t with confidence bands
    ax.plot(time_index, beta_t, color='blue', linewidth=1.5, label=r'$\beta_t$ (Wind coefficient)')
    ax.fill_between(time_index, lower_95, upper_95, color='blue', alpha=0.2, label='95% CI')

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add horizontal line at mean
    ax.axhline(y=beta_t.mean(), color='red', linestyle=':', linewidth=1.5,
               label=f'Mean = {beta_t.mean():.4f}')

    ax.set_title(f'Time-Varying Wind Coefficient - {zone}\n(Kalman Filter State-Space Estimation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(r'$\beta_t$ (Wind Coefficient)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation with key statistics
    stats_text = (f'Mean: {beta_t.mean():.4f}\n'
                  f'Std: {beta_t.std():.4f}\n'
                  f'Min: {beta_t.min():.4f}\n'
                  f'Max: {beta_t.max():.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot to zone-specific subfolder
    zone_plots_dir = os.path.join(plots_dir, zone)
    os.makedirs(zone_plots_dir, exist_ok=True)
    plot_path = os.path.join(zone_plots_dir, f'tvp_beta_wind_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("TVP KALMAN FILTER ANALYSIS COMPLETE")
    print("="*80)

    return beta_t, se_t, tvp_results


def get_regression_variable_names(df, target_region='SE1'):
    """Return dependent variable name and exogenous variable names used in regression."""
    y_name = 'Price_Log_Deseasonalized'
    exog_vars = [
        'Wind_Forecast_Log',
        'Hydro_Reserves_Log_Deseasonalized',
        'Net_Exchange',
        'Consumption_Log_Deseasonalized',
        'Oil_Price_Log_Deseasonalized',
        'Gas_Price_Log_Deseasonalized'
    ]

    trading_partners = TRADING_PARTNERS.get(target_region, [])
    for partner in trading_partners:
        bneck_col = f'BNECK_{target_region}_{partner}'
        if bneck_col in df.columns:
            exog_vars.append(bneck_col)

    return y_name, exog_vars
