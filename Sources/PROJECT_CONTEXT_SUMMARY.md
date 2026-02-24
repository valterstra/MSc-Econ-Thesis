# Project Context Brief (Narrative Version)

## What this project is about
This project studies how wind power generation affects electricity spot prices in Sweden, and especially how that relationship changes over time across the Swedish bidding zones (SE1, SE2, SE3, SE4).

The central idea is that the wind-price relationship is not necessarily constant. In some periods, additional wind generation may have a stronger price-reducing effect; in other periods, that effect may weaken or shift due to market integration, transmission constraints, fuel prices, demand conditions, and broader structural events.

So the project has two goals at the same time:
1. Estimate the contemporaneous relationship between prices and wind while controlling for key market drivers.
2. Test whether the wind coefficient itself appears stable over time or shows evidence of structural change.

## Conceptual framing
The project is built around a merit-order interpretation: when low marginal-cost generation (wind) increases, it tends to displace higher-cost generation and push spot prices downward. But that theoretical effect is moderated by other parts of the system, so the empirical model includes controls that represent hydro conditions, consumption, cross-border or inter-zonal exchange, and fossil fuel proxies.

Rather than assuming one global constant effect over the full sample, the project explicitly examines whether the wind effect evolves over time.

## What is being modeled
At the core, the empirical setup is a multivariate electricity price model where price is the dependent variable and wind is a key explanatory variable among controls.

The control structure is designed to capture major confounders:
- Wind forecast (primary variable of interest)
- Hydro reservoir conditions
- Net exchange flows
- Consumption (demand proxy)
- Oil and gas price proxies
- Congestion/bottleneck indicators where relevant

The analysis is done zone-by-zone, so each Swedish area can have a different wind-price relationship and different dynamics.

## Data treatment philosophy
The project uses hourly market data over a long horizon and applies a consistent preprocessing pipeline to improve comparability and model fit.

In plain terms, preprocessing is intended to remove mechanical effects that can obscure the coefficient interpretation:
- Handle missingness in a consistent way
- Treat negative prices before logs
- Log-transform selected variables
- Remove strong seasonal patterns where appropriate
- Treat extreme outliers in the price series with explicit rules

This is not presented as “data cleaning for convenience,” but as a deliberate econometric design choice so that coefficient comparisons across time are meaningful.

## How time variation is studied
The project estimates the wind effect repeatedly in rolling windows, creating a time series of wind-coefficient estimates.

That derived coefficient series is then analyzed for structural changes. Conceptually:
1. Estimate local wind effects over many windows.
2. Ask whether those estimated effects can be described by one stable trend, or whether there are statistically supported break regimes.

This gives a direct way to test whether the wind-price relationship appears to have changed structurally over the sample.

## Structural break logic used here
Two break-selection perspectives are used in the project:

1. Information-criterion perspective (BIC)
- Compare candidate models with different numbers of breaks.
- Penalize complexity.
- Select the model balancing fit and parsimony.

2. Sequential hypothesis-testing perspective (Bai-Perron style)
- Test no additional break versus one additional break at each step.
- Continue while the null is rejected.
- Stop at first non-rejection.

The sequential test is used as the primary inferential logic for break count, while information criteria are used as complementary support.

## Important interpretation caveat
The structural-break analysis is applied to a generated series of rolling coefficients, not directly to the original raw equation in one single estimation step. So interpretation is: “evidence of structural change in estimated coefficient dynamics,” not a claim that every detected break is automatically a clean economic regime switch in the underlying data-generating process.

This is why robustness across design choices matters.

## Why rolling design choices matter
There is a deliberate tradeoff between:
- More overlap (smoother coefficient path, but stronger serial dependence), and
- Less overlap (more independent points, but fewer observations for break testing).

Because of this, conclusions are based on whether break signals are stable across reasonable window/step choices, not on a single run.

## What counts as a credible empirical conclusion
A break narrative is considered strongest when:
1. Sequential tests support it.
2. Information criteria are directionally consistent.
3. Similar break timing appears across nearby rolling designs.
4. The result remains economically interpretable (not just statistically possible).

In other words, the project prioritizes robust and explainable structural patterns over maximizing the number of statistically detectable breaks.

## What to tell another model/tool about this project
If you need to hand this project to another chatbot or analyst, the key message is:

This is a zone-level Swedish electricity-price thesis that estimates wind’s effect under multivariate controls and then evaluates whether that effect is stable over time using rolling estimation plus structural break testing (including Bai-Perron-style sequential logic). The emphasis is on methodological rigor, robustness across rolling designs, and interpretable structural narratives rather than one-shot point estimates.
