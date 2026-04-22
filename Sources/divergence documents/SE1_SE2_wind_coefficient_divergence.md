# SE1–SE2 Wind Coefficient Divergence: Mechanism and Explanation
### From conversation — SSE Master's Thesis

---

## The Phenomenon

SE1 and SE2 had nearly identical wind coefficients in 2023 — both around -0.02. By 2025, SE1 stays flat at approximately -0.014 while SE2 deepens to approximately -0.089. A fourfold divergence between two zones with broadly similar generation mixes. The question is why.

---

## Background: Why Transmission Limits Exist

Transmission lines are physical objects with hard thermal limits. If too much current flows through a wire, it overheats and fails — potentially cascading into a blackout. The day-ahead market clearing algorithm must never produce a solution where a physical line is overloaded. Transmission limits exist to enforce this. They are not administrative preferences — they are the market's way of reflecting physical engineering constraints.

---

## The Old Regime: ATC (Before 29 October 2024)

### What ATC was

Under the Available Transfer Capacity methodology, each border between bidding zones received a single fixed hourly capacity ceiling, pre-computed by engineers and published before the auction. For SE2→SE3, this ceiling was approximately 6,300 MW. The market clearing algorithm treated this number as a hard constraint: no more than 6,300 MW would be scheduled to flow from SE2 to SE3 in any given hour.

### The fundamental problem with ATC

To know the true safe capacity on a corridor, you would need to already know the full market solution — who is producing, who is consuming, and therefore how the entire meshed grid is loaded. But you cannot know that before the auction runs. It is circular.

ATC broke this circularity by having engineers pre-compute a conservative bilateral estimate. That estimate was necessarily based on assumptions about what the rest of the grid would be doing. It did not account for how a transaction between SE2 and SE3 simultaneously loads lines elsewhere in Norway, Finland, and Sweden.

### What this meant in practice

The ATC ceiling was real and enforced — the algorithm would not schedule more flow than the published number. But the ceiling was an approximation of the true physical limit, not the true physical limit itself.

For SE2→SE3 specifically, the ATC turned out to be **too generous** — it permitted more scheduled flow than the physical grid could safely handle in many hours. When the algorithm scheduled 6,300 MW southward, the physical grid operator (Svenska kraftnät) would intervene after the market closed, ordering SE2 generators to produce less and SE3 generators to produce more. This is called redispatch. The physical flows were corrected; the day-ahead prices were not.

### The consequence for day-ahead prices

The day-ahead spot price is set by the algorithm at the point of clearing. If the algorithm believes 6,300 MW can flow from SE2 to SE3, it sets SE2 and SE3 at the same price — treating them as one combined market. SE3 consumers received the cheap SE2 price. The redispatch cost that corrected the physical reality afterwards was recovered through grid tariffs, spread invisibly across all market participants. The price divergence that should have reflected the true congestion never appeared in the spot market.

---

## The New Regime: FBMC (From 29 October 2024)

### What changed

On 29 October 2024, the Nordic day-ahead market transitioned from ATC to Flow-Based Market Coupling (FBMC). This is a pan-European regulatory framework implemented region by region: the Core CCR (France, Germany, Belgium, Netherlands, and others) transitioned in June 2022; the Nordic CCR followed on 29 October 2024.

Under FBMC, the market does not use pre-computed bilateral border capacities. Instead, the clearing algorithm has the full grid physics embedded inside it. It uses Power Transfer Distribution Factors (PTDFs) — which describe how much a change in any zone's net position loads each physical transmission line — and clears the market subject to constraints on those physical lines, called Critical Network Elements (CNEs).

### What this means structurally

The SE2→SE3 border no longer has a fixed capacity number. Instead, the algorithm computes in each hour how much SE2's net position loads the critical internal Swedish transmission lines. Those lines — the 400kV corridor and associated structural cuts between SE2 and SE3 — are the binding CNEs in approximately 76% of post-transition hours. When they are near their limit, the effective SE2→SE3 capacity collapses, not because of a bilateral ceiling but because the physics of the meshed grid will not allow more.

Crucially, the algorithm now knows this at the point of price-setting. The price divergence that previously appeared only in redispatch costs now appears directly in the day-ahead spot price. The hidden congestion became visible.

### The empirical result

- In 2023, SE2 and SE3 shared the same price in 77% of hours. The SE2→SE3 corridor was binding in approximately 26% of hours under the ATC ceiling.
- In 2025, SE2 and SE3 share the same price in only 4.1% of hours. The average spread when diverging is 30.55 EUR/MWh. SE3 was cheaper than SE2 in 1,660 hours — a pattern that never occurred in any prior year.

The transition date is confirmed by multiple authoritative sources: Nordic RCC, Nord Pool, EPEX SPOT, and Nordic Balancing Model all announced go-live on 29 October 2024. The Nordic TSOs' June 2025 follow-up report to Nordic regulators documents the SE2→SE3 intraday capacity falling by approximately 50% and intraday traded volumes falling by a factor of ten — confirming this is structural, not temporary.

---

## Why the SE2 Wind Coefficient Deepened

### What β_wind measures

β_wind is the elasticity of the log day-ahead spot price with respect to log wind power production. A more negative coefficient means wind has a stronger price-suppressing effect in that zone.

### The mechanism under ATC

When wind production rose in SE2, local supply increased and the SE2 price began to fall. Under ATC, the algorithm believed the SE2→SE3 corridor could carry up to 6,300 MW. It scheduled surplus wind to flow south. SE3 absorbed it. The price impact of the additional wind was shared across the combined SE2-plus-SE3 market. The price suppression was diluted into a large pool. β_wind in SE2 remained modest because the algorithm was not pricing SE2's wind against SE2's local demand alone — it was pricing it into a much larger effective market.

### The mechanism under FBMC

Now when wind rises in SE2, the algorithm knows the SE2→SE3 CNEs are already near their limits. It cannot schedule large southward flows. The wind surplus stays in SE2. The algorithm prices that surplus against SE2's own local demand, which is thin — SE2 is a geographically large zone with relatively modest consumption. The same increment of wind hits a smaller pool and suppresses the price further.

β_wind deepened because the effective market the algorithm prices SE2's wind into shrank from SE2-plus-SE3 to SE2-alone. The supply shock is the same; the demand pool absorbing it is smaller; the price response is larger.

### The key distinction

The physical grid did not change on 29 October 2024. What changed is what the pricing algorithm knew and acted on. Under ATC the algorithm priced SE2 wind into a large market because it believed that market was accessible. Under FBMC the algorithm prices SE2 wind into a small local market because it now correctly understands the physical constraints. The regression captures the day-ahead spot price — which is exactly what the algorithm sets. The coefficient deepened because the algorithm's model of SE2's market boundary changed.

---

## Why the SE1 Wind Coefficient Stayed Flat

### The apparent analogy

SE1 exports to SE2. By analogy with SE2→SE3, one might expect that FBMC tightened the SE1→SE2 corridor, trapped SE1's wind surplus locally, and deepened SE1's coefficient. FBMC does technically declare the SE1-SE2 border constrained in approximately 95% of hours post-transition.

### Why the analogy fails

The SE1-SE2 price spread in 2025 is 0.48 EUR/MWh on average. This is effectively zero. A constraint is only economically meaningful if it is preventing a trade that both sides want to make. With a 0.48 EUR/MWh spread, the SE1-SE2 constraint has no economic bite — the algorithm does not need to force large flows to equilibrate the two zones because they are already at nearly the same price.

Under ATC, the SE1→SE2 corridor was already barely used: median utilisation was 35.6%, and the ceiling was binding in only 0.3% of hours. SE1 was not a heavy exporter into SE2 to begin with.

### Why SE1 and SE2 remained one effective market

SE2's price collapsed under FBMC because its southward outlet closed. SE2 became cheap — suppressed by its own trapped surplus. SE1 looks at SE2 and sees a market already at SE1's price level. There is no large price differential pulling SE1's surplus southward under pressure. The algorithm can move SE1's wind surplus into SE2 freely because the economic shadow price of that constraint is near zero. SE1's wind is still priced into a combined SE1-plus-SE2 market, just as before.

The Fingrid/Nordic TSO pre-implementation documentation is directly consistent with this: "In the FB solution, it is not possible for market actors to sell in SE2 and buy in SE1 since an increased net position in SE2 would lead to an overload on the CNE in SE3." The SE1-SE2 constraint under FBMC is not an independent corridor constraint — it is a shadow of the SE2-SE3 CNEs. It is technically activated to protect the lines further south, not because SE1 and SE2 are genuinely economically separated.

### The consequence for β_wind in SE1

SE1's wind surplus continues to drain freely into SE2. The effective market SE1's wind is priced into did not shrink. The coefficient stayed flat.

---

## The Complete Explanation of the Divergence

The fourfold divergence between SE1 and SE2 wind coefficients from 2023 to 2025 is explained by a single asymmetry in outlet availability, which was structurally predetermined before FBMC and made near-permanent by it.

**SE2's southward outlet (→SE3):** Under ATC, generously open in the algorithm's pricing model — binding in only 26% of hours in the market solution. Under FBMC, the true physical constraint is revealed: binding in approximately 96% of hours. Wind surplus in SE2 is now priced into SE2's thin local market. Coefficient deepens from -0.02 to -0.089.

**SE1's southward outlet (→SE2):** Under ATC, structurally underutilised — binding at ceiling in only 0.3% of hours, median utilisation 35.6%. Under FBMC, technically constrained in 95% of hours but with a shadow price of 0.48 EUR/MWh — economically inert. SE1's wind surplus continues to drain into SE2 freely because SE1 and SE2 are already priced as one market. Coefficient stays at -0.014.

The mechanism is not that wind changed. It is not that physical infrastructure changed. It is that the pricing algorithm's understanding of market boundaries changed on 29 October 2024 — revealing a congestion structure that had always existed physically but had been approximated away by the ATC methodology for years.

---

## Key Sources

- Nordic TSOs (2025). *Follow-up report concerning Flow-Based implementation and ID ATCE — response from Nordic TSOs.* Nordic RCC. [Link](https://nordic-rcc.net/wp-content/uploads/2025/06/Follow-up-report-concerning-Flow-Based-implementation-and-ID-ATCE-%E2%80%93-response-from-Nordic-TSOs.pdf)
- Nordic RCC FBMC go-live confirmation: [Link](https://nordic-rcc.net/flow-based/)
- Nord Pool go-live announcement: [Link](https://www.nordpoolgroup.com/en/message-center-container/newsroom/exchange-message-list/2024/q4/nordic-flow-based-market-coupling-project-confirms-go-live-for-29th-october-2024/)
- Fingrid/Nordic TSOs principle approach welfare assessment: [Link](https://www.fingrid.fi/globalassets/dokumentit/fi/tiedotteet/sahkomarkkinat/2015/principle-approach-for-assessing-nordic-welfare-under-flow-based-methodology.pdf)
- Montel Analytics post-transition analysis (November 2024 – March 2025)
- Volue one-year anniversary analysis (October 2025)
