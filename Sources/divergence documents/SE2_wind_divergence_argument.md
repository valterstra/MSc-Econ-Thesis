# The SE1-SE2 Wind Coefficient Divergence: Full Chain of Argumentation

---

## The phenomenon

SE1 and SE2 had nearly identical wind coefficients in 2023 — both around -0.02. By 2025, SE1 stays at -0.014 while SE2 deepens to -0.089. A fourfold divergence between two zones with broadly similar generation mixes. That is what needs explaining.

---

## The event

On 29 October 2024, the Nordic day-ahead market transitioned from Available Transfer Capacity to flow-based market coupling. Under ATC, each border had a fixed administrative capacity ceiling published for each hour. Under FBMC, capacity between Swedish zones is no longer computed as a bilateral number — it emerges from simultaneous optimisation across Critical Network Elements in the whole Nordic grid. The data structure confirms this directly: AuctionCapacity stops reporting SE2-SE3 as a series after 30 October 2024 because that bilateral concept no longer exists in the market clearing.

---

## What the SE2-SE3 corridor looked like before the transition

From the 2023 AuctionCapacity and AuctionFlow data: the SE2→SE3 corridor was operating at a mean allocated capacity of 6,288 MW — already below the nominal 7,300 ATC ceiling — and flow reached that allocated ceiling in 26% of hours. Median utilisation was 83%. The corridor was heavily loaded in the typical hour and regularly hitting its limit. This is confirmed independently by the coupling frequency table: SE2 and SE3 had the same price in 77% of 2023 hours, meaning the corridor was binding and producing price divergence in roughly 23% of hours — the same number as the flow-based measure from a completely independent data source.

---

## What happened to SE2-SE3 after the transition

In 2025, SE2 and SE3 have the same price in only 4.1% of hours. The average spread when diverging is 30.55 EUR/MWh. SE3 was cheaper than SE2 in 1,660 hours — a pattern that never occurred once in any prior year. The price evidence implies the effective constraint on SE2→SE3 is binding in approximately 96% of hours. The mechanism is that FBMC activates dynamic CNE constraints — internal Swedish transmission lines whose limits are computed fresh each hour — which bind in the vast majority of hours given the persistent north-to-south generation surplus. A single-hour PTDF matrix confirmed this structure directly: the CNEs with the highest SE2 sensitivity have tight RAM values, while SE3's PTDF on those same lines is near zero. SE3's net position barely loads the critical internal lines; SE2's loads them heavily. That is the physical reason SE2 and SE3 decouple while SE3 is largely unaffected.

---

## Why wind surplus in SE2 is now trapped

When SE2 produces more wind, its surplus needs to move southward into SE3 — where demand is. In 2023, the corridor accommodated that movement in 74% of hours. In 2025, it cannot in 96% of hours. The surplus has nowhere to go. It accumulates within SE2's thin, locally-priced market and suppresses SE2's price directly. The wind coefficient deepens because the price impact of wind is no longer shared across a larger market — it is contained entirely within SE2.

---

## What the SE1-SE2 corridor looked like in the same period

From the 2023 AuctionCapacity and AuctionFlow data for SE1: the SE1→SE2 corridor was at its allocated ceiling in 0.3% of hours. Median utilisation was 35.6%. In the typical hour, SE1 was using barely a third of its available export capacity into SE2. The outlet was structurally open — not occasionally open, but open in virtually every hour of the year. In 2025, the SE1-SE2 price spread is 0.48 EUR/MWh on average — essentially zero — confirming that despite FBMC technically declaring the SE1-SE2 border constrained in 95% of hours, the economic reality is that SE1 and SE2 remain the same market. SE2 is always available to absorb SE1's wind surplus because SE2's own price is suppressed relative to SE3, making it a willing destination for additional supply.

---

## Why SE1's wind coefficient stayed flat

SE1 sends surplus into SE2. SE2 accepts it because SE2 is already cheaper than SE3 and has demand for additional supply. SE1's wind surplus drains freely in essentially every hour. The price impact of wind in SE1 is not trapped locally — it disperses into a larger effective market. The coefficient stays small.

---

## The complete explanation

The FBMC transition of 29 October 2024 activated binding constraints on the SE2-SE3 corridor in approximately 96% of 2025 hours, up from 26% under ATC in 2023. This blocked SE2's southward outlet in nearly every hour, trapping wind surplus in SE2's thin local market and deepening the wind coefficient from -0.02 to -0.089. SE1's outlet into SE2 was barely utilised at 35.6% median capacity in 2023 and remained economically open throughout — confirmed by a 0.48 EUR/MWh price spread in 2025 — so SE1's wind surplus continued to drain freely and its coefficient stayed flat. The fourfold divergence is explained entirely by this asymmetry in outlet availability, which was structurally predetermined before FBMC and made near-permanent by it.
