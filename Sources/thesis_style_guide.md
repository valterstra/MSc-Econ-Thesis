# Thesis Writing Style Guide
## For use with Claude when writing economics thesis sections

---

## 1. GENERAL PRINCIPLES

- Every sentence must either (a) report a result, (b) interpret a result, (c) connect to prior literature, or (d) set up the next point. If it does none of these, delete it.
- One idea per sentence. If a sentence requires more than one comma clause to make its point, split it.
- No filler openers. Never begin a sentence with: "It is worth noting that", "Importantly,", "It is interesting that", "One can observe that", "It is clear that", or any similar throat-clearing phrase.
- No word inflation. Use the shortest word that is precise. "Use" not "utilise". "Show" not "demonstrate". "Find" not "ascertain".
- Avoid adverbs that hedge without adding information: "relatively", "somewhat", "quite", "rather". If a result is significant, state the number.

---

## 2. RESULTS SECTIONS

### Introducing tables and figures
- Always refer to tables and figures by number before or at the point where the reader needs them: "Table 6.1 reports..." or "As shown in Figure 6.2..."
- Never write "the table below shows" — always use the numbered reference.
- The first sentence about a table should state what the table contains, not what it shows. The second sentence should state the key finding.

### Reporting coefficients
- Always report the coefficient value alongside its significance: "$\hat{\beta}_{\text{wind}} = -0.035$ (significant at the 1\% level)"
- Never write "the coefficient is negative and significant" without also stating the magnitude.
- When comparing across zones or models, anchor to the strongest or most relevant result first, then describe deviations from it.
- Use the correct tense: present tense for what the table shows ("Table 6.1 reports..."), past tense for what the estimation found ("the wind coefficient was negative...").

### Interpreting results
- Every coefficient that is discussed must be given an economic interpretation, not just a statistical one. "The coefficient of $-0.103$ implies that a 1\% increase in wind power production is associated with a $0.103\%$ decrease in the day-ahead spot price."
- Do not over-claim. "is associated with" not "causes". Unless you have an identification strategy that supports causal language.
- When a result is insignificant, say so directly: "The oil price coefficient is statistically insignificant across SE1--SE3, suggesting limited pass-through at the daily frequency."

---

## 3. METHODOLOGY SECTIONS

- Describe what the model does, not why it is good. Justification belongs in one sentence, not a paragraph.
- Equations should be numbered and referenced in the text by number.
- Every variable in an equation must be defined in the text immediately following the equation. No exceptions.
- The order of variable definitions in the text should follow the order they appear in the equation.
- When selecting a specification (e.g. ARMA order), state the criterion used, the result, and move on. Do not over-justify.

---

## 4. HEDGING AND CONFIDENCE

- Results from a regression are findings, not opinions. State them with appropriate confidence.
- Reserve hedging language ("may suggest", "could indicate") for interpretations that go beyond what the data directly show — for example, causal claims or forward-looking implications.
- When referring to the existing literature to support an interpretation, cite specifically: "(Ketterer 2014)" not "as shown in previous research".

---

## 5. STRUCTURE AND FLOW

### Paragraphs
- Each paragraph has one purpose. State it in the first sentence (topic sentence), develop it in the middle, and do not summarise it at the end — that is redundant.
- Paragraphs in a results section should follow this logic: (1) what does the table/figure show, (2) what is the key result, (3) what does it mean economically, (4) how does it relate to prior literature or the next result.

### Transitions
- Do not use transitional padding: "Having established X, we now turn to Y." Simply begin the next point.
- Use transitional words only when the logical connection is not obvious: "however", "by contrast", "consistent with this".

---

## 6. LANGUAGE CONVENTIONS FOR ECONOMICS THESES

- Use "we" throughout (this is co-authored). Do not mix "we" and "the authors".
- Passive voice is acceptable for methodology: "The model is estimated using QML." Active voice is preferred for findings: "SE4 exhibits the strongest wind coefficient."
- Spell out numbers below ten unless they are coefficients, percentages, or statistical values.
- Percentages: always use the \% symbol when paired with a number: "a 1\% increase", not "a one percent increase".
- P-values and significance: "significant at the 1\% level" or use stars with a table note. Do not mix conventions within the same document.
- When referring to a zone by name, always use the format SE1, SE2, SE3, SE4 — never "zone 1" or "the first zone".
- Refer to the dependent variable consistently throughout: "the log day-ahead spot price" — pick one formulation and use it everywhere.

---

## 7. THINGS TO NEVER WRITE

- "As can be seen from..." → just say what is seen
- "It is important to note that..." → just note it
- "This is consistent with the literature" → cite the specific paper
- "The results are interesting" → say why or delete
- "We can observe that" → just state the observation
- "In order to" → use "to"
- "Due to the fact that" → use "because"
- "It should be noted that" → delete entirely
- "The above results suggest that" → "These results suggest that" or just make the point directly

---

## 8. SPECIFIC TO THIS THESIS

- The coefficient of primary interest is always $\beta_{\text{wind}}$. It measures the elasticity of the log day-ahead spot price with respect to log wind power production.
- The four zones are SE1 (Northern Sweden, hydro-dominated, net exporter), SE2 (North-Central, hydro-dominated, net exporter), SE3 (South-Central, nuclear-heavy, net importer), SE4 (Southern, wind-heavy at 63.6\% of production, net importer).
- The sample period for baseline results is 1 January 2024 to 31 December 2025. The rolling window analysis covers 2015 to 2025 in 1-year windows with 1-year steps.
- The structural break is modelled at October 2021 (onset of the European energy crisis).
- Always refer to the ARMA order selection as a "grid search over $p, q \in \{0, 1, 2, 3\}$" and the selected specification as "ARMA(2,1)".
- When discussing rolling window results, refer to the x-axis variable as "window midpoint", not "year" or "date".
