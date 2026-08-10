# Sim-Fed: Canadian Macroeconomic Policy Simulator

A policy simulator where you take the role of the Bank of Canada and the
Department of Finance, adjusting interest rates, taxes, and benefits, and
watching the effect on unemployment, inflation, GDP, and federal debt.

## Design note: why this isn't agent-based

The first version of this project was built as an agent-based model (ABM)
using [Mesa](https://mesa.readthedocs.io/) — individual `Household` and
`Firm` agents that hired, spent, and set prices on their own, with
macro metrics (unemployment, GDP, inflation) emerging from thousands of
individual decisions each tick.

That approach is a good fit when the *mechanism* you want to study is
emergent behavior from heterogeneous, interacting individuals — contact
networks in epidemic spread, spatial segregation dynamics, market
microstructure, and similar. It is a poor fit for a policy simulator whose
whole point is to teach a **known, predictable transmission mechanism**
(rate → demand → inflation/unemployment). In the ABM version, small-sample
hiring/firing decisions, discrete rounding in capacity targets, and random
firm selection for consumer spending injected noise that swamped the
actual interest-rate signal — moving the policy rate didn't visibly move
the metrics the way it does in reality, and the charts were jumpy in a way
that didn't correspond to any real economic story.

This version replaces the agent population with a **reduced-form
macroeconomic model** — the same family of equations used in standard
intro macro courses and other Fed-simulator teaching tools (e.g. MIT's
*Chair the Fed*). Government, Business, and Household behavior are still
all represented, but as aggregate equations rather than individual agents.

## The model

Each **tick is one calendar quarter**. Clicking "Advance 1 Year" runs 4
ticks. State carries forward between ticks (`EconomyModel` holds the full
history), so consecutive quarters feed into each other.

### Core equations

The model is a standard 3-equation New Keynesian system: an IS curve
(demand), a Phillips curve (inflation), and Okun's law (unemployment) —
plus a fiscal/debt block.

**1. Output gap (IS curve)** — how far real GDP sits from its potential,
in percent. Driven by the *real* interest rate (lagged) and by fiscal
policy:

```
output_gap[t] = 0.75 * output_gap[t-1]
              - 0.55 * (real_rate_effective - neutral_real_rate)
              + fiscal_impulse
              + demand_shock
```

- `0.75` — persistence: demand conditions don't reset to zero every
  quarter, they decay gradually.
- `real_rate_effective` — the real policy rate from **2 quarters ago**
  (see "Monetary transmission lag" below), not the current-quarter rate.
- `neutral_real_rate` — 2.0%, the real rate consistent with stable
  inflation at potential output (a standard macro assumption).
- `demand_shock` — small random noise (`N(0, 0.08)`), fixed seed, so runs
  are reproducible. Present so the sim doesn't feel mechanically
  deterministic, but small enough that the policy signal dominates.
- Clamped to ±10%.

**2. Inflation (Phillips curve)** — adaptive-expectations Phillips curve:
inflation responds to the output gap, with persistence from
last quarter's inflation (proxy for inflation expectations):

```
inflation[t] = 0.65 * inflation[t-1]
             + 0.35 * target_inflation
             + 0.30 * output_gap[t]
             + supply_shock
```

- `target_inflation` — 2.0% (Bank of Canada's actual target).
- `supply_shock` — small random noise (`N(0, 0.05)`).
- Clamped to [-2%, 12%].

**3. Unemployment (Okun's law)** — unemployment moves opposite the output
gap:

```
unemployment[t] = natural_unemployment - 0.5 * output_gap[t]
```

- `natural_unemployment` — 5.0% (NAIRU).
- Clamped to [1%, 20%].

**4. GDP level** — potential GDP compounds at a steady long-run growth
rate; actual GDP tracks potential, scaled by the output gap:

```
potential_gdp[t] = potential_gdp[t-1] * (1 + 0.005)      # ~2%/yr, quarterly compounding
gdp[t]           = potential_gdp[t] * (1 + output_gap[t] / 100)
gdp_growth[t]     = ((gdp[t] / gdp[t-1]) ** 4 - 1) * 100  # annualized, like real GDP reporting
```

Starting level: $2,100B (annualized), roughly Canada's actual GDP scale.

### Monetary transmission lag

Real interest-rate policy does not affect the economy instantly — rate
decisions take time to filter into borrowing, investment, and spending
decisions. The model holds a rolling history of the real policy rate
(`nominal_rate - expected_inflation`, using last quarter's inflation as
the expectation) and the IS curve uses the value from **2 quarters
before the current one**. This is why a rate hike doesn't show up in
unemployment/GDP until roughly half a year later in the sim.

### Government sector

**Policy levers** (player-controlled):

| Lever | Range | Baseline |
|---|---|---|
| Central Bank Policy Rate | 0.25% – 10.00% | 4.0% |
| Income Tax Rate | 5% – 45% | 20% |
| Corporate Tax Rate | 5% – 40% | 20% |
| EI Benefit | $1B – $10B / quarter per 1pt of unemployment | $3B |

**Fiscal impulse** — how tax/benefit settings feed into aggregate demand
(the IS curve above), measured as deviation from baseline:

```
fiscal_impulse = 0.06 * (20.0 - income_tax_rate)     # tax cut -> stimulus
                + 0.15 * (ei_benefit - 3.0)           # richer EI -> stimulus
                + 0.02 * (20.0 - corp_tax_rate)       # corp tax cut -> mild stimulus
```

Income tax and EI benefits hit demand directly and quickly (they change
disposable income for a large share of the population). Corporate tax
has a much smaller coefficient — it works through business investment
decisions, a slower and less direct channel to aggregate demand than
household spending.

**Revenue and spending** (per quarter, all as a share of that quarter's
output, `quarterly_output = gdp / 4`):

```
income_tax_revenue = (income_tax_rate / 100) * 0.55 * quarterly_output   # 0.55 = labor share of output
corp_tax_revenue   = (corp_tax_rate / 100)   * 0.15 * quarterly_output   # 0.15 = profit share of output
gov_other_spending = 0.15 * quarterly_output                             # baseline non-EI spending
ei_spending        = ei_benefit * unemployment                          # EI lever x unemployment rate
interest_on_debt   = debt[t-1] * (interest_rate / 100) / 4

primary_balance = income_tax_revenue + corp_tax_revenue
                  - gov_other_spending - ei_spending

debt[t] = debt[t-1] - primary_balance + interest_on_debt
```

Starting debt: $1,200B.

**Note on structural balance:** at baseline settings (20% income tax, 20%
corp tax, unemployment near 5%), the government runs a **structural
deficit** — baseline spending (15% of output) exceeds baseline tax
revenue (~14% of output) even before EI or interest. Debt therefore
drifts upward over time unless taxes are raised or spending levers are
adjusted. This is a deliberate calibration choice, not a bug — real
governments frequently run structural deficits — but it means "debt is
rising" isn't automatically bad policy; what matters is whether it's
rising faster or slower than GDP (debt-to-GDP), and whether the real
interest rate on debt is above or below the economy's growth rate (the
classic "r vs. g" sustainability condition).

## Interpreting the levers together

- **Interest rate** — primary inflation-fighting tool, works with a
  ~2-quarter lag, trades off against unemployment in the short run
  (textbook Phillips curve tradeoff), and raises the government's own
  interest costs on debt.
- **Income tax** — fastest, strongest fiscal lever on demand (large
  `0.06` coefficient plus direct revenue effect); cuts stimulate but
  worsen the deficit unless the resulting growth (and larger tax base)
  offsets it.
- **Corporate tax** — weaker/slower demand effect than income tax, but
  still has a real, direct effect on federal revenue and debt.
- **EI benefit** — automatic-stabilizer-style lever: its cost scales with
  the unemployment rate itself, so it's cheap during booms and
  expensive during downturns, on top of its own stimulus effect on
  demand.

## Calibration constants

All structural parameters (not player-controlled) are defined at the top
of the `EconomyModel` class for easy retuning:

| Constant | Value | Meaning |
|---|---|---|
| `POTENTIAL_GROWTH_Q` | 0.5%/quarter | Long-run potential GDP growth (~2%/yr) |
| `NEUTRAL_REAL_RATE` | 2.0% | Real rate consistent with stable inflation |
| `TARGET_INFLATION` | 2.0% | Central bank's inflation target |
| `NATURAL_UNEMPLOYMENT` | 5.0% | NAIRU |
| `LABOR_SHARE` | 0.55 | Share of output subject to income tax |
| `PROFIT_SHARE` | 0.15 | Share of output subject to corporate tax |
| `GOV_SPENDING_SHARE` | 0.15 | Baseline non-EI government spending, % of output |
| `RATE_LAG_QUARTERS` | 2 | Quarters before a rate change fully hits demand |

## Known simplifications

- Inflation expectations are purely adaptive (last quarter's actual
  inflation), not forward-looking/rational — real expectations formation
  is more complex.
- No explicit exchange rate, trade, or external sector.
- No zero-lower-bound or unconventional monetary policy (QE) modeling.
- Corporate tax's effect on investment/hiring is captured only through
  its (small) demand coefficient and its direct revenue effect, not
  through a separate investment/capital-stock channel.


*Generated with help of Claude.*
