import random
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Macro Policy Simulator", layout="wide"
)

# Fixed seed so shocks are reproducible run-to-run for a given policy path
random.seed(42)


# ---------------------------------------------------------------------------
# COUNTRY CALIBRATION PROFILES
#
# Parameters with a clean single real-world analog (NAIRU, neutral rate,
# Okun's coefficient, GDP/debt levels, policy lag) are set from cited
# published estimates - see README.md for full sourcing.
#
# Fiscal composition parameters (LABOR_SHARE, PROFIT_SHARE,
# GOV_SPENDING_SHARE) are necessarily stylized: this model has only two
# tax instruments (income + corporate), while real federal budgets also
# collect consumption taxes we don't represent. Rather than inflate the
# corporate tax base to force-match official total revenue, spending
# shares are calibrated so the BASELINE DEFICIT matches each country's
# actual reported deficit-to-GDP ratio - the number that actually drives
# gameplay - even though the underlying revenue composition is simplified.
# ---------------------------------------------------------------------------
CALIBRATIONS = {
    "Canada": dict(
        currency="CAD",
        neutral_nominal_rate=2.75,      # BoC 2025 neutral rate assessment (2.25-3.25%)
        target_inflation=2.0,           # BoC inflation target
        natural_unemployment=5.5,       # Consensus pre-2020 Canadian NAIRU estimate
        potential_growth_q=0.003,       # ~1.2%/yr, reflecting BoC's slower 2025 potential growth outlook
        okun_coefficient=0.40,          # Ball, Leigh & Loungani (2013), G7 average
        rate_lag_quarters=3,            # BoC: effect "economically significant by 3rd-4th quarter"
        inflation_lag_weight=0.39,      # Fillion & Léonard (1997) Canadian Phillips curve estimate
        inflation_target_weight=0.61,
        labor_share=0.45,
        profit_share=0.15,
        gov_spending_share=0.10,        # calibrated for ~1.8% GDP baseline deficit (actual FY24-25 figure)
        base_income_tax=18.0,           # approx. average effective federal PIT rate
        base_corp_tax=15.0,             # actual Canada federal general corporate tax rate
        base_ei_benefit=3.0,            # $B/quarter per 1pt of unemployment
        ei_range=(1.0, 10.0, 0.5),
        starting_gdp=3075.0,            # $B CAD, implied by federal debt / 41.2% debt-to-GDP (FY24-25 Annual Financial Report)
        starting_debt=1266.5,           # $B CAD, federal debt at March 31, 2025
    ),
    "United States": dict(
        currency="USD",
        neutral_nominal_rate=3.0,       # Within HLW (~2.84%) / Cleveland Fed (~3.7%) / BoC cross-country range
        target_inflation=2.0,           # Fed inflation target
        natural_unemployment=4.5,       # CBO 2025-26 NAIRU estimate
        potential_growth_q=0.005,       # ~2%/yr, standard US potential growth estimate
        okun_coefficient=0.45,          # Ball, Leigh & Loungani (2013), US-specific estimate
        rate_lag_quarters=4,            # Commonly cited ~1yr lag to peak output effect
        inflation_lag_weight=0.55,      # More backward-looking; less rigorously sourced than the Canadian figure
        inflation_target_weight=0.45,
        labor_share=0.50,
        profit_share=0.12,
        gov_spending_share=0.146,       # calibrated for ~6.8% GDP baseline deficit (actual 2025 figure)
        base_income_tax=14.0,           # approx. average effective federal individual income tax rate
        base_corp_tax=21.0,             # actual US federal corporate tax rate (post-TCJA)
        base_ei_benefit=28.0,           # $B/quarter per 1pt of unemployment (~9.4x Canada, matching GDP ratio)
        ei_range=(10.0, 100.0, 5.0),
        starting_gdp=29000.0,           # $B USD, 2025 nominal GDP estimate
        starting_debt=36000.0,          # $B USD, ~124% debt-to-GDP
    ),
}

# Shared structural elasticities - standard textbook/DSGE-calibration magnitudes,
# not fitted per-country. See README.md for discussion.
IS_PERSISTENCE = 0.75
IS_RATE_SENSITIVITY = 0.55
PHILLIPS_OUTPUT_GAP_SLOPE = 0.30

# Fiscal multipliers - literature-cited ranges (see README.md):
#   income tax: Barro & Redlick (2011) ~1.1; CBO 0.3-1.5
#   EI/transfers: CBO transfer-payment range, high end ~1.7-2.1 (high MPC recipients)
#   corporate tax: FAVAR estimate ~0.83 (Heterogeneous Responses to US Narrative Tax Changes)
INCOME_TAX_MULTIPLIER = 1.1
EI_MULTIPLIER = 1.7
CORP_TAX_MULTIPLIER = 0.83


class EconomyModel:
    """Reduced-form macro model: IS curve (demand) + Phillips curve
    (inflation) + Okun's law (unemployment), with a lagged monetary
    transmission channel and literature-calibrated fiscal multipliers.
    Supports Canada and US calibration profiles - see CALIBRATIONS above
    and README.md for full sourcing.
    """

    def __init__(self, country="Canada"):
        cal = CALIBRATIONS[country]
        self.country = country
        self.currency = cal["currency"]

        # Structural calibration (not player-controlled)
        self.NEUTRAL_REAL_RATE = cal["neutral_nominal_rate"] - cal["target_inflation"]
        self.TARGET_INFLATION = cal["target_inflation"]
        self.NATURAL_UNEMPLOYMENT = cal["natural_unemployment"]
        self.POTENTIAL_GROWTH_Q = cal["potential_growth_q"]
        self.OKUN_COEFFICIENT = cal["okun_coefficient"]
        self.RATE_LAG_QUARTERS = cal["rate_lag_quarters"]
        self.INFLATION_LAG_WEIGHT = cal["inflation_lag_weight"]
        self.INFLATION_TARGET_WEIGHT = cal["inflation_target_weight"]
        self.LABOR_SHARE = cal["labor_share"]
        self.PROFIT_SHARE = cal["profit_share"]
        self.GOV_SPENDING_SHARE = cal["gov_spending_share"]
        self.BASE_INCOME_TAX = cal["base_income_tax"]
        self.BASE_CORP_TAX = cal["base_corp_tax"]
        self.BASE_EI_BENEFIT = cal["base_ei_benefit"]
        self.ei_range = cal["ei_range"]

        # Policy levers (player controlled) - start at baseline
        self.interest_rate = cal["neutral_nominal_rate"]
        self.income_tax_rate = self.BASE_INCOME_TAX
        self.corp_tax_rate = self.BASE_CORP_TAX
        self.ei_benefit = self.BASE_EI_BENEFIT

        # State
        self.quarter = 0
        self.output_gap = 0.0
        self.inflation = self.TARGET_INFLATION
        self.unemployment = self.NATURAL_UNEMPLOYMENT
        self.potential_gdp = cal["starting_gdp"]
        self.gdp = cal["starting_gdp"]
        self.debt = cal["starting_debt"]
        self._last_growth = 0.0
        self._last_primary_balance = 0.0

        # Real-rate history, needed for the lagged transmission channel
        self.rate_history = [self.interest_rate - self.inflation]

        self.history = []
        self._record()

    def _record(self):
        self.history.append({
            "Quarter": self.quarter,
            "Interest Rate (%)": round(self.interest_rate, 2),
            "Real Interest Rate (%)": round(self.interest_rate - self.inflation, 2),
            "Unemployment (%)": round(self.unemployment, 2),
            "Inflation (%)": round(self.inflation, 2),
            "GDP Growth (%)": round(self._last_growth, 2),
            "GDP ($B)": round(self.gdp, 1),
            "Federal Debt ($B)": round(self.debt, 1),
            "Debt-to-GDP (%)": round(self.debt / self.gdp * 100.0, 1),
            "Primary Balance ($B)": round(self._last_primary_balance, 1),
            "Output Gap (%)": round(self.output_gap, 2),
        })

    def step(self):
        self.quarter += 1

        # --- Monetary transmission (lagged real interest rate) ---
        expected_inflation = self.inflation  # adaptive expectations
        real_rate_now = self.interest_rate - expected_inflation
        self.rate_history.append(real_rate_now)
        if len(self.rate_history) > self.RATE_LAG_QUARTERS:
            real_rate_effective = self.rate_history[-1 - self.RATE_LAG_QUARTERS]
        else:
            real_rate_effective = self.NEUTRAL_REAL_RATE

        # --- Fiscal impulse: literature multiplier x fiscal shock (% of GDP) ---
        # Shock = size of the revenue/spending change as a share of GDP;
        # long-run output-gap contribution = multiplier x shock (see README).
        # Each quarter only (1 - IS_PERSISTENCE) of that long-run effect
        # lands immediately, converging to the full effect over time.
        income_tax_shock_pct = (self.BASE_INCOME_TAX - self.income_tax_rate) * self.LABOR_SHARE / 100.0
        corp_tax_shock_pct = (self.BASE_CORP_TAX - self.corp_tax_rate) * self.PROFIT_SHARE / 100.0
        quarterly_output_prior = self.gdp / 4.0
        ei_shock_dollars = (self.ei_benefit - self.BASE_EI_BENEFIT) * self.NATURAL_UNEMPLOYMENT
        ei_shock_pct = ei_shock_dollars / quarterly_output_prior

        fiscal_impulse = (1 - IS_PERSISTENCE) * (
            INCOME_TAX_MULTIPLIER * income_tax_shock_pct * 100.0
            + EI_MULTIPLIER * ei_shock_pct * 100.0
            + CORP_TAX_MULTIPLIER * corp_tax_shock_pct * 100.0
        )

        demand_shock = random.gauss(0, 0.08)

        # --- IS curve: output gap ---
        self.output_gap = (
            IS_PERSISTENCE * self.output_gap
            - IS_RATE_SENSITIVITY * (real_rate_effective - self.NEUTRAL_REAL_RATE)
            + fiscal_impulse
            + demand_shock
        )
        self.output_gap = max(-10.0, min(10.0, self.output_gap))

        # --- Phillips curve: inflation ---
        supply_shock = random.gauss(0, 0.05)
        self.inflation = (
            self.INFLATION_LAG_WEIGHT * self.inflation
            + self.INFLATION_TARGET_WEIGHT * self.TARGET_INFLATION
            + PHILLIPS_OUTPUT_GAP_SLOPE * self.output_gap
            + supply_shock
        )
        self.inflation = max(-2.0, min(12.0, self.inflation))

        # --- Okun's law: unemployment ---
        self.unemployment = self.NATURAL_UNEMPLOYMENT - self.OKUN_COEFFICIENT * self.output_gap
        self.unemployment = max(1.0, min(20.0, self.unemployment))

        # --- GDP level (annualized), potential compounds, actual tracks the gap ---
        prev_gdp = self.gdp
        self.potential_gdp *= (1 + self.POTENTIAL_GROWTH_Q)
        self.gdp = self.potential_gdp * (1 + self.output_gap / 100.0)
        self._last_growth = ((self.gdp / prev_gdp) ** 4 - 1) * 100.0  # annualized

        # --- Federal debt: revenue vs spending vs interest owed ---
        quarterly_output = self.gdp / 4.0
        income_tax_rev = (self.income_tax_rate / 100.0) * self.LABOR_SHARE * quarterly_output
        corp_tax_rev = (self.corp_tax_rate / 100.0) * self.PROFIT_SHARE * quarterly_output
        gov_other_spend = self.GOV_SPENDING_SHARE * quarterly_output
        ei_spend = self.ei_benefit * self.unemployment
        interest_on_debt = self.debt * (self.interest_rate / 100.0) / 4.0

        primary_balance = income_tax_rev + corp_tax_rev - gov_other_spend - ei_spend
        self._last_primary_balance = primary_balance
        self.debt = self.debt - primary_balance + interest_on_debt

        self._record()


# --- STREAMLIT DASHBOARD UI ---
st.title("🏦 Sim-Fed: Macroeconomic Policy Simulator")
st.markdown(
    "Take control of a central bank and finance ministry. Adjust policy levers and observe macro outcomes."
)

country = st.sidebar.selectbox("Country", list(CALIBRATIONS.keys()))

# Reset the model whenever the country selection changes, since starting
# conditions (GDP, debt, NAIRU, etc.) differ per profile.
if "model" not in st.session_state or st.session_state.get("country") != country:
    st.session_state.model = EconomyModel(country=country)
    st.session_state.country = country

model = st.session_state.model

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Policy Levers")

interest_rate = st.sidebar.slider(
    "Central Bank Policy Rate (%)",
    min_value=0.25,
    max_value=10.00,
    value=float(model.interest_rate),
    step=0.25,
)

income_tax = st.sidebar.slider(
    "Income Tax Rate (%)",
    min_value=5.0,
    max_value=45.0,
    value=float(model.income_tax_rate),
    step=1.0,
)

corp_tax = st.sidebar.slider(
    "Corporate Tax Rate (%)",
    min_value=5.0,
    max_value=40.0,
    value=float(model.corp_tax_rate),
    step=1.0,
)

ei_min, ei_max, ei_step = model.ei_range
ei_benefit = st.sidebar.slider(
    f"EI/UI Benefit (${model.currency} B / quarter per 1pt unemployment)",
    min_value=ei_min,
    max_value=ei_max,
    value=float(model.ei_benefit),
    step=ei_step,
)

# Apply Control Settings to Active Model
model.interest_rate = interest_rate
model.income_tax_rate = income_tax
model.corp_tax_rate = corp_tax
model.ei_benefit = ei_benefit

st.sidebar.caption(
    f"1 tick = 1 quarter, matching real central bank meeting cadence. "
    f"Rate changes take ~{model.RATE_LAG_QUARTERS} quarters to fully hit the economy, like real policy lags."
)

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
if col_btn1.button("▶ 1 Quarter"):
    model.step()

if col_btn2.button("▶▶ 1 Year"):
    for _ in range(4):
        model.step()

if col_btn3.button("🔄 Reset"):
    st.session_state.model = EconomyModel(country=country)
    st.rerun()

# --- MAIN DISPLAY METRICS ---
df = pd.DataFrame(model.history)

if len(df) > 1:
    latest = df.iloc[-1]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Unemployment", f"{latest['Unemployment (%)']}%")
    m2.metric("GDP Growth", f"{latest['GDP Growth (%)']}%")
    m3.metric("Inflation", f"{latest['Inflation (%)']}%")
    m4.metric("GDP Level", f"${latest['GDP ($B)']:,.1f}B")
    m5.metric("Federal Debt", f"${latest['Federal Debt ($B)']:,.1f}B")
    m6.metric("Debt-to-GDP", f"{latest['Debt-to-GDP (%)']}%")

    st.divider()

    # --- CHARTS ---
    c1, c2 = st.columns(2)
    n_quarters = len(df)

    with c1:
        st.subheader("Labor Market & Interest Rates")
        fig1 = px.line(
            df,
            x="Quarter",
            y=["Unemployment (%)", "Interest Rate (%)"],
            title="Unemployment vs Policy Rate",
            labels={"value": "Percent (%)", "variable": "Metric"},
        )
        fig1.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=40))
        dtick1 = max(1, round(n_quarters / 10))
        fig1.update_xaxes(dtick=dtick1, tick0=0, tickformat="d")
        st.plotly_chart(fig1, width="stretch")

    with c2:
        st.subheader("Economic Output & Inflation")
        fig2 = px.line(
            df,
            x="Quarter",
            y=["GDP Growth (%)", "Inflation (%)"],
            title="GDP Growth Rate vs Inflation",
            labels={"value": "Percent (%)", "variable": "Metric"},
        )
        fig2.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=40))
        dtick2 = max(1, round(n_quarters / 10))
        fig2.update_xaxes(dtick=dtick2, tick0=0, tickformat="d")
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Fiscal Sustainability")
        fig3 = px.line(
            df,
            x="Quarter",
            y="Debt-to-GDP (%)",
            title="Federal Debt as % of GDP",
            labels={"value": "Percent (%)"},
        )
        fig3.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=40))
        dtick3 = max(1, round(n_quarters / 10))
        fig3.update_xaxes(dtick=dtick3, tick0=0, tickformat="d")
        st.plotly_chart(fig3, width="stretch")

    with c4:
        st.subheader("Budget Balance")
        fig4 = px.line(
            df,
            x="Quarter",
            y="Primary Balance ($B)",
            title="Primary Balance (surplus above zero, before interest)",
            labels={"value": "$B / quarter"},
        )
        fig4.add_hline(y=0, line_dash="dot", line_color="gray")
        fig4.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=40))
        dtick4 = max(1, round(n_quarters / 10))
        fig4.update_xaxes(dtick=dtick4, tick0=0, tickformat="d")
        st.plotly_chart(fig4, width="stretch")

    st.subheader("Simulation Ledger")
    st.dataframe(df.tail(15), width="stretch")
else:
    st.info("Click **1 Quarter** or **1 Year** in the sidebar to begin.")