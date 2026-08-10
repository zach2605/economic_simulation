import random
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Canada Macro Policy Simulator", layout="wide"
)

# Fixed seed so shocks are reproducible run-to-run for a given policy path
random.seed(42)


class EconomyModel:
    """Reduced-form macro model: IS curve (demand) + Phillips curve
    (inflation) + Okun's law (unemployment), with a lagged monetary
    transmission channel. This replaces the old agent-based version so
    that interest-rate changes produce the textbook, predictable
    transmission a policy simulator needs, instead of emergent noise
    from thousands of individual hiring/spending decisions.
    """

    # --- Structural calibration (not player-controlled) ---
    POTENTIAL_GROWTH_Q = 0.005      # ~2%/yr potential growth, compounded quarterly
    NEUTRAL_REAL_RATE = 2.0         # % - real policy rate consistent with stable inflation
    TARGET_INFLATION = 2.0          # % - central bank's inflation target
    NATURAL_UNEMPLOYMENT = 5.0      # % - NAIRU
    LABOR_SHARE = 0.55              # share of quarterly output subject to income tax
    PROFIT_SHARE = 0.15             # share of quarterly output subject to corp tax
    GOV_SPENDING_SHARE = 0.15       # baseline non-EI government spending, % of output
    RATE_LAG_QUARTERS = 2           # monetary policy transmission lag

    # Baselines fiscal levers are compared against, to compute stimulus/drag
    BASE_INCOME_TAX = 20.0
    BASE_CORP_TAX = 20.0
    BASE_EI_BENEFIT = 3.0

    def __init__(self):
        # Policy levers (player controlled) - sensible starting stance
        self.interest_rate = 4.0        # nominal rate; real = 4.0 - 2.0 = neutral
        self.income_tax_rate = self.BASE_INCOME_TAX
        self.corp_tax_rate = self.BASE_CORP_TAX
        self.ei_benefit = self.BASE_EI_BENEFIT   # $B / quarter per 1pt of unemployment

        # State
        self.quarter = 0
        self.output_gap = 0.0
        self.inflation = self.TARGET_INFLATION
        self.unemployment = self.NATURAL_UNEMPLOYMENT
        self.potential_gdp = 2100.0     # $B, annualized (~ Canada-scale)
        self.gdp = 2100.0
        self.debt = 1200.0              # $B, federal debt level
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

        # --- Fiscal impulse from tax/benefit levers (deviation from baseline) ---
        fiscal_impulse = (
            0.06 * (self.BASE_INCOME_TAX - self.income_tax_rate)   # tax cut -> stimulus
            + 0.15 * (self.ei_benefit - self.BASE_EI_BENEFIT)      # richer EI -> stimulus
            + 0.02 * (self.BASE_CORP_TAX - self.corp_tax_rate)     # corp tax cut -> mild stimulus
        )

        demand_shock = random.gauss(0, 0.08)

        # --- IS curve: output gap ---
        self.output_gap = (
            0.75 * self.output_gap
            - 0.55 * (real_rate_effective - self.NEUTRAL_REAL_RATE)
            + fiscal_impulse
            + demand_shock
        )
        self.output_gap = max(-10.0, min(10.0, self.output_gap))

        # --- Phillips curve: inflation ---
        supply_shock = random.gauss(0, 0.05)
        self.inflation = (
            0.65 * self.inflation
            + 0.35 * self.TARGET_INFLATION
            + 0.30 * self.output_gap
            + supply_shock
        )
        self.inflation = max(-2.0, min(12.0, self.inflation))

        # --- Okun's law: unemployment ---
        self.unemployment = self.NATURAL_UNEMPLOYMENT - 0.5 * self.output_gap
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
st.title("🏦 Sim-Fed: Canadian Macroeconomic Policy Simulator")
st.markdown(
    "Take control of the Bank of Canada and Finance Department. Adjust policy levers and observe macro outcomes."
)

# Initialize Session State for Model Persistence
if "model" not in st.session_state:
    st.session_state.model = EconomyModel()

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

ei_benefit = st.sidebar.slider(
    "EI Benefit ($B / quarter per 1pt unemployment)",
    min_value=1.0,
    max_value=10.0,
    value=float(model.ei_benefit),
    step=0.5,
)

# Apply Control Settings to Active Model
model.interest_rate = interest_rate
model.income_tax_rate = income_tax
model.corp_tax_rate = corp_tax
model.ei_benefit = ei_benefit

st.sidebar.caption(
    "1 tick = 1 quarter, matching real Bank of Canada meeting cadence. "
    "Rate changes take ~2 quarters to fully hit the economy, like real policy lags."
)

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
if col_btn1.button("▶ 1 Quarter"):
    model.step()

if col_btn2.button("▶▶ 1 Year"):
    for _ in range(4):
        model.step()

if col_btn3.button("🔄 Reset"):
    st.session_state.model = EconomyModel()
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
        n_quarters = len(df)
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
    st.info("Click **Advance 1 Year** in the sidebar to begin.")