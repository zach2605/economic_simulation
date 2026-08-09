import random
import mesa
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Canada Macro Policy Simulator", layout="wide"
)

# Set random seed for reproducible runs
random.seed(42)


# --- AGENT DEFINITIONS ---
class Household(mesa.Agent):

    def __init__(self, model, labor_class):
        super().__init__(model)
        self.labor_class = labor_class
        self.wage = model.wage_rates[labor_class]
        self.employer = None
        self.savings = self.wage * 2.0

    def step(self):
        # Wages track the model's current (tightness-adjusted) wage rates
        self.wage = self.model.wage_rates[self.labor_class]

        # Voluntary turnover (1% per tick)
        if self.employer is not None and random.random() < 0.01:
            self.employer.employees.remove(self)
            self.employer = None

        # Income calculation
        if self.employer is None:
            income = self.model.ei_benefit
            self.model.treasury_balance -= self.model.ei_benefit
            net_income = income
        else:
            income = self.wage
            tax = income * self.model.income_tax_rate
            net_income = income - tax
            self.model.treasury_balance += tax

        # Interest rate sensitivity on spending (Baseline MPC = 82%)
        # Widened multiplier so the full rate slider range produces a
        # meaningful swing in spending behavior (previously too weak
        # relative to the employment feedback loop).
        rate_impact = (self.model.interest_rate - 0.035) * 5.0
        effective_mpc = max(0.50, min(0.95, 0.82 - rate_impact))

        # Spending & Savings Math
        income_spent = net_income * effective_mpc
        savings_spent = self.savings * 0.02
        consumption = income_spent + savings_spent

        # Correctly update savings pool (Income saved minus savings drawn)
        income_saved = net_income * (1.0 - effective_mpc)
        self.savings = (self.savings - savings_spent) + income_saved

        self.model.period_gdp += consumption

        firms = self.model.firms
        if firms and consumption > 0:
            # Weight customer spending by firm capacity
            firm_weights = [f.capacity for f in firms]
            chosen_firm = random.choices(firms, weights=firm_weights, k=1)[0]
            chosen_firm.revenue += consumption


class Firm(mesa.Agent):

    def __init__(self, model, firm_size_tier):
        super().__init__(model)
        self.firm_size_tier = firm_size_tier

        # Calibrated capacities: Total equilibrium sum ~ 950 jobs (5% NAIRU)
        tier_config = {
            "small": {"init_cap": 7, "min_cap": 2, "hr": 1},    # throttled hr
            "medium": {"init_cap": 22, "min_cap": 6, "hr": 2},  # throttled hr
            "large": {"init_cap": 75, "min_cap": 18, "hr": 5},  # throttled hr
        }

        config = tier_config[firm_size_tier]
        self.capacity = config["init_cap"]
        self.min_cap = config["min_cap"]
        self.hr_limit = config["hr"]

        self.employees = []
        self.revenue = 0.0

        # Pre-warm sales memory aligned with aggregate household spending ($24.50)
        init_rev_per_worker = 24.50
        self.smoothed_sales = self.capacity * init_rev_per_worker

    def step(self):
        # Smoothed sales memory
        if self.smoothed_sales == 0:
            self.smoothed_sales = self.revenue
        else:
            self.smoothed_sales = (
                0.7 * self.smoothed_sales + 0.3 * self.revenue
            )

        # Required revenue per worker: cost of capital component (interest
        # rate), a wage component so rising labor costs from a tight market
        # feed into hiring decisions, and a corp-tax component so higher
        # corporate tax makes firms need more pre-tax revenue to justify
        # each additional worker (lower tax -> easier to justify hiring).
        capital_cost_burden = 1.0 + ((self.model.interest_rate - 0.035) * 4.0)
        wage_burden = self.model.wage_index  # 1.0 at baseline tightness
        tax_burden = 1.0 + ((self.model.corp_tax_rate - 0.20) * 1.2)
        required_rev_per_worker = (
            24.50 * capital_cost_burden * wage_burden * tax_burden
        )

        target_cap = max(
            self.min_cap, int(self.smoothed_sales / required_rev_per_worker)
        )

        # Capacity adjustments (Rate-limited by hr_limit, now throttled)
        if target_cap > self.capacity:
            self.capacity += min(self.hr_limit, target_cap - self.capacity)
        elif target_cap < self.capacity:
            self.capacity = max(self.min_cap, self.capacity - self.hr_limit, target_cap)
            if len(self.employees) > self.capacity:
                excess = len(self.employees) - self.capacity
                layoffs = random.sample(self.employees, min(excess, self.hr_limit))
                for w in layoffs:
                    w.employer = None
                    self.employees.remove(w)

        # Hiring from unemployed pool (throttled by hr_limit, same as before,
        # but hr_limit values are now smaller so the market can't saturate
        # in a single tick)
        unemployed = [
            a
            for a in self.model.agents
            if isinstance(a, Household) and a.employer is None
        ]
        vacancies = self.capacity - len(self.employees)

        if vacancies > 0 and unemployed:
            hires = random.sample(
                unemployed, min(len(unemployed), vacancies, self.hr_limit)
            )
            for w in hires:
                w.employer = self
                self.employees.append(w)

        # Operations & Corporate Tax
        payroll = sum(e.wage for e in self.employees)
        profit = self.revenue - payroll

        if profit > 0:
            corp_tax = profit * self.model.corp_tax_rate
            self.model.treasury_balance += corp_tax

        self.revenue = 0.0


class EconomyModel(mesa.Model):

    def __init__(self, num_households=1000):
        super().__init__()
        self.num_households = num_households
        self.treasury_balance = 35000.0
        self.period_gdp = 24500.0
        self.prev_gdp = 24500.0

        # Interactive Policy Levers
        self.interest_rate = 0.035
        self.income_tax_rate = 0.20
        self.corp_tax_rate = 0.20
        self.ei_benefit = 15.00

        # Base wages (fixed reference point); actual wage_rates paid out
        # flex around this based on labor market tightness each tick.
        self.base_wage_rates = {"lower": 18.00, "middle": 38.00, "upper": 88.00}
        self.wage_rates = dict(self.base_wage_rates)
        self.wage_index = 1.0  # multiplier firms see on required revenue

        # Setup Firms
        firm_distribution = (
            ["small"] * 28 + ["medium"] * 14 + ["large"] * 6
        )
        self.firms = []
        for size in firm_distribution:
            f = Firm(self, size)
            self.agents.add(f)
            self.firms.append(f)

        # Setup Households
        class_distribution = (
            ["lower"] * 450 + ["middle"] * 450 + ["upper"] * 100
        )
        households = []
        for labor_class in class_distribution:
            h = Household(self, labor_class)
            self.agents.add(h)
            households.append(h)

        # Employ ~950 households initially (5% baseline NAIRU)
        employed_targets = random.sample(households, 950)
        for h in employed_targets:
            avail = [f for f in self.firms if len(f.employees) < f.capacity]
            if avail:
                f = random.choice(avail)
                h.employer = f
                f.employees.append(h)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Unemployment (%)": lambda m: round(
                    m.calculate_unemployment() * 100, 2
                ),
                "Interest Rate (%)": lambda m: round(m.interest_rate * 100, 2),
                "Inflation Rate (%)": lambda m: round(
                    m.calculate_inflation() * 100, 2
                ),
                "GDP Growth (%)": lambda m: round(
                    m.calculate_gdp_growth() * 100, 2
                ),
                "GDP ($)": lambda m: round(m.period_gdp, 2),
                "Treasury ($)": lambda m: round(m.treasury_balance, 2),
                "Wage Index": lambda m: round(m.wage_index, 3),
            }
        )

        # Collect baseline state snapshot at Tick 0
        self.datacollector.collect(self)

    def calculate_unemployment(self):
        households = [a for a in self.agents if isinstance(a, Household)]
        unemployed = sum(1 for h in households if h.employer is None)
        return unemployed / len(households)

    def calculate_gdp_growth(self):
        if self.prev_gdp <= 0:
            return 0.0
        return ((self.period_gdp - self.prev_gdp) / self.prev_gdp)

    def calculate_inflation(self):
        u = self.calculate_unemployment()
        tightness = 0.05 - u  # Phillips Curve pressure relative to 5% NAIRU
        gdp_growth = self.calculate_gdp_growth()
        inflation = 0.02 + (tightness * 0.7) + (gdp_growth * 0.2)
        return max(-0.01, min(0.09, inflation))

    def update_wages(self):
        """Wage stabilizer: tight labor market pushes wages (and required
        firm revenue-per-worker) up, which is the missing brake that
        previously let the hiring loop run away unchecked. Loose labor
        market lets wages/wage_index drift back down."""
        u = self.calculate_unemployment()
        tightness = 0.05 - u  # positive = tight market, negative = slack
        target_index = 1.0 + max(-0.35, min(0.35, tightness * 3.0))
        # Smooth adjustment so wages don't jump discontinuously tick to tick
        self.wage_index = 0.9 * self.wage_index + 0.1 * target_index
        for k in self.wage_rates:
            self.wage_rates[k] = self.base_wage_rates[k] * self.wage_index

    def step(self):
        self.prev_gdp = self.period_gdp if self.period_gdp > 0 else 24500.0
        self.period_gdp = 0.0

        # Update wages BEFORE households/firms act this tick, based on
        # last tick's unemployment reading.
        self.update_wages()

        # Fiscal spending capped at a sustainable 2% of treasury
        gov_spending = max(0, self.treasury_balance * 0.02)
        if gov_spending > 0 and self.firms:
            self.treasury_balance -= gov_spending
            per_firm = gov_spending / len(self.firms)
            for f in self.firms:
                f.revenue += per_firm
            self.period_gdp += gov_spending

        # Two-phase deterministic stepping: ALL households spend/earn
        # before ANY firm reacts to that revenue. Previously shuffle_do
        # interleaved households and firms randomly, so some firms saw
        # partial/incomplete revenue for the tick, injecting noise
        # unrelated to policy levers.
        self.agents_by_type[Household].shuffle_do("step")
        self.agents_by_type[Firm].shuffle_do("step")

        self.datacollector.collect(self)


# --- STREAMLIT DASHBOARD UI ---
st.title("🏦 Sim-Fed: Canadian Macroeconomic Policy Simulator")
st.markdown(
    "Take control of the Bank of Canada and Finance Department. Adjust policy levers and observe macro outcomes."
)

# Initialize Session State for Model Persistence
if "model" not in st.session_state:
    st.session_state.model = EconomyModel()
    st.session_state.tick_count = 0

model = st.session_state.model

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Policy Levers")

interest_rate = (
    st.sidebar.slider(
        "Central Bank Policy Rate (%)",
        min_value=0.25,
        max_value=10.00,
        value=float(model.interest_rate * 100),
        step=0.25,
    )
    / 100.0
)

income_tax = (
    st.sidebar.slider(
        "Income Tax Rate (%)",
        min_value=5.0,
        max_value=45.0,
        value=float(model.income_tax_rate * 100),
        step=1.0,
    )
    / 100.0
)

corp_tax = (
    st.sidebar.slider(
        "Corporate Tax Rate (%)",
        min_value=5.0,
        max_value=40.0,
        value=float(model.corp_tax_rate * 100),
        step=1.0,
    )
    / 100.0
)

ei_benefit = st.sidebar.slider(
    "EI Benefit ($ / tick)",
    min_value=5.0,
    max_value=40.0,
    value=float(model.ei_benefit),
    step=1.0,
)

# Apply Control Settings to Active Model
model.interest_rate = interest_rate
model.income_tax_rate = income_tax
model.corp_tax_rate = corp_tax
model.ei_benefit = ei_benefit

st.sidebar.caption("1 tick = 1 quarter · each step advances 1 year (4 ticks)")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶ Advance 1 Year"):
    for _ in range(4):
        model.step()
        st.session_state.tick_count += 1

if col_btn2.button("🔄 Reset"):
    st.session_state.model = EconomyModel()
    st.session_state.tick_count = 0
    st.rerun()

# --- MAIN DISPLAY METRICS ---
df = model.datacollector.get_model_vars_dataframe()

if not df.empty:
    latest = df.iloc[-1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Unemployment", f"{latest['Unemployment (%)']}%")
    m2.metric("GDP Growth", f"{latest['GDP Growth (%)']}%")
    m3.metric("Inflation", f"{latest['Inflation Rate (%)']}%")
    m4.metric("GDP Level", f"${latest['GDP ($)']:,.2f}")
    m5.metric("Treasury", f"${latest['Treasury ($)']:,.2f}")

    st.divider()

    # --- CHARTS ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Labor Market & Interest Rates")
        fig1 = px.line(
            df,
            y=["Unemployment (%)", "Interest Rate (%)"],
            title="Unemployment vs Policy Rate",
            labels={"index": "Quarter", "value": "Percent (%)", "variable": "Metric"},
        )
        fig1.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        st.subheader("Economic Output & Inflation")
        fig2 = px.line(
            df,
            y=["GDP Growth (%)", "Inflation Rate (%)"],
            title="GDP Growth Rate vs Inflation",
            labels={"index": "Quarter", "value": "Percent (%)", "variable": "Metric"},
        )
        fig2.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Simulation Ledger")
    st.dataframe(df.tail(15), width="stretch")
else:
    st.info("Click **Step 1 Tick** or **Run 10 Ticks** in the sidebar to begin.")