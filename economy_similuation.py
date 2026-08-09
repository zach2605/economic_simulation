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

        self.savings += net_income

        # Interest rate sensitivity on spending
        # Baseline MPC = 78%. Higher interest rates lower MPC (encourage saving)
        rate_impact = (self.model.interest_rate - 0.035) * 3.5
        effective_mpc = max(0.50, min(0.90, 0.78 - rate_impact))

        consumption = (net_income * effective_mpc) + (self.savings * 0.02)
        self.savings -= consumption
        self.model.period_gdp += consumption

        firms = [a for a in self.model.agents if isinstance(a, Firm)]
        if firms and consumption > 0:
            chosen_firm = random.choice(firms)
            chosen_firm.revenue += consumption


class Firm(mesa.Agent):

    def __init__(self, model, firm_size_tier):
        super().__init__(model)
        self.firm_size_tier = firm_size_tier

        # Calibrated capacities: Total equilibrium sum ~ 950 jobs (5% NAIRU)
        tier_config = {
            "small": {"init_cap": 7, "min_cap": 2, "hr": 2},  # 28 * 7 = 196
            "medium": {"init_cap": 22, "min_cap": 6, "hr": 4},  # 14 * 22 = 308
            "large": {"init_cap": 75, "min_cap": 18, "hr": 10},  # 6 * 75 = 450
        }

        config = tier_config[firm_size_tier]
        self.capacity = config["init_cap"]
        self.min_cap = config["min_cap"]
        self.hr_limit = config["hr"]

        self.employees = []
        self.revenue = 0.0
        self.smoothed_sales = 0.0

    def step(self):
        # Smoothed sales memory
        if self.smoothed_sales == 0:
            self.smoothed_sales = self.revenue
        else:
            self.smoothed_sales = (
                0.7 * self.smoothed_sales + 0.3 * self.revenue
            )

        avg_wage = self.model.get_average_wage()

        # Debt Service / CapEx Burden from Policy Interest Rate
        capital_cost_burden = 1.0 + (self.model.interest_rate * 2.2)
        required_rev_per_worker = avg_wage * capital_cost_burden * 1.05

        target_cap = max(
            self.min_cap, int(self.smoothed_sales / required_rev_per_worker)
        )

        # Capacity adjustments
        if target_cap > self.capacity:
            self.capacity += min(self.hr_limit, target_cap - self.capacity)
        elif target_cap < self.capacity:
            self.capacity = max(self.min_cap, target_cap)
            if len(self.employees) > self.capacity:
                excess = len(self.employees) - self.capacity
                layoffs = random.sample(self.employees, excess)
                for w in layoffs:
                    w.employer = None
                    self.employees.remove(w)

        # Hiring from pool
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
        self.period_gdp = 0.0
        self.prev_gdp = 29000.0

        # Interactive Policy Levers
        self.interest_rate = 0.035
        self.income_tax_rate = 0.20
        self.corp_tax_rate = 0.20
        self.ei_benefit = 15.00

        self.wage_rates = {"lower": 18.00, "middle": 38.00, "upper": 88.00}

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

        # Employ ~950 households initially (5% baseline unemployment)
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
            }
        )

    def get_average_wage(self):
        return 34.00

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

    def step(self):
        self.prev_gdp = self.period_gdp if self.period_gdp > 0 else 29000.0
        self.period_gdp = 0.0

        # Fiscal spending capped at a sustainable 2% of treasury
        gov_spending = max(0, self.treasury_balance * 0.02)
        if gov_spending > 0 and self.firms:
            self.treasury_balance -= gov_spending
            per_firm = gov_spending / len(self.firms)
            for f in self.firms:
                f.revenue += per_firm
            self.period_gdp += gov_spending

        self.agents.shuffle_do("step")
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

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
if col_btn1.button("▶ Step 1 Tick"):
    model.step()
    st.session_state.tick_count += 1

if col_btn2.button("⏩ Run 10 Ticks"):
    for _ in range(10):
        model.step()
        st.session_state.tick_count += 1

if col_btn3.button("🔄 Reset"):
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
        )
        st.plotly_chart(fig1, width=True)

    with c2:
        st.subheader("Economic Output & Inflation")
        fig2 = px.line(
            df,
            y=["GDP Growth (%)", "Inflation Rate (%)"],
            title="GDP Growth Rate vs Inflation",
        )
        st.plotly_chart(fig2, width=True)

    st.subheader("Simulation Ledger")
    st.dataframe(df.tail(15), width=True)
else:
    st.info("Click **Step 1 Tick** or **Run 10 Ticks** in the sidebar to begin.")