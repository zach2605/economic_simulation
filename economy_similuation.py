import random
import mesa
import pandas as pd

# Set seed for reproducible dynamics
random.seed(42)


class Household(mesa.Agent):
    """Represents a Canadian worker/household."""

    def __init__(self, model, labor_class):
        super().__init__(model)
        self.labor_class = labor_class  # 'lower', 'middle', 'upper'
        self.wage = model.wage_rates[labor_class]
        self.employer = None
        # Smooth initial savings to match steady-state buffer (prevents T0 spike)
        self.savings = self.wage * 1.25

    def step(self):
        # 1. Voluntary Turnover: 2% chance per tick to resign/switch jobs
        if self.employer is not None and random.random() < 0.02:
            self.employer.employees.remove(self)
            self.employer = None

        # 2. Earn income or collect Employment Insurance (EI)
        if self.employer is None:
            self.savings += self.model.ei_benefit
            self.model.treasury_balance -= self.model.ei_benefit
        else:
            gross_wage = self.wage
            income_tax = gross_wage * self.model.income_tax_rate
            net_wage = gross_wage - income_tax

            self.savings += net_wage
            self.model.treasury_balance += income_tax

        # 3. Demand-Driven Consumption Spending (MPC = 0.85)
        consumption = self.savings * 0.85
        self.savings -= consumption
        self.model.period_gdp += consumption

        # Route household spending into firms to drive demand
        firms = [a for a in self.model.agents if isinstance(a, Firm)]
        if firms and consumption > 0:
            chosen_firm = random.choice(firms)
            chosen_firm.revenue += consumption


class Firm(mesa.Agent):
    """Represents a Canadian employer firm with dynamic capacity expansion/layoffs."""

    def __init__(self, model, firm_size_tier):
        super().__init__(model)
        self.firm_size_tier = firm_size_tier  # 'small', 'medium', 'large'

        # Tier-based initial capacity, max scaling rate, and floor limit
        tier_config = {
            "small": {
                "init_cap": 12,
                "max_scale_rate": 3,
                "min_cap": 5,
                "hr_limit": 3,
            },
            "medium": {
                "init_cap": 35,
                "max_scale_rate": 6,
                "min_cap": 12,
                "hr_limit": 6,
            },
            "large": {
                "init_cap": 110,
                "max_scale_rate": 18,
                "min_cap": 30,
                "hr_limit": 18,
            },
        }

        config = tier_config[firm_size_tier]
        self.capacity = config["init_cap"]
        self.max_scale_rate = config["max_scale_rate"]
        self.min_cap = config["min_cap"]
        self.hr_limit = config["hr_limit"]

        self.employees = []
        self.sales_history = []
        self.revenue = 0.0

    def step(self):
        # 1. Evaluate Demand & Adjust Capacity (Expand or Lay Off)
        if len(self.sales_history) >= 2:
            avg_recent_sales = sum(self.sales_history[-2:]) / 2.0

            # Divisor set to 27.0 to target ~950 jobs under total economic demand
            target_capacity = max(self.min_cap, int(avg_recent_sales / 27.0))

            if target_capacity > self.capacity:
                # Demand-driven expansion
                growth = min(
                    self.max_scale_rate, target_capacity - self.capacity
                )
                self.capacity += growth
            elif target_capacity < self.capacity:
                # Demand-driven contraction
                self.capacity = max(self.min_cap, target_capacity)

                # Execute layoffs if overstaffed relative to contracted capacity
                if len(self.employees) > self.capacity:
                    excess_count = len(self.employees) - self.capacity
                    layoffs = random.sample(self.employees, excess_count)
                    for worker in layoffs:
                        worker.employer = None
                        self.employees.remove(worker)

        # 2. Labor Market Matching (Hiring with HR Bandwidth Friction)
        unemployed = [
            agent
            for agent in self.model.agents
            if isinstance(agent, Household) and agent.employer is None
        ]
        vacancies = self.capacity - len(self.employees)

        if vacancies > 0 and unemployed:
            max_hires = min(vacancies, self.hr_limit)
            hires = random.sample(unemployed, min(len(unemployed), max_hires))
            for worker in hires:
                worker.employer = self
                self.employees.append(worker)

        # 3. Production, Profitability & Corporate Taxation
        total_payroll = sum(e.wage for e in self.employees)
        profit = self.revenue - total_payroll

        if profit > 0:
            corp_tax = profit * self.model.corp_tax_rate
            self.model.treasury_balance += corp_tax

        # Record sales history and clear tick revenue buffer for next round
        self.sales_history.append(self.revenue)
        self.revenue = 0.0


class CanadaEconomyModel(mesa.Model):
    """Macroeconomic Agent-Based Model with Fiscal Balance & ~5% Unemployment Calibration."""

    def __init__(
        self, num_households=1000, num_firms=48, initial_treasury=50000.0
    ):
        super().__init__()
        self.num_households = num_households
        self.num_firms = num_firms
        self.treasury_balance = initial_treasury
        self.period_gdp = 0.0

        # Calibration parameters
        self.income_tax_rate = 0.24
        self.corp_tax_rate = 0.265
        self.ei_benefit = 9.35

        self.wage_rates = {"lower": 17.00, "middle": 37.00, "upper": 85.00}

        # 1. Initialize Tiered Firms (28 Small, 14 Medium, 6 Large)
        firm_distribution = (
            ["small"] * 28 + ["medium"] * 14 + ["large"] * 6
        )
        self.firms = []
        for firm_size in firm_distribution:
            agent = Firm(self, firm_size)
            self.agents.add(agent)
            self.firms.append(agent)

        # 2. Initialize Households & Pre-Assign Employment to Match Capacity (~95% initial fill)
        class_distribution = (
            ["lower"] * 450 + ["middle"] * 450 + ["upper"] * 100
        )
        for labor_class in class_distribution:
            agent = Household(self, labor_class)

            # Assign to an open firm vacancy on setup
            available_firms = [
                f for f in self.firms if len(f.employees) < f.capacity
            ]
            if available_firms:
                chosen = random.choice(available_firms)
                agent.employer = chosen
                chosen.employees.append(agent)

            self.agents.add(agent)

        # Data Collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Unemployment Rate": lambda m: m.calculate_unemployment(),
                "Treasury Balance": lambda m: m.treasury_balance,
                "Gross Domestic Product (GDP)": lambda m: m.period_gdp,
            }
        )

    def calculate_unemployment(self):
        households = [a for a in self.agents if isinstance(a, Household)]
        unemployed = sum(1 for h in households if h.employer is None)
        return unemployed / len(households)

    def step(self):
        self.period_gdp = 0.0

        # Fiscal Recirculation: Government spending recycles tax revenue back to firms
        gov_spending = self.treasury_balance * 0.12
        if gov_spending > 0 and self.firms:
            self.treasury_balance -= gov_spending
            spend_per_firm = gov_spending / len(self.firms)
            for firm in self.firms:
                firm.revenue += spend_per_firm
            self.period_gdp += gov_spending

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


# --- Run 35-Tick Simulation ---
if __name__ == "__main__":
    model = CanadaEconomyModel()
    for tick in range(35):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    print("Simulation Finished! Dynamic historical metrics:")
    print(df.to_string())