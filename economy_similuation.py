import random
import mesa
import pandas as pd

# Set seed for reproducible friction dynamics
random.seed(42)


class Household(mesa.Agent):
    """Represents a Canadian worker/household."""

    def __init__(self, model, labor_class):
        super().__init__(model)
        self.labor_class = labor_class  # 'lower', 'middle', 'upper'
        self.wage = model.wage_rates[labor_class]
        self.employer = None
        self.savings = 100.0  # Initial liquidity buffer

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

        # 3. Consumption Spending (MPC = 0.85)
        consumption = self.savings * 0.85
        self.savings -= consumption
        self.model.period_gdp += consumption


class Firm(mesa.Agent):
    """Represents a Canadian employer firm."""

    def __init__(self, model, firm_size):
        super().__init__(model)
        self.firm_size = firm_size  # 'small', 'medium', 'large'
        self.capacity = model.capacity_limits[firm_size]
        self.employees = []
        self.revenue = 0.0

    def step(self):
        # 1. Labor Market Matching with Hiring Friction
        unemployed = [
            agent
            for agent in self.model.agents
            if isinstance(agent, Household) and agent.employer is None
        ]
        vacancies = self.capacity - len(self.employees)

        if vacancies > 0 and unemployed:
            # Hiring friction: Max 2 hires per firm per tick (interview pipeline)
            max_hires = min(vacancies, 2)
            hires = random.sample(unemployed, min(len(unemployed), max_hires))
            for worker in hires:
                worker.employer = self
                self.employees.append(worker)

        # 2. Production & Revenue (Value-Add Multiplier)
        total_payroll = sum(e.wage for e in self.employees)
        self.revenue = total_payroll * 1.35

        # 3. Corporate Taxation (26.5%)
        profit = self.revenue - total_payroll
        if profit > 0:
            corp_tax = profit * self.model.corp_tax_rate
            self.model.treasury_balance += corp_tax


class CanadaEconomyModel(mesa.Model):
    """Macroeconomic Agent-Based Model with Search & Friction Dynamics."""

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
        self.capacity_limits = {"small": 10, "medium": 30, "large": 100}

        # Initialize Households
        class_distribution = (
            ["lower"] * 450 + ["middle"] * 450 + ["upper"] * 100
        )
        for labor_class in class_distribution:
            agent = Household(self, labor_class)
            self.agents.add(agent)

        # Initialize Firms
        firm_distribution = (
            ["small"] * 28 + ["medium"] * 14 + ["large"] * 6
        )
        for firm_size in firm_distribution:
            agent = Firm(self, firm_size)
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