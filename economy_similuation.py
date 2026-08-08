import mesa
import random
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# =============================================================================
# 1. WORKER / HOUSEHOLD AGENT
# =============================================================================
class HouseholdAgent(mesa.Agent):
    def __init__(self, model, household_class):
        super().__init__(model)
        self.household_class = household_class  # "Lower", "Middle", or "Upper"
        self.employment_status = "Unemployed"   # Everyone starts unemployed
        self.employer = None  
        self.wealth = 100.0   # Starting cash savings
        
        # Marginal Propensity to Consume (MPC) & Baseline Wage
        if household_class == "Lower":
            self.mpc = 0.95        
            self.baseline_wage = 15.0
        elif household_class == "Middle":
            self.mpc = 0.70        
            self.baseline_wage = 35.0
        elif household_class == "Upper":
            self.mpc = 0.40        
            self.baseline_wage = 100.0

    def step(self):
        pass

# =============================================================================
# 2. BUSINESS / FIRM AGENT
# =============================================================================
class BusinessAgent(mesa.Agent):
    def __init__(self, model, size):
        super().__init__(model)
        self.size = size  
        self.employees = []
        self.bankrupt = False
        
        # Calibrate baseline characteristics based on firm scale
        if size == "Small":
            self.capital = 1000.0
            self.hire_target = 3
        elif size == "Medium":
            self.capital = 5000.0
            self.hire_target = 8
        elif size == "Large":
            self.capital = 20000.0
            self.hire_target = 20

    def step(self):
        if self.bankrupt:
            return

        # Simple hiring logic: If below targets and has capital, try to hire unemployed agents
        unemployed_workers = [w for w in self.model.agents_by_type[HouseholdAgent] 
                              if w.employment_status == "Unemployed"]
        
        # Introduce hiring friction: A business only has a 75% chance to fill a vacancy per tick
        while len(self.employees) < self.hire_target and unemployed_workers and self.capital > 100:
            if random.random() < 0.75:  # 75% chance to successfully match with a candidate
                candidate = random.choice(unemployed_workers)
                candidate.employment_status = "Employed"
                candidate.employer = self
                self.employees.append(candidate)
                unemployed_workers.remove(candidate)
            else:
                break  # Stop trying to hire this tick (friction)

# =============================================================================
# 3. ECONOMIC MODEL (THE POLICY ENVIRONMENT)
# =============================================================================
class EconomicModel(mesa.Model):
    def __init__(self, num_households, num_firms, income_tax_rate, corporate_tax_rate, upskilling_subsidy):
        super().__init__()
        
        self.income_tax_rate = income_tax_rate          
        self.corporate_tax_rate = corporate_tax_rate    
        self.upskilling_subsidy = upskilling_subsidy    
        
        self.gov_treasury = 5000.0                      
        self.consumption_pool = 0.0                     
        
        # Spawn Stratified Households (Initially Unemployed)
        classes = ["Lower"] * int(num_households * 0.6) + \
                  ["Middle"] * int(num_households * 0.3) + \
                  ["Upper"] * int(num_households * 0.1)
                  
        for h_class in classes:
            HouseholdAgent(model=self, household_class=h_class)
            
        # Spawn Stratified Businesses
        sizes = ["Small"] * int(num_firms * 0.6) + \
                ["Medium"] * int(num_firms * 0.3) + \
                ["Large"] * int(num_firms * 0.1)
                
        for b_size in sizes:
            BusinessAgent(model=self, size=b_size)
            
        # Initialize DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Unemployment Rate": lambda m: m.calculate_unemployment(),
                "Treasury Balance": "gov_treasury",
                "Gross Domestic Product (GDP)": "consumption_pool"
            }
        )

    def calculate_unemployment(self):
        households = self.agents_by_type[HouseholdAgent]
        unemployed = [h for h in households if h.employment_status == "Unemployed"]
        return len(unemployed) / len(households)

    def run_economic_cycle(self):
        # 1. PAYROLL & INCOME TAXES
        for biz in self.agents_by_type[BusinessAgent]:
            if biz.bankrupt:
                continue
            for emp in biz.employees:
                gross_wage = emp.baseline_wage
                tax_due = gross_wage * self.income_tax_rate
                self.gov_treasury += tax_due
                net_wage = gross_wage - tax_due
                emp.wealth += net_wage
                biz.capital -= gross_wage  

        # 2. HOUSEHOLD CONSUMPTION
        self.consumption_pool = 0.0
        for household in self.agents_by_type[HouseholdAgent]:
            if household.employment_status == "Employed":
                income_this_tick = household.baseline_wage * (1 - self.income_tax_rate)
                spent_amount = income_this_tick * household.mpc
            else:
                spent_amount = min(household.wealth, 5.0) * household.mpc
            
            household.wealth -= spent_amount
            self.consumption_pool += spent_amount  

        # 3. BUSINESS REVENUE & CORPORATE TAXATION
        active_firms = [b for b in self.agents_by_type[BusinessAgent] if not b.bankrupt]
        if active_firms:
            total_shares = sum(1 if b.size == "Small" else 3 if b.size == "Medium" else 10 for b in active_firms)
            
            for biz in active_firms:
                share_multiplier = 1 if biz.size == "Small" else 3 if biz.size == "Medium" else 10
                revenue_earned = self.consumption_pool * (share_multiplier / total_shares)
                biz.capital += revenue_earned
                
                if revenue_earned > 0:
                    corp_tax = revenue_earned * self.corporate_tax_rate
                    biz.capital -= corp_tax
                    self.gov_treasury += corp_tax

        # 4. GOVERNMENT WELFARE PAYOUTS (SUBSIDIES)
        for household in self.agents_by_type[HouseholdAgent]:
            if household.employment_status == "Unemployed":
                self.gov_treasury -= self.upskilling_subsidy
                household.wealth += self.upskilling_subsidy

        # 5. BUSINESS SOLVENCY & RESTRUCTURING
        for biz in self.agents_by_type[BusinessAgent]:
            if biz.bankrupt:
                continue
            if biz.capital < 0:
                biz.bankrupt = True
                print(f"[Bankruptcy] A {biz.size} business went bankrupt. Releasing employees.")
                for emp in biz.employees:
                    emp.employment_status = "Unemployed"
                    emp.employer = None
                biz.employees = []

        # 6. HIRING PROCESS (Runs for the NEXT cycle's payroll)
        self.agents_by_type[BusinessAgent].shuffle_do("step")

    def step(self):
        # 1. Collect data BEFORE running the new cycle. This captures the true end-of-tick state
        # including bankruptcies/unemployment from the previous round before hiring sweeps them up.
        self.datacollector.collect(self)
        
        # 2. Run the cycle
        self.run_economic_cycle()

# =============================================================================
# 4. RUN SYSTEM WITH REALISTIC PROPORTIONS (140 Households vs 20 Firms)
# =============================================================================
if __name__ == "__main__":
    print("--- Scenario: Balanced System (140 Households, 20 Firms) ---")
    model = EconomicModel(
        num_households=140, 
        num_firms=20, 
        income_tax_rate=0.40, 
        corporate_tax_rate=0.20, 
        upskilling_subsidy=30.0
    )
    
    for _ in range(20):
        model.step()
        
    history_df = model.datacollector.get_model_vars_dataframe()
    print("\nSimulation Finished! Dynamic historical metrics:")
    print(history_df.to_string())