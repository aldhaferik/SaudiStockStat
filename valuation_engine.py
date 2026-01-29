import pandas as pd
import numpy as np

class ValuationEngine:
    def __init__(self, financials):
        self.bs = financials.get('balance_sheet', pd.DataFrame())
        self.is_ = financials.get('income_statement', pd.DataFrame())
        self.cf = financials.get('cash_flow', pd.DataFrame())

    def get_latest_value(self, df, row_name):
        """Safely retrieves the most recent value for a given financial row."""
        try:
            if row_name in df.index:
                return df.loc[row_name].iloc[0]
            for idx in df.index:
                if row_name.lower() in str(idx).lower():
                    return df.loc[idx].iloc[0]
            return 0
        except:
            return 0

    def dcf_valuation(self, growth_rate=0.03, discount_rate=0.10, terminal_growth=0.02, projection_years=5):
        try:
            free_cash_flow = self.get_latest_value(self.cf, "Free Cash Flow")
            if free_cash_flow == 0:
                ocf = self.get_latest_value(self.cf, "Operating Cash Flow")
                capex = abs(self.get_latest_value(self.cf, "Capital Expenditure"))
                free_cash_flow = ocf - capex
            
            if free_cash_flow <= 0: return 0

            shares = self.get_latest_value(self.bs, "Share Issued")
            if shares == 0: shares = self.get_latest_value(self.bs, "Ordinary Shares Number")
            if shares == 0: return 0

            future_cash_flows = []
            for i in range(1, projection_years + 1):
                fcf = free_cash_flow * ((1 + growth_rate) ** i)
                discounted_fcf = fcf / ((1 + discount_rate) ** i)
                future_cash_flows.append(discounted_fcf)

            terminal_value = (free_cash_flow * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            discounted_tv = terminal_value / ((1 + discount_rate) ** projection_years)
            
            total_equity_value = sum(future_cash_flows) + discounted_tv
            net_debt = self.get_latest_value(self.bs, "Net Debt") 
            if net_debt == 0:
                total_debt = self.get_latest_value(self.bs, "Total Debt")
                cash = self.get_latest_value(self.bs, "Cash And Cash Equivalents")
                net_debt = total_debt - cash

            fair_value = (total_equity_value - net_debt) / shares
            return max(0, fair_value)
        except:
            return 0

    def multiples_valuation(self, pe_ratio=15, pb_ratio=2.0, ev_ebitda_ratio=10):
        try:
            eps = self.get_latest_value(self.is_, "Diluted EPS")
            if eps == 0: eps = self.get_latest_value(self.is_, "Basic EPS")
            pe_val = eps * pe_ratio if eps > 0 else 0

            book_val = self.get_latest_value(self.bs, "Total Equity Gross Minority Interest")
            shares = self.get_latest_value(self.bs, "Share Issued")
            bvps = (book_val / shares) if shares > 0 else 0
            pb_val = bvps * pb_ratio if bvps > 0 else 0

            ebitda = self.get_latest_value(self.is_, "EBITDA")
            net_debt = self.get_latest_value(self.bs, "Net Debt")
            if net_debt == 0:
                total_debt = self.get_latest_value(self.bs, "Total Debt")
                cash = self.get_latest_value(self.bs, "Cash And Cash Equivalents")
                net_debt = total_debt - cash

            ev_val = 0
            if ebitda > 0 and shares > 0:
                enterprise_value = ebitda * ev_ebitda_ratio
                equity_value = enterprise_value - net_debt
                ev_val = equity_value / shares

            return {
                "PE_Valuation": max(0, pe_val),
                "PB_Valuation": max(0, pb_val),
                "EBITDA_Valuation": max(0, ev_val)
            }
        except:
            return {"PE_Valuation": 0, "PB_Valuation": 0, "EBITDA_Valuation": 0}
