# portfolio_optimizer.py
from itertools import product
import json
import random 

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from dimod import Integer, Binary
from dimod import quicksum 
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler 

class PortfolioOptimizer:
    def __init__(self, stocks, budget, bin_size=None, gamma=None, 
                 file_path='data/stock_data.csv', dates=None, 
                 model_type='CQM', alpha=0.005, 
                 baseline='^GSPC', sampler_args=None, 
                 t_cost=0.01, verbose=True,max_risk=0, min_return=0,num=0):
        self.stocks = stocks
        self.budget = budget
        self.bin_size = bin_size
        self.gamma = gamma
        self.file_path = file_path
        self.dates = dates
        self.model_type = model_type
        self.alpha = alpha
        self.init_budget = budget 
        self.baseline = baseline
        self.sampler_args = sampler_args
        self.t_cost = t_cost
        self.verbose = verbose
        self.max_risk=max_risk
        self.min_return=min_return
        self.num=num
        
        # Load basic data
        self.data = self.load_data()


        if isinstance(alpha, (list, tuple)):
            self.alpha = alpha[0]
            self.alpha_list = list(alpha)
        elif isinstance (alpha, (int, float)):
            self.alpha = alpha
        else:
            raise TypeError("Use integer or float for alpha, or a list or tuple of either type.")
        
        if gamma:
            if isinstance(gamma, (list, tuple)):
                self.gamma = gamma[-1]
                self.gamma_list = list(gamma)
            elif isinstance(gamma, (int, float)):
                self.gamma = gamma 
            else:
                raise TypeError("Use integer or float for gamma, or a list or tuple of either type.")
        else: 
            self.gamma = 100

        if bin_size:
            self.bin_size = bin_size
        else:
            self.bin_size = 10
       
        self.model = {'CQM': None, 'DQM': None}

        self.sample_set = {}
        if sampler_args:
            self.sampler_args = json.loads(sampler_args) 
        else:
            self.sampler_args = {}

        self.sampler = {'CQM': LeapHybridCQMSampler(**self.sampler_args,token='DEV-081b757ac8fa9601e4fd99a1710375235deaa0c4'),
                        'DQM': LeapHybridDQMSampler(**self.sampler_args,token='DEV-081b757ac8fa9601e4fd99a1710375235deaa0c4')}

        self.solution = {}

        self.precision = 2
       
        

    def load_data(self, file_path='', dates=None, df=None, num=0):
            """Load the relevant stock data from file, dataframe, or Yahoo!. 

            Args:
                file_path (string): Full path of csv file containing stock price data
                    for the single period problem.
                dates (list): [Start_Date, End_Date] to query data from Yahoo!.
                df (dataframe): Table of stock prices.   
                num (int): Number of stocks to be randomnly generated. 
            """
            if df is not None:
                print("\nLoading data from DataFrame...")
                self.df = df 
                self.stocks = df.columns.tolist()
            # elif dates or self.dates: 
            #     if dates:
            #         self.dates = dates 

            #     print(f"\nLoading live data from the web from Yahoo! finance",
            #           f"from {self.dates[0]} to {self.dates[1]}...")

            #     # Generating randomn list of stocks 
            #     if num > 0: 
            #         if (self.dates[0] < '2010-01-01'):
            #             raise Exception(f"Start date must be >= '2010-01-01' " 
            #                             f"when using option 'num'.") 
            #         symbols_df = pd.read_csv('data/stocks_symbols.csv')
            #         self.stocks = random.sample(list(symbols_df.loc[:,'Symbol']), num)

            #     # Read in daily data; resample to monthly
            #     panel_data = yf.download(self.stocks, 
            #                              start=self.dates[0], end=self.dates[1])
            #     panel_data = panel_data.resample('BM').last()
            #     self.df_all = pd.DataFrame(index=panel_data.index, 
            #                                columns=self.stocks)

            #     for i in self.stocks:
            #         self.df_all[i] = panel_data[[('Adj Close',  i)]]

            #     nan_columns = self.df_all.columns[self.df_all.isna().any()].tolist()
            #     if nan_columns:
            #         print("The following tickers are dropped due to invalid data: ", nan_columns)
            #         self.df_all = self.df_all.dropna(axis=1)
            #         if len(self.df_all.columns) < 2:
            #             raise Exception(f"There must be at least 2 valid stock tickers.") 
            #         self.stocks = list(self.df_all.columns)

            #     # Read in baseline data; resample to monthly
            #     index_df = yf.download(self.baseline, 
            #                            start=self.dates[0], end=self.dates[1])
            #     index_df = index_df.resample('BM').last()
            #     self.df_baseline = pd.DataFrame(index=index_df.index)
                
            #     for i in self.baseline:
            #         self.df_baseline[i] = index_df[[('Adj Close')]]

            #     self.df = self.df_all 
            else:
                print("\nLoading data from provided CSV file...")
                if file_path:
                    self.file_path = file_path

                self.df = pd.read_csv(self.file_path, index_col=0)

            self.init_holdings = {s:0 for s in self.stocks}
            print("self.init_holdings",self.init_holdings)
            self.max_num_shares = (self.budget/self.df.iloc[-1]).astype(int)
            print("self.max_num_shares",self.max_num_shares)
            print("self.stocks",self.stocks)
            print("self.bin_size",self.bin_size)
            if self.verbose:
                print("\nMax shares we can afford with a budget of", self.budget)
                print(self.max_num_shares.to_string())

            self.shares_intervals = {}
            for stock in self.stocks:
                if self.max_num_shares[stock]+1 <= self.bin_size:
                    self.shares_intervals[stock] = list(range(self.max_num_shares[stock] + 1))
                else:
                    span = (self.max_num_shares[stock]+1) / self.bin_size
                    self.shares_intervals[stock] = [int(i*span) 
                                            for i in range(self.bin_size)]
            print("self.shares_intervals",self.shares_intervals)
            self.price = self.df.iloc[-1]
            self.monthly_returns = self.df[list(self.stocks)].pct_change().iloc[1:]
            self.avg_monthly_returns = self.monthly_returns.mean(axis=0)
            self.covariance_matrix = covariance_matrix = self.monthly_returns.cov()
            print("self.price",self.price)
            print("self.monthly_returns",self.monthly_returns)
            print(" self.avg_monthly_returns", self.avg_monthly_returns)
            print("self.covariance_matrix",self.covariance_matrix)

            # convert any NaNs in the covariance matrix to 0s
            covariance_matrix.replace(np.nan, 0)

    def build_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Build and store a CQM. 
        This method allows the user a choice of 3 problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state. 
        """
        # Instantiating the CQM object 
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model 
        x = {s: Integer("%s" %s, lower_bound=0, 
                        upper_bound=self.max_num_shares[s]) for s in self.stocks}
        print("x----",x)
        print("covariance_matrix",self.covariance_matrix)
        # Defining risk expression 
        risk = 0
        for s1, s2 in product(self.stocks, self.stocks):
            coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
            print("coeff--",coeff)
            risk = risk + coeff*x[s1]*x[s2]
        print("risk---",risk)
        # Defining the returns expression 
        returns = 0
        for s in self.stocks: 
            returns = returns + self.price[s] * self.avg_monthly_returns[s] * x[s]
        print("returns---",returns)
        # Adding budget and related constraints
        if not init_holdings:
            init_holdings = self.init_holdings
        else:
            self.init_holdings = init_holdings

        if not self.t_cost:  
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.stocks])
                            <= self.budget, label='upper_budget')
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.stocks])
                            >= 0.997*self.budget, label='lower_budget')
        else:
            # Modeling transaction cost 
            x0 = init_holdings

            y = {s: Binary("Y[%s]" %s) for s in self.stocks}

            lhs = 0 
            for s in self.stocks:
                lhs = lhs + 2*self.t_cost*self.price[s]*x[s]*y[s] \
                            + self.price[s]*(1 - self.t_cost)*x[s] \
                            - 2*self.t_cost*self.price[s]*x0[s]*y[s] \
                            - self.price[s]*(1 - self.t_cost)*x0[s]
                            
            cqm.add_constraint( lhs <= self.budget, label='upper_budget')
            cqm.add_constraint( lhs >= self.budget - 0.003*self.init_budget, 
                                label='lower_budget')

            # indicator constraints 
            for s in self.stocks:
                cqm.add_constraint(x[s] - x0[s]*y[s] >= 0, 
                                    label=f'indicator_constraint_gte_{s}')
                cqm.add_constraint(x[s] - x[s]*y[s] <= x0[s], 
                                    label=f'indicator_constraint_lte_{s}')

        if max_risk: 
            # Adding maximum risk constraint 
            cqm.add_constraint(risk <= max_risk, label='max_risk')

            # Objective: maximize return 
            cqm.set_objective(-1*returns)
        elif min_return:
            # Adding minimum returns constraint
            cqm.add_constraint(returns >= min_return, label='min_return') 

            # Objective: minimize risk 
            cqm.set_objective(risk)
        else: 
            # Objective: minimize mean-variance expression 
            cqm.set_objective(self.alpha*risk - returns)

        cqm.substitute_self_loops()

        self.model['CQM'] = cqm 

    def solve_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Solve CQM.  
        This method allows the user to solve one of 3 cqm problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state. 

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format 
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        self.build_cqm(max_risk, min_return, init_holdings)

        self.sample_set['CQM'] = self.sampler['CQM'].sample_cqm(self.model['CQM'], 
                                                                label="Example - Portfolio Optimization")
        n_samples = len(self.sample_set['CQM'].record)

        feasible_samples = self.sample_set['CQM'].filter(lambda d: d.is_feasible) 

        if not feasible_samples: 
            raise Exception("No feasible solution could be found for this problem instance.")
        else:
            best_feasible = feasible_samples.first

            solution = {}

            solution['stocks'] = {k:int(best_feasible.sample[k]) for k in self.stocks}

            solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['stocks'])

            spending = sum([self.price[s]*max(0, solution['stocks'][s] - self.init_holdings[s]) for s in self.stocks])
            sales = sum([self.price[s]*max(0, self.init_holdings[s] - solution['stocks'][s]) for s in self.stocks])

            transaction = self.t_cost*(spending + sales)

            if self.verbose:
                print(f'Number of feasible solutions: {len(feasible_samples)} out of {n_samples} sampled.')
                print(f'\nBest energy: {self.sample_set["CQM"].first.energy: .2f}')
                print(f'Best energy (feasible): {best_feasible.energy: .2f}')  

            print(f'\nBest feasible solution:')
            print("\n".join("{}\t{:>3}".format(k, v) for k, v in solution['stocks'].items())) 

            print(f"\nEstimated Returns: {solution['return']}")

            print(f"Sales Revenue: {sales:.2f}")

            print(f"Purchase Cost: {spending:.2f}")

            print(f"Transaction Cost: {transaction:.2f}")

            print(f"Variance: {solution['risk']}\n")
            # Calculate total units
            total_units = sum(solution['stocks'].values())

            # Calculate percentage of units for each stock
            unit_percentages = {stock: (units / total_units) * 100 for stock, units in solution['stocks'].items()}
            return {
                "Purchase Cost":spending,
                "Risk":solution['risk'],
                "Estimated Returns":solution['return'],
                "Stocks":solution['stocks'],
                "Investment Ratio":unit_percentages,
                "Model Type": "CQM (Contraint Quadratic Model)"
            } 
    def compute_risk_and_returns(self, solution):
        """Compute the risk and return values of solution.
        """
        variance = 0.0
        for s1, s2 in product(solution, solution):
            variance += (solution[s1] * self.price[s1] 
                        * solution[s2] * self.price[s2]  
                        * self.covariance_matrix[s1][s2])

        est_return = 0
        for stock in solution:
            est_return += solution[stock]*self.price[stock]*self.avg_monthly_returns[stock]

        return round(est_return, 2), round(variance, 2)

    def run(self, min_return=0, max_risk=0, num=0, init_holdings=None): 
        """Execute sequence of load_data --> build_model --> solve.

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            num (int): Number of stocks to be randomnly generated. 
            init_holdings (float): Initial holdings, or initial portfolio state. 
        """
        print("--------------------sdsdsd-sdsd")
        self.load_data(num=num)
        if self.model_type=='CQM': 
            print(f"\nCQM run...")
            self.solution['CQM'] = self.solve_cqm(min_return=min_return, 
                                                    max_risk=max_risk, 
                                                    init_holdings=init_holdings)
        else:
            print(f"\nDQM run...")
            if len(self.alpha_list) > 1 or len(self.gamma_list) > 1:
                print("\nStarting DQM Grid Search...")
                self.dqm_grid_search()

            self.build_dqm()
            self.solution['DQM'] = self.solve_dqm()
        return self.solution['CQM']

