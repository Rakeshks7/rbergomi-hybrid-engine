import numpy as np
from scipy.optimize import minimize
from src.model import RBergomiPricer
from src.bs_utils import implied_volatility

class RBergomiCalibrator:
    def __init__(self, market_strikes, market_vols, S0, T, r=0.0):
        self.market_strikes = np.array(market_strikes)
        self.market_vols = np.array(market_vols)
        self.S0 = S0
        self.T = T
        self.r = r # Risk free rate assumed 0 for simplicity in this demo

    def objective_function(self, params):
        H, nu, rho = params

        if not (0.01 < H < 0.5): return 1e6
        if not (0.1 < nu < 5.0): return 1e6
        if not (-1.0 < rho < 1.0): return 1e6

        np.random.seed(42) 

        atm_vol = self.market_vols[len(self.market_vols)//2] 
        xi_0 = atm_vol ** 2
        
        pricer = RBergomiPricer(xi_0, nu, rho, H, self.T, n_steps=50) 
        
        model_vols = []
        n_paths = 5000 # Lower paths for speed during optimization loop

        for K in self.market_strikes:
            price = pricer.price_option(self.S0, K, n_paths, option_type='call')
            iv = implied_volatility(price, self.S0, K, self.T, self.r, option_type='call')
            model_vols.append(iv)
            
        model_vols = np.array(model_vols)

        error = np.sum((model_vols - self.market_vols)**2) * 10000
        
        return error

    def calibrate(self):
        x0 = [0.1, 1.5, -0.6]
        
        print("Starting Calibration (Nelder-Mead)...")
        result = minimize(
            self.objective_function, 
            x0, 
            method='Nelder-Mead', 
            tol=1e-2,
            options={'maxiter': 50, 'disp': True}
        )
        
        return result.x