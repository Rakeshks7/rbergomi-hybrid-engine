import numpy as np
from src.fbm import GaussianNoiseGenerator, VolterraProcess

try:
    import rbergomi_cpp
    CPP_AVAILABLE = True
    print("Optimization: C++ Backend Loaded.")
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ Backend not found. Falling back to slow Python loops.")

class RBergomiPricer:

    def generate_paths_cpp(self, S0: float, n_paths: int) -> np.ndarray:
        dW1, dW2 = GaussianNoiseGenerator.generate_correlated_brownian(n_paths, self.n_steps, self.rho)
        Z1 = np.random.normal(size=(n_paths, self.n_steps))
        Z2 = np.random.normal(size=(n_paths, self.n_steps))

        Y_t = self.volterra_engine.simulate_volterra_paths(Z2)

        t_grid = np.linspace(self.dt, self.T, self.n_steps)
        drift_correction = -0.5 * (self.nu**2) * (t_grid**(2 * self.H))
        V_t = self.xi_0 * np.exp(self.nu * Y_t + drift_correction)

        dW_price = (self.rho * Z2 + np.sqrt(1 - self.rho**2) * Z1) * np.sqrt(self.dt)

        S_T = rbergomi_cpp.evolve_paths(S0, self.dt, dW_price, V_t)
        
        return S_T

    def price_option(self, S0: float, K: float, n_paths: int, option_type: str = 'call') -> float:
        if CPP_AVAILABLE:
            S_T = self.generate_paths_cpp(S0, n_paths)
        else:
            paths = self.generate_paths(S0, n_paths)
            S_T = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
            
        return np.mean(payoffs)