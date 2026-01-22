import numpy as np
from typing import Tuple

class GaussianNoiseGenerator:
    @staticmethod
    def generate_correlated_brownian(
        n_paths: int, 
        n_steps: int, 
        rho: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        Z1 = np.random.normal(size=(n_paths, n_steps))
        Z2 = np.random.normal(size=(n_paths, n_steps))

        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        return dW1, dW2

class VolterraProcess:
    def __init__(self, n_steps: int, T: float, H: float):
        self.n_steps = n_steps
        self.T = T
        self.H = H
        self.dt = T / n_steps
        self.cov_matrix = self._build_covariance_matrix()
        self.L = np.linalg.cholesky(self.cov_matrix)

    def _build_covariance_matrix(self) -> np.ndarray:
        
        t = np.linspace(self.dt, self.T, self.n_steps)
        cov = np.zeros((self.n_steps, self.n_steps))
        
        alpha = self.H - 0.5
        
        for i in range(self.n_steps):
            for j in range(i + 1):
                s = np.linspace(0, t[j], 100) # Grid for integration
                integrand = ((t[i] - s)**alpha) * ((t[j] - s)**alpha)
                val = np.trapz(integrand, s)
                cov[i, j] = val
                cov[j, i] = val
                
        return cov + 1e-9 * np.eye(self.n_steps)

    def simulate_volterra_paths(self, dW_uncorrelated: np.ndarray) -> np.ndarray:
        Y = self.L @ dW_uncorrelated.T
        return Y.T # Return to (n_paths, n_steps)