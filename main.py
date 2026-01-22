import numpy as np
import matplotlib.pyplot as plt
from src.model import RBergomiPricer
import time

def run_production_demo():
    print("--- Rough Bergomi (rBergomi) Pricing Engine ---")

    params = {
        'xi_0': 0.04,     
        'nu': 1.9,        
        'rho': -0.7,      
        'H': 0.07,        
        'T': 1/12,        
        'n_steps': 100    
    }
    
    S0 = 4000
    K = 4000
    n_paths = 20000
    
    print(f"Configuration: {params}")
    print(f"Simulating {n_paths} paths...")

    start_time = time.time()
    pricer = RBergomiPricer(**params)

    price = pricer.price_option(S0, K, n_paths, option_type='call')
    end_time = time.time()
    
    print(f"\nResults:")
    print(f"ATM Call Price: {price:.4f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    Z_demo = np.random.normal(size=(5, params['n_steps']))
    Y_paths = pricer.volterra_engine.simulate_volterra_paths(Z_demo)
    
    plt.figure(figsize=(10, 6))
    time_grid = np.linspace(0, params['T'], params['n_steps'])
    for i in range(5):
        plt.plot(time_grid, Y_paths[i], label=f'Path {i}')
    
    plt.title(f"Rough Volatility Paths (H={params['H']})")
    plt.xlabel("Time (Years)")
    plt.ylabel("Volterra Process Y_t")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_production_demo()