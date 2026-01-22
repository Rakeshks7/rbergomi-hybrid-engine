import numpy as np
import matplotlib.pyplot as plt
from src.calibration import RBergomiCalibrator
from src.model import RBergomiPricer
from src.bs_utils import implied_volatility

def run_calibration_demo():
    print("--- rBergomi Calibration Demo ---")

    S0 = 4000
    T = 1/12
    strikes = [3800, 3900, 4000, 4100, 4200]

    market_vols = [0.28, 0.24, 0.20, 0.17, 0.15] 
    
    print(f"Target Market Vols: {market_vols}")

    calibrator = RBergomiCalibrator(strikes, market_vols, S0, T)
    opt_params = calibrator.calibrate()
    
    H_opt, nu_opt, rho_opt = opt_params
    print("\n--- Calibration Complete ---")
    print(f"Optimal H   (Roughness)  : {H_opt:.4f}")
    print(f"Optimal nu  (Vol of Vol) : {nu_opt:.4f}")
    print(f"Optimal rho (Correlation): {rho_opt:.4f}")

    np.random.seed(42) # Ensure we match the calibration seed
    xi_0 = market_vols[2]**2 # rough approximation of ATM var
    
    pricer = RBergomiPricer(xi_0, nu_opt, rho_opt, H_opt, T, n_steps=100)
    n_paths_verify = 20000
    
    calibrated_vols = []
    print("\nVerifying fit with high-precision simulation...")
    for K in strikes:
        price = pricer.price_option(S0, K, n_paths_verify, 'call')
        iv = implied_volatility(price, S0, K, T, 0.0, 'call')
        calibrated_vols.append(iv)

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_vols, 'ro-', label='Market Data (Target)')
    plt.plot(strikes, calibrated_vols, 'b*--', label=f'rBergomi Fit (H={H_opt:.2f})')
    
    plt.title(f"rBergomi Calibration\nParams: H={H_opt:.2f}, Nu={nu_opt:.2f}, Rho={rho_opt:.2f}")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    run_calibration_demo()