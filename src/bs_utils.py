import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    if price <= 0: return 0.0
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price
    
    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan