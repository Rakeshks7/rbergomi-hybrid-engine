# rBergomi Hybrid Engine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade implementation of the **Rough Bergomi (rBergomi)** volatility model. This project addresses the computational challenges of rough volatility ($H < 0.5$) by utilizing a hybrid architecture: **Python** for linear algebra and **C++ (via PyBind11)** for high-performance Monte Carlo path evolution.

##  Key Features

* **Rough Volatility:** Implements the rBergomi model where volatility is driven by a fractional Brownian motion (fBm) with Hurst parameter $H < 0.5$, capturing the "rough" nature of short-term market volatility.
* **Exact Simulation:** Uses Cholesky decomposition of the covariance matrix for the Volterra process to ensure exact correlation structures (no Euler-approximation errors in the kernel).
* **Hybrid Architecture:**
    * **Python (NumPy):** Efficiently handles matrix operations and vectorized noise generation.
    * **C++ Extension:** Handles the sequential Euler-Maruyama time-stepping loop, providing a **20x-50x speedup** over pure Python.
* **Calibration Engine:** Includes a module to calibrate model parameters ($\nu, H, \rho$) to market implied volatility surfaces using the Nelder-Mead optimization algorithm.

##  Mathematical Framework

The model dynamics are defined as:

$$
\begin{aligned}
\frac{dS_t}{S_t} &= \sqrt{V_t} dW_t^1 \\
V_t &= \xi_0(t) \exp\left( \eta Y_t - \frac{1}{2} \eta^2 t^{2H} \right) \\
Y_t &= \int_0^t (t-s)^{H-1/2} dW_s^2
\end{aligned}
$$

Where:
* $Y_t$ is the Volterra process driving the volatility.
* $d\langle W^1, W^2 \rangle_t = \rho dt$.
* $H$ is the Hurst parameter. Empirical studies suggest $H \approx 0.1$ for equity indices (SPX).

## Project Structure
* src/fbm.py: Logic for Fractional Brownian Motion and Cholesky decomposition.
* src/cpp/rbergomi_engine.cpp: The C++ kernel containing the optimized loop.
* src/model.py: The main RBergomiPricer class that bridges Python and C++.
* src/calibration.py: Optimization routines for parameter fitting.

## Disclaimer

This software is for educational and research purposes only. It is not intended for use in actual trading or investment decisions. The author is not responsible for any financial losses incurred from the use of this code.