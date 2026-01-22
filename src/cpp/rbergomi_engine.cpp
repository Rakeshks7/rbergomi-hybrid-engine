/*
 * src/cpp/rbergomi_engine.cpp
 * * Production-grade C++ Kernel for rBergomi Path Evolution
 * Uses PyBind11 for seamless Python interop.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h> // OpenMP for multi-threading (Optional but recommended)

namespace py = pybind11;

/*
 * Evolve Paths Kernel
 * * Args:
 * S0: Initial Spot Price
 * dt: Time step size
 * dW_price: (n_paths x n_steps) numpy array of Brownian increments
 * V_t: (n_paths x n_steps) numpy array of Variance paths (pre-computed in Python)
 * * Returns:
 * S_T: (n_paths) numpy array of Terminal Prices
 */
py::array_t<double> evolve_paths(
    double S0,
    double dt,
    py::array_t<double> dW_price,
    py::array_t<double> V_t
) {
    // 1. Request buffer info (Zero-Copy access)
    py::buffer_info buf_dW = dW_price.request();
    py::buffer_info buf_V = V_t.request();

    if (buf_dW.ndim != 2 || buf_V.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");

    size_t n_paths = buf_dW.shape[0];
    size_t n_steps = buf_dW.shape[1];

    // 2. Get pointers to data (Fastest access method)
    double* ptr_dW = static_cast<double*>(buf_dW.ptr);
    double* ptr_V = static_cast<double*>(buf_V.ptr);

    // 3. Prepare Output Array
    auto result = py::array_t<double>(n_paths);
    py::buffer_info buf_res = result.request();
    double* ptr_res = static_cast<double*>(buf_res.ptr);

    // 4. The Heavy Loop (Parallelized with OpenMP if available)
    // We iterate over paths independently.
    
    #pragma omp parallel for
    for (size_t i = 0; i < n_paths; i++) {
        double current_log_S = std::log(S0);
        
        for (size_t j = 0; j < n_steps; j++) {
            // Index logic: Matrix is flat in memory [row * cols + col]
            size_t idx = i * n_steps + j;
            
            double v_curr = ptr_V[idx]; // Variance at start of step
            double dw = ptr_dW[idx];    // Brownian increment
            
            // Euler-Maruyama Step in Log-Space (More stable)
            // d(ln S) = -0.5 * V * dt + sqrt(V) * dW
            current_log_S += -0.5 * v_curr * dt + std::sqrt(v_curr) * dw;
        }
        
        ptr_res[i] = std::exp(current_log_S);
    }

    return result;
}

// PyBind11 Module Definition
PYBIND11_MODULE(rbergomi_cpp, m) {
    m.doc() = "High-performance C++ backend for rBergomi Model";
    m.def("evolve_paths", &evolve_paths, "Evolve asset paths using Euler-Maruyama");
}