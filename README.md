# Multithreaded Monte Carlo Option Pricer

A modern C++ quantitative finance engine designed to price financial derivatives using Monte Carlo simulations. This project leverages C++ multithreading (`std::async`, `<future>`) to distribute millions of simulated price paths across available CPU cores, significantly reducing computation time.

## 🚀 Features

* **European Options:** Prices standard European Call options and benchmarks the Monte Carlo result against the analytical Black-Scholes formula.
* **Asian Options:** Prices path-dependent Arithmetic Asian Call options using discrete time steps.
* **Greeks Calculation:** Computes first and second-order sensitivities (Delta and Gamma) using the bump-and-reprice (finite difference) method.
* **Parallel Execution:** Automatically detects hardware concurrency and divides workload across worker threads.
* **Thread-Safe PRNG:** Uses thread-local `std::mt19937_64` instances with a robust seeding strategy to ensure statistically independent and reproducible paths without race conditions.

## 🧮 Mathematical Models

* **Underlying Dynamics:** Geometric Brownian Motion (GBM) under the risk-neutral measure.
* **European Payoff:** `max(S_T - K, 0)`
* **Asian Payoff (Arithmetic):** `max(S_avg - K, 0)` where `S_avg` is the average spot price across `N` observation steps.

## 🛠️ Prerequisites

To build and run this project, you need a C++ compiler that supports at least **C++11** (C++14/17 recommended). 
* GCC (g++)
* Clang
* MSVC (Visual Studio)

## ⚙️ Build and Run

You can compile this directly from the terminal. 

**Using GCC/Clang:**
```bash
# Compile the code with O3 optimization and pthread (required for threading on Linux/Mac)
g++ -O3 -std=c++11 -pthread main.cpp -o pricer

# Run the executable
./pricer
