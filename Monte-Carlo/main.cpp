


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>


struct OptionParams {
    double spot;          // S0
    double strike;        // K
    double maturity;      // T 
    double riskFreeRate;  // r
    double volatility;    // sigma
};

struct MonteCarloResult {
    double price;          
    double stdError;       
    std::uint64_t paths;   
};

struct Greeks {
    double delta;
    double gamma;
};


double blackScholesCall(const OptionParams& params) {
    const double S = params.spot;
    const double K = params.strike;
    const double T = params.maturity;
    const double r = params.riskFreeRate;
    const double v = params.volatility;

    const double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
    const double d2 = d1 - v * std::sqrt(T);

    auto normCDF = [](double x) -> double {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    return S * normCDF(d1) - K * std::exp(-r * T) * normCDF(d2);
}


struct WorkerStats {
    double sumPayoff;
    double sumPayoffSq;
    std::uint64_t paths;
};

template <typename PathFunc>
MonteCarloResult monteCarloParallel(std::uint64_t numPaths,
                                    double discountFactor,
                                    PathFunc pathFunc,
                                    std::uint64_t baseSeed = 42) {
    if (numPaths == 0) {
        return {0.0, 0.0, 0};
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned numWorkers = static_cast<unsigned>(
        std::min<std::uint64_t>(hw, numPaths)
    );

    const std::uint64_t pathsPerWorker = numPaths / numWorkers;
    const std::uint64_t remainder      = numPaths % numWorkers;

    auto makeSeed = [baseSeed](std::uint64_t workerIndex) {
      
        std::random_device rd;
        std::uint64_t rdHigh = static_cast<std::uint64_t>(rd()) << 32;
        std::uint64_t rdLow  = static_cast<std::uint64_t>(rd());
        std::uint64_t rdMix  = rdHigh ^ rdLow;

        const std::uint64_t k = 0x9e3779b97f4a7c15ULL;
        std::uint64_t h = baseSeed;
        h ^= workerIndex + k + (h << 6) + (h >> 2);

        return rdMix ^ h;
    };

    auto worker = [=](std::uint64_t workerIndex,
                      std::uint64_t localPaths) -> WorkerStats {
        if (localPaths == 0) {
            return {0.0, 0.0, 0};
        }

        thread_local std::mt19937_64 rng;
        rng.seed(makeSeed(workerIndex));

        std::normal_distribution<double> normal(0.0, 1.0);

        double sum   = 0.0;
        double sumSq = 0.0;

        for (std::uint64_t i = 0; i < localPaths; ++i) {
            const double payoff = pathFunc(rng, normal);
            sum   += payoff;
            sumSq += payoff * payoff;
        }

        return {sum, sumSq, localPaths};
    };

    std::vector<std::future<WorkerStats>> futures;
    futures.reserve(numWorkers);

    std::uint64_t launchedPaths = 0;
    for (unsigned w = 0; w < numWorkers; ++w) {
        std::uint64_t local = pathsPerWorker + (w < remainder ? 1 : 0);
        launchedPaths += local;
        futures.emplace_back(
            std::async(std::launch::async, worker,
                       static_cast<std::uint64_t>(w), local)
        );
    }

    (void)launchedPaths;

    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    std::uint64_t totalPaths = 0;

    for (auto& f : futures) {
        WorkerStats ws = f.get();
        totalSum   += ws.sumPayoff;
        totalSumSq += ws.sumPayoffSq;
        totalPaths += ws.paths;
    }

    const double n       = static_cast<double>(totalPaths);
    const double mean    = totalSum / n;
    const double meanSq  = totalSumSq / n;
    const double variance = std::max(0.0, meanSq - mean * mean);

    const double price    = discountFactor * mean;
    const double stdError = discountFactor * std::sqrt(variance / n);

    return {price, stdError, totalPaths};
}

MonteCarloResult priceEuropeanCallMC(const OptionParams& params,
                                     std::uint64_t numPaths,
                                     std::uint64_t baseSeed = 42) {
    const double S = params.spot;
    const double K = params.strike;
    const double T = params.maturity;
    const double r = params.riskFreeRate;
    const double v = params.volatility;

    const double drift     = (r - 0.5 * v * v) * T;
    const double diffusion = v * std::sqrt(T);
    const double discount  = std::exp(-r * T);

    auto pathFunc = [=](std::mt19937_64& rng,
                        std::normal_distribution<double>& normal) -> double {
        const double Z  = normal(rng);
        const double ST = S * std::exp(drift + diffusion * Z);
        return std::max(ST - K, 0.0);
    };

    return monteCarloParallel(numPaths, discount, pathFunc, baseSeed);
}


MonteCarloResult priceAsianArithmeticCallMC(const OptionParams& params,
                                            std::uint64_t numPaths,
                                            std::uint64_t baseSeed = 4242,
                                            std::uint32_t numTimeSteps = 252) {
    const double S0 = params.spot;
    const double K  = params.strike;
    const double T  = params.maturity;
    const double r  = params.riskFreeRate;
    const double v  = params.volatility;

    const double dt         = T / static_cast<double>(numTimeSteps);
    const double drift_dt   = (r - 0.5 * v * v) * dt;
    const double diff_sdt   = v * std::sqrt(dt);
    const double discount   = std::exp(-r * T);

    auto pathFunc = [=](std::mt19937_64& rng,
                        std::normal_distribution<double>& normal) -> double {
        double S      = S0;
        double sumS   = 0.0;

        for (std::uint32_t step = 0; step < numTimeSteps; ++step) {
            const double Z = normal(rng);
            S *= std::exp(drift_dt + diff_sdt * Z);
            sumS += S;
        }

        const double Savg  = sumS / static_cast<double>(numTimeSteps);
        const double payoff = std::max(Savg - K, 0.0);
        return payoff;
    };

    return monteCarloParallel(numPaths, discount, pathFunc, baseSeed);
}

template <typename Pricer>
Greeks computeGreeksSpot(const OptionParams& baseParams,
                         double bumpRelative,
                         Pricer pricer) {
    const double S  = baseParams.spot;
    const double h  = S * bumpRelative;

    OptionParams down = baseParams;
    OptionParams mid  = baseParams;
    OptionParams up   = baseParams;

    down.spot = S - h;
    mid.spot  = S;
    up.spot   = S + h;

    const std::uint64_t seedDown = 1001;
    const std::uint64_t seedMid  = 2001;
    const std::uint64_t seedUp   = 3001;

    const MonteCarloResult priceDown = pricer(down, seedDown);
    const MonteCarloResult priceMid  = pricer(mid,  seedMid);
    const MonteCarloResult priceUp   = pricer(up,   seedUp);

    const double delta = (priceUp.price - priceDown.price) / (2.0 * h);
    const double gamma = (priceUp.price - 2.0 * priceMid.price + priceDown.price)
                         / (h * h);

    return {delta, gamma};
}


void printParams(const OptionParams& params) {
    std::cout << "OPTION PARAMETERS\n"
              << "  Spot Price   (S0)  : " <<std::setw(10) << params.spot << "\n"
              << "  Strike Price (K)   : " << std::setw(10) << params.strike << "\n"
              << "  Maturity     (T)   : " << std::setw(10) << params.maturity << " years\n"
              << "  Risk-Free    (r)   : " << std::setw(10)
              << (params.riskFreeRate * 100.0) << " %\n"
              << "  Volatility   (sig) : " << std::setw(10)
              << (params.volatility * 100.0) << " %\n\n";
}

void printBenchmarkHeader() {
    std::cout << "MULTITHREADED MONTE CARLO BENCHMARK\n" ; 
             
}

void printEuropeanReport(const MonteCarloResult& mc,
                         const Greeks& greeks,
                         double bsPrice) {
    std::cout << "European Call (GBM, risk-neutral)\n"
              << "  Monte Carlo Price   : " << std::setw(12) << mc.price << "\n"
              << "  Std Error           : " << std::setw(12) << mc.stdError << "\n"
              << "  Black-Scholes Price : " << std::setw(12) << bsPrice << "\n"
              << "  Delta (MC)          : " << std::setw(12) << greeks.delta << "\n"
              << "  Gamma (MC)          : " << std::setw(12) << greeks.gamma << "\n"
              << "  Paths               : " << std::setw(12) << mc.paths << "\n\n";
}

void printAsianReport(const MonteCarloResult& mc,
                      const Greeks& greeks,
                      std::uint32_t numTimeSteps) {
    std::cout << "Arithmetic Asian Call (discrete, N = " << numTimeSteps << ")\n"
              << "  Monte Carlo Price   : " << std::setw(12) << mc.price << "\n"
              << "  Std Error           : " << std::setw(12) << mc.stdError << "\n"
              << "  Delta (MC)          : " << std::setw(12) << greeks.delta << "\n"
              << "  Gamma (MC)          : " << std::setw(12) << greeks.gamma << "\n"
              << "  Paths               : " << std::setw(12) << mc.paths << "\n\n";
}


int main() {
    const OptionParams option {
        100.0,   // spot
        105.0,   // strike
        1.0,     // maturity
        0.05,    // risk-free rate
        0.20     // volatility
    };

    constexpr std::uint64_t NUM_PATHS   = 10'000'000ULL;
    constexpr double        BUMP_REL    = 0.01;      
    constexpr std::uint32_t ASIAN_STEPS = 252;       

    printBenchmarkHeader();
    printParams(option);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // European call
    const MonteCarloResult euroMc =
        priceEuropeanCallMC(option, NUM_PATHS, 42);

    const Greeks euroGreeks =
        computeGreeksSpot(option, BUMP_REL,
                          [](const OptionParams& p, std::uint64_t seed) {
                              return priceEuropeanCallMC(p, NUM_PATHS, seed);
                          });

    const double euroBs = blackScholesCall(option);

    // Asian call
    const MonteCarloResult asianMc =
        priceAsianArithmeticCallMC(option, NUM_PATHS, 4242, ASIAN_STEPS);

    const Greeks asianGreeks =
        computeGreeksSpot(option, BUMP_REL,
                          [=](const OptionParams& p, std::uint64_t seed) {
                              return priceAsianArithmeticCallMC(p, NUM_PATHS, seed, ASIAN_STEPS);
                          });

    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = t1 - t0;

    printEuropeanReport(euroMc, euroGreeks, euroBs);
    printAsianReport(asianMc, asianGreeks, ASIAN_STEPS);

    std::cout << "Total elapsed time (European + Asian, multithreaded): "
              << elapsed.count() << " seconds\n\n";

 

    return 0;
}
