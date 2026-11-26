/// @file sbc.cpp
/// @brief Rank computation, binned chi-squared and ECDF uniformity tests for simulation-based calibration.
#include <datamunge/bayes/sbc.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::bayes {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Maximum absolute deviation of the ECDF of ranks on {0..L} from the discrete uniform CDF.
/// @param ranks Ranks.
/// @param draws L.
/// @return max_k |#(r <= k)/N - (k + 1)/(L + 1)| over k = 0..L-1.
double ecdf_deviation(const std::vector<std::size_t>& ranks, std::size_t draws) {
    std::vector<std::size_t> counts(draws + 1, 0);
    for (const std::size_t r : ranks)
        ++counts[r];
    const double n = static_cast<double>(ranks.size()), m = static_cast<double>(draws + 1);
    double cum = 0, worst = 0;
    for (std::size_t k = 0; k < draws; ++k) {
        cum += static_cast<double>(counts[k]);
        worst = std::max(worst, std::abs(cum / n - static_cast<double>(k + 1) / m));
    }
    return worst;
}
} // namespace

SbcResult simulation_based_calibration(const SbcPriorDraw& prior, const SbcSimulator& simulate, const SbcFit& fit,
                                       const SbcOptions& options) {
    if (options.replications < 2 || options.bins < 2)
        throw std::invalid_argument("simulation_based_calibration: need at least two replications and bins");
    random::SplitMix64 rng(options.seed);
    SbcResult out;
    std::size_t dim = 0;
    for (std::size_t rep = 0; rep < options.replications; ++rep) {
        const std::vector<double> theta = prior(rng);
        if (theta.empty() || (dim != 0 && theta.size() != dim))
            throw std::invalid_argument("simulation_based_calibration: prior draws must share a nonzero dimension");
        const std::vector<double> data = simulate(theta, rng);
        std::vector<std::vector<double>> draws;
        try {
            draws = fit(data, rng);
        } catch (const std::exception&) {
            ++out.failures;
            continue;
        }
        if (draws.empty() || (out.posterior_draws != 0 && draws.size() != out.posterior_draws))
            throw std::invalid_argument("simulation_based_calibration: every fit must return the same number of draws");
        for (const auto& d : draws)
            if (d.size() != theta.size())
                throw std::invalid_argument("simulation_based_calibration: draw dimension differs from the prior");
        if (dim == 0) {
            dim = theta.size();
            out.parameters.resize(dim);
            out.posterior_draws = draws.size();
        }
        for (std::size_t j = 0; j < dim; ++j) {
            std::size_t below = 0, equal = 0;
            for (const auto& d : draws) {
                below += d[j] < theta[j];
                equal += d[j] == theta[j];
            }
            const std::size_t extra =
                equal ? std::min(equal, static_cast<std::size_t>(rng.uniform01() * static_cast<double>(equal + 1)))
                      : 0;
            out.parameters[j].ranks.push_back(below + extra);
        }
    }
    if (out.parameters.empty() || out.parameters[0].ranks.size() < 2)
        throw std::runtime_error("simulation_based_calibration: too few successful fits");
    const std::size_t l = out.posterior_draws, n = out.parameters[0].ranks.size();
    const std::size_t bins = std::min(options.bins, l + 1);
    // Expected counts per bin from the exact number of rank values each bin holds.
    std::vector<double> expected(bins, 0.0);
    for (std::size_t r = 0; r <= l; ++r)
        expected[r * bins / (l + 1)] += static_cast<double>(n) / static_cast<double>(l + 1);
    // Null distribution of the ECDF deviation.
    std::vector<double> null_dev;
    std::vector<std::size_t> sim(n);
    for (std::size_t s = 0; s < options.ecdf_simulations; ++s) {
        for (auto& r : sim)
            r = std::min(l, static_cast<std::size_t>(rng.uniform01() * static_cast<double>(l + 1)));
        null_dev.push_back(ecdf_deviation(sim, l));
    }
    for (auto& p : out.parameters) {
        p.histogram.assign(bins, 0);
        double mean = 0;
        for (const std::size_t r : p.ranks) {
            ++p.histogram[r * bins / (l + 1)];
            mean += static_cast<double>(r) / static_cast<double>(l) / static_cast<double>(n);
        }
        p.mean_rank_fraction = mean;
        for (std::size_t b = 0; b < bins; ++b) {
            const double diff = static_cast<double>(p.histogram[b]) - expected[b];
            p.chi_squared += diff * diff / expected[b];
        }
        p.chi_squared_p_value = 1 - random::chi_squared_cdf(p.chi_squared, static_cast<double>(bins - 1));
        p.ecdf_max_deviation = ecdf_deviation(p.ranks, l);
        std::size_t exceed = 0;
        for (const double v : null_dev)
            exceed += v >= p.ecdf_max_deviation - 1e-12;
        p.ecdf_p_value = (1.0 + static_cast<double>(exceed)) / (1.0 + static_cast<double>(null_dev.size()));
    }
    return out;
}

} // namespace datamunge::bayes
