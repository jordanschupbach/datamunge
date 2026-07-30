#pragma once

/// \file self_organizing_map.hpp
/// \brief Kohonen's Self-Organizing Map (SOM): an unsupervised neural network that
///        learns a low-dimensional, topology-preserving representation of data.
///
/// A SOM (Kohonen 1982) is a grid of neurons, each carrying a *codebook* weight
/// vector in the input space. Training is competitive: for each input \f$x\f$ the
/// *best-matching unit* (BMU) -- the neuron whose weight is nearest \f$x\f$ -- and
/// its grid neighbors are pulled toward \f$x\f$,
/// \f[
///   w_k \leftarrow w_k + \alpha(t)\,h_{bk}(t)\,(x - w_k),
/// \f]
/// where \f$\alpha(t)\f$ is a decaying learning rate and
/// \f$h_{bk}(t)=\exp\!\big(-\|r_b-r_k\|^2/2\sigma(t)^2\big)\f$ is a Gaussian
/// neighborhood in *grid* coordinates around the BMU \f$b\f$, with the radius
/// \f$\sigma(t)\f$ shrinking over time. Because neighbors move together, grid
/// neighbors come to represent nearby regions of input space: the map *preserves
/// topology*, folding a high-dimensional distribution onto a 2-D sheet. SOMs are
/// used for clustering, visualization, and vector quantization.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A trained Self-Organizing Map: a \c rows x \c cols grid of codebook vectors.
struct SelfOrganizingMap {
    std::size_t                      rows = 0;
    std::size_t                      cols = 0;
    std::size_t                      dim  = 0;
    std::vector<std::vector<double>> weights;  ///< (rows*cols) codebook vectors, row-major grid.

    /// Flattened index of grid cell (r, c).
    std::size_t index(std::size_t r, std::size_t c) const { return r * cols + c; }
};

/// Squared Euclidean distance between two equal-length vectors.
inline double som_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/// Best-matching unit: grid coordinates (row, col) of the codebook vector nearest \p input.
inline std::pair<std::size_t, std::size_t> som_bmu(const SelfOrganizingMap& som,
                                                   const std::vector<double>& input) {
    std::size_t best     = 0;
    double      best_dist = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < som.weights.size(); ++k) {
        const double d = som_sq_dist(som.weights[k], input);
        if (d < best_dist) {
            best_dist = d;
            best      = k;
        }
    }
    return {best / som.cols, best % som.cols};
}

/// Hyper-parameters for SOM training.
struct SOMParameters {
    std::size_t   rows        = 5;
    std::size_t   cols        = 5;
    std::size_t   iterations  = 1000;   ///< Total number of single-sample updates.
    double        learning_rate = 0.5;  ///< Initial \f$\alpha_0\f$ (decays exponentially).
    double        radius        = 0.0;  ///< Initial \f$\sigma_0\f$; <=0 selects max(rows,cols)/2.
    std::uint64_t seed          = 0;
};

/// \brief Train a SOM by online competitive learning with a shrinking Gaussian neighborhood.
///
/// Weights are initialized to random samples drawn from the data. Over \p iterations
/// single-sample updates, both the learning rate \f$\alpha(t)=\alpha_0 e^{-t/T}\f$ and
/// the neighborhood radius \f$\sigma(t)=\sigma_0 e^{-t/\lambda}\f$ decay, so the map
/// first unfolds globally, then fine-tunes locally.
///
/// \param data    Input vectors (all of the same dimension).
/// \param params  Grid size, iteration budget, initial rate/radius, seed.
inline SelfOrganizingMap som_train(const std::vector<std::vector<double>>& data, const SOMParameters& params) {
    if (data.empty()) throw std::invalid_argument("som: no data");
    SelfOrganizingMap som;
    som.rows = params.rows;
    som.cols = params.cols;
    som.dim  = data.front().size();

    std::mt19937_64                             rng(params.seed);
    std::uniform_int_distribution<std::size_t>  pick(0, data.size() - 1);
    som.weights.resize(som.rows * som.cols);
    for (auto& w : som.weights) w = data[pick(rng)];  // initialize from random samples

    const double sigma0 = params.radius > 0.0 ? params.radius
                                              : 0.5 * static_cast<double>(std::max(som.rows, som.cols));
    const double time_const = static_cast<double>(params.iterations) / std::log(sigma0 + 1.0);

    for (std::size_t t = 0; t < params.iterations; ++t) {
        const std::vector<double>& x     = data[pick(rng)];
        const auto [br, bc]              = som_bmu(som, x);
        const double alpha = params.learning_rate * std::exp(-static_cast<double>(t) / static_cast<double>(params.iterations));
        const double sigma = sigma0 * std::exp(-static_cast<double>(t) / time_const);
        const double two_sigma2 = 2.0 * sigma * sigma;

        for (std::size_t r = 0; r < som.rows; ++r)
            for (std::size_t c = 0; c < som.cols; ++c) {
                const double dr = static_cast<double>(r) - static_cast<double>(br);
                const double dc = static_cast<double>(c) - static_cast<double>(bc);
                const double grid_d2 = dr * dr + dc * dc;
                const double h = std::exp(-grid_d2 / two_sigma2);
                if (h < 1e-6) continue;
                auto& w = som.weights[som.index(r, c)];
                for (std::size_t i = 0; i < som.dim; ++i) w[i] += alpha * h * (x[i] - w[i]);
            }
    }
    return som;
}

/// \brief Mean quantization error: average distance from each input to its BMU codebook vector.
inline double som_quantization_error(const SelfOrganizingMap& som, const std::vector<std::vector<double>>& data) {
    if (data.empty()) return 0.0;
    double total = 0.0;
    for (const auto& x : data) {
        const auto [r, c] = som_bmu(som, x);
        total += std::sqrt(som_sq_dist(som.weights[som.index(r, c)], x));
    }
    return total / static_cast<double>(data.size());
}

}  // namespace datamunge::algorithms
