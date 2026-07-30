#pragma once

/// \file hopfield_network.hpp
/// \brief The Hopfield network: a recurrent associative memory that stores binary
///        patterns as fixed points and recovers them from noisy or partial cues.
///
/// A Hopfield network (Hopfield 1982) is a fully connected recurrent network of
/// \f$N\f$ bipolar neurons \f$s_i\in\{-1,+1\}\f$ with a symmetric, zero-diagonal
/// weight matrix \f$W\f$. Patterns are stored by the *Hebbian* outer-product rule
/// \f[
///   W_{ij} = \frac{1}{N}\sum_{p} \xi^{(p)}_i \xi^{(p)}_j \quad (i\ne j),\qquad W_{ii}=0,
/// \f]
/// so each stored pattern \f$\xi^{(p)}\f$ becomes a (local) minimum of the energy
/// \f[
///   E(s) = -\tfrac12 \sum_{i\ne j} W_{ij}\, s_i s_j.
/// \f]
/// *Recall* starts from a probe state and repeatedly applies
/// \f$s_i \leftarrow \operatorname{sign}\big(\sum_j W_{ij} s_j\big)\f$. With a
/// symmetric matrix and *asynchronous* updates (one neuron at a time) the energy is
/// non-increasing and the dynamics converge to a fixed point -- ideally the stored
/// pattern closest to the probe. This makes the network a *content-addressable
/// memory*: present a corrupted version of a memory and the network relaxes to the
/// clean one. Capacity is limited: roughly \f$0.138N\f$ random patterns can be
/// stored before spurious minima and crosstalk dominate.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// A trained Hopfield network: a symmetric, zero-diagonal weight matrix.
struct HopfieldNetwork {
    std::vector<std::vector<double>> weights;  ///< \f$W_{ij}\f$, \f$N\times N\f$, symmetric, zero diagonal.
    std::size_t                      size() const { return weights.size(); }
};

/// \brief Store bipolar patterns by the Hebbian outer-product rule.
///
/// Each pattern is a vector of \f$\pm 1\f$ of common length \f$N\f$. The weights are
/// \f$W_{ij}=\frac1N\sum_p \xi^{(p)}_i\xi^{(p)}_j\f$ for \f$i\ne j\f$ and \f$W_{ii}=0\f$.
inline HopfieldNetwork hopfield_train(const std::vector<std::vector<int>>& patterns) {
    HopfieldNetwork net;
    if (patterns.empty()) return net;
    const std::size_t n = patterns.front().size();
    for (const auto& p : patterns)
        if (p.size() != n) throw std::invalid_argument("hopfield: patterns differ in length");

    net.weights.assign(n, std::vector<double>(n, 0.0));
    for (const auto& p : patterns)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (i != j) net.weights[i][j] += static_cast<double>(p[i] * p[j]);
    const double inv = 1.0 / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) net.weights[i][j] *= inv;
    return net;
}

/// Energy \f$E(s)=-\tfrac12\sum_{i\ne j}W_{ij}s_i s_j\f$ of a state.
inline double hopfield_energy(const HopfieldNetwork& net, const std::vector<int>& state) {
    const std::size_t n = net.size();
    double            e = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) e -= net.weights[i][j] * state[i] * state[j];
    return 0.5 * e;
}

/// \brief Recall: relax a probe state to a fixed point by asynchronous updates.
///
/// Repeatedly updates neurons one at a time in a random order,
/// \f$s_i\leftarrow\operatorname{sign}(\sum_j W_{ij}s_j)\f$ (ties keep the current
/// sign), until a full sweep makes no change or \p max_sweeps is reached. Because
/// each asynchronous flip cannot increase the energy, convergence is guaranteed.
///
/// \param net        Trained network.
/// \param state      Initial (possibly corrupted) bipolar state; returned relaxed.
/// \param max_sweeps Maximum full update sweeps.
/// \param seed       RNG seed for the neuron update order.
inline std::vector<int> hopfield_recall(const HopfieldNetwork& net, std::vector<int> state,
                                        std::size_t max_sweeps = 100, std::uint64_t seed = 0) {
    const std::size_t n = net.size();
    if (state.size() != n) throw std::invalid_argument("hopfield: state length mismatch");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::mt19937_64 rng(seed);

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        std::shuffle(order.begin(), order.end(), rng);
        bool changed = false;
        for (std::size_t idx : order) {
            double field = 0.0;
            for (std::size_t j = 0; j < n; ++j) field += net.weights[idx][j] * state[j];
            const int updated = field >= 0.0 ? 1 : -1;
            if (updated != state[idx]) {
                state[idx] = updated;
                changed    = true;
            }
        }
        if (!changed) break;
    }
    return state;
}

/// Count of positions where two bipolar states differ (Hamming distance).
inline std::size_t hopfield_hamming(const std::vector<int>& a, const std::vector<int>& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) d += (a[i] != b[i]);
    return d;
}

}  // namespace datamunge::algorithms
