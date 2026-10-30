/// @file almeida_pineda.cpp
/// @brief Forward and adjoint relaxations, gradients and training for Almeida-Pineda networks.

#include <datamunge/algorithms/almeida_pineda.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

AlmeidaPinedaNetwork almeida_pineda_network(std::size_t num_units, std::size_t num_inputs,
                                            std::vector<std::size_t> output_units, double init_sd, std::uint64_t seed) {
    if (num_inputs >= num_units) throw std::invalid_argument("almeida_pineda_network: need at least one free unit");
    for (const std::size_t u : output_units)
        if (u < num_inputs || u >= num_units)
            throw std::invalid_argument("almeida_pineda_network: output units must be free units");
    AlmeidaPinedaNetwork net;
    net.num_units = num_units;
    net.num_inputs = num_inputs;
    net.output_units = std::move(output_units);
    net.W.assign(num_units, std::vector<double>(num_units, 0.0));
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> init(0.0, init_sd);
    for (std::size_t i = num_inputs; i < num_units; ++i)
        for (std::size_t j = 0; j < num_units; ++j)
            if (i != j) net.W[i][j] = init(rng);  // no self-loops
    return net;
}

AlmeidaPinedaState almeida_pineda_settle(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                         const AlmeidaPinedaSettleOptions& options) {
    const std::size_t n = net.num_units, m = net.num_inputs;
    if (input.size() < m) throw std::invalid_argument("almeida_pineda_settle: input shorter than num_inputs");
    AlmeidaPinedaState s;
    s.x.assign(n, 0.0);
    std::copy_n(input.begin(), m, s.x.begin());
    s.a.assign(n, 0.0);
    std::vector<double> next(n);
    for (s.iterations = 0; s.iterations < options.max_iterations;) {
        double change = 0.0;
        for (std::size_t i = m; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) sum += net.W[i][j] * s.x[j];
            next[i] = (1.0 - options.relax) * s.x[i] + options.relax * std::tanh(sum);
            change = std::max(change, std::abs(next[i] - s.x[i]));
        }
        std::copy(next.begin() + static_cast<std::ptrdiff_t>(m), next.end(), s.x.begin() + static_cast<std::ptrdiff_t>(m));
        ++s.iterations;
        s.residuals.push_back(change);
        if (change < options.tolerance) {
            s.converged = true;
            break;
        }
    }
    for (std::size_t i = m; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += net.W[i][j] * s.x[j];
        s.a[i] = sum;
    }
    return s;
}

AlmeidaPinedaGradient almeida_pineda_gradient(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                              const std::vector<double>& target,
                                              const AlmeidaPinedaSettleOptions& options) {
    const std::size_t n = net.num_units, m = net.num_inputs;
    if (target.size() != net.output_units.size())
        throw std::invalid_argument("almeida_pineda_gradient: one target per output unit is needed");
    AlmeidaPinedaGradient g;
    g.forward = almeida_pineda_settle(net, input, options);
    const auto& x = g.forward.x;
    std::vector<double> e(n, 0.0), fp(n, 0.0);
    for (std::size_t k = 0; k < net.output_units.size(); ++k) {
        const std::size_t u = net.output_units[k];
        e[u] = x[u] - target[k];
        g.error += 0.5 * e[u] * e[u];
    }
    for (std::size_t i = m; i < n; ++i) {
        const double t = std::tanh(g.forward.a[i]);
        fp[i] = 1.0 - t * t;
    }
    // Adjoint relaxation y <- (1 - rho) y + rho D (W^T y + e), synchronous like the forward pass.
    g.y.assign(n, 0.0);
    std::vector<double> next(n, 0.0);
    bool adjoint_converged = false;
    for (g.adjoint_iterations = 0; g.adjoint_iterations < options.max_iterations;) {
        double change = 0.0;
        for (std::size_t i = m; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t j = m; j < n; ++j) sum += net.W[j][i] * g.y[j];
            next[i] = (1.0 - options.relax) * g.y[i] + options.relax * fp[i] * (sum + e[i]);
            change = std::max(change, std::abs(next[i] - g.y[i]));
        }
        std::copy(next.begin() + static_cast<std::ptrdiff_t>(m), next.end(), g.y.begin() + static_cast<std::ptrdiff_t>(m));
        ++g.adjoint_iterations;
        g.adjoint_residuals.push_back(change);
        if (change < options.tolerance) {
            adjoint_converged = true;
            break;
        }
    }
    g.converged = g.forward.converged && adjoint_converged;
    g.dW.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = m; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j) g.dW[i][j] = g.y[i] * x[j];
    return g;
}

AlmeidaPinedaResult almeida_pineda_train(const std::vector<std::vector<double>>& inputs,
                                         const std::vector<std::vector<double>>& targets, std::size_t num_units,
                                         std::size_t num_inputs, const std::vector<std::size_t>& output_units,
                                         std::size_t epochs, double learning_rate, std::uint64_t seed,
                                         std::size_t settle_iters, double relax) {
    AlmeidaPinedaResult res;
    res.net = almeida_pineda_network(num_units, num_inputs, output_units, 0.3, seed);
    const AlmeidaPinedaSettleOptions options{settle_iters, 0.0, relax};
    for (std::size_t ep = 0; ep < epochs; ++ep) {
        std::vector<std::vector<double>> grad(num_units, std::vector<double>(num_units, 0.0));
        double total = 0.0;
        for (std::size_t p = 0; p < inputs.size(); ++p) {
            const auto g = almeida_pineda_gradient(res.net, inputs[p], targets[p], options);
            total += 2.0 * g.error;
            for (std::size_t i = num_inputs; i < num_units; ++i)
                for (std::size_t j = 0; j < num_units; ++j) grad[i][j] += g.dW[i][j];
        }
        for (std::size_t i = num_inputs; i < num_units; ++i)
            for (std::size_t j = 0; j < num_units; ++j) res.net.W[i][j] -= learning_rate * grad[i][j];
        res.error_history.push_back(total / static_cast<double>(inputs.size() * output_units.size()));
    }
    return res;
}

std::vector<double> almeida_pineda_predict(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                           std::size_t settle_iters, double relax) {
    const auto s = almeida_pineda_settle(net, input, {settle_iters, 0.0, relax});
    std::vector<double> out;
    for (const std::size_t u : net.output_units) out.push_back(s.x[u]);
    return out;
}

}  // namespace datamunge::algorithms
