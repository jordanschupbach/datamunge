#pragma once

/// \file almeida_pineda.hpp
/// \brief Almeida-Pineda recurrent backpropagation: gradient learning for recurrent
///        networks that relax to a fixed point (Almeida 1987; Pineda 1987).
///
/// Ordinary backpropagation trains feed-forward networks. Almeida-Pineda trains
/// *recurrent* networks whose units settle to an equilibrium: with recurrent weights
/// \f$W\f$ and clamped inputs, the free units relax until
/// \f$x_i = f\big(\sum_j W_{ij} x_j\big)\f$. Designated output units are compared with
/// targets, and the gradient of the output error w.r.t. \f$W\f$ is obtained by
/// settling a *second* recurrent system -- the adjoint, run with the *transposed*
/// weights -- to its own fixed point:
/// \f[
///   y_i = f'(a_i)\Big(\sum_j W_{ji}\,y_j + e_i\Big),\qquad
///   \frac{\partial E}{\partial W_{ij}} = y_i\,x_j,
/// \f]
/// where \f$e_i = (x_i - t_i)\f$ for output units and 0 otherwise. This is the
/// recurrent analogue of backprop: the forward pass is a relaxation to equilibrium,
/// and the backward pass is *another* relaxation of the error signal -- both are fixed
/// points, so no unrolling through time is needed (unlike BPTT).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A recurrent network trained by Almeida-Pineda.
struct AlmeidaPinedaNetwork {
    std::size_t                      num_units   = 0;
    std::size_t                      num_inputs  = 0;  ///< Units 0..num_inputs-1 are clamped to the input.
    std::vector<std::size_t>         output_units;     ///< Which units carry the targets.
    std::vector<std::vector<double>> W;                ///< num_units x num_units recurrent weights.
};

/// Result of training.
struct AlmeidaPinedaResult {
    AlmeidaPinedaNetwork net;
    std::vector<double>  error_history;  ///< Mean squared output error after each epoch.
};

namespace detail {

/// Relax the free units to equilibrium; returns activations x (and fills net-inputs a).
inline std::vector<double> ap_settle(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                     std::vector<double>& a, std::size_t iters, double relax) {
    const std::size_t n = net.num_units;
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < net.num_inputs; ++i) x[i] = input[i];  // clamp inputs
    a.assign(n, 0.0);
    for (std::size_t it = 0; it < iters; ++it)
        for (std::size_t i = net.num_inputs; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += net.W[i][j] * x[j];
            a[i]        = s;
            const double xnew = std::tanh(s);
            x[i]              = (1.0 - relax) * x[i] + relax * xnew;
        }
    // Recompute a at the settled state.
    for (std::size_t i = net.num_inputs; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += net.W[i][j] * x[j];
        a[i] = s;
    }
    return x;
}

}  // namespace detail

/// \brief Train a recurrent network with Almeida-Pineda backpropagation.
///
/// \param inputs        Input patterns (each of length \p num_inputs).
/// \param targets       Target values for the \p output_units (each row aligned with \p output_units).
/// \param num_units     Total units (inputs + hidden + outputs).
/// \param num_inputs    Number of clamped input units (indices 0..num_inputs-1).
/// \param output_units  Indices of the output units.
/// \param epochs        Training epochs.
/// \param learning_rate Gradient step size.
/// \param seed          RNG seed for weight init.
/// \param settle_iters  Relaxation iterations for the forward/adjoint fixed points.
/// \param relax         Relaxation damping in (0,1].
inline AlmeidaPinedaResult almeida_pineda_train(const std::vector<std::vector<double>>& inputs,
                                                const std::vector<std::vector<double>>& targets,
                                                std::size_t num_units, std::size_t num_inputs,
                                                const std::vector<std::size_t>& output_units,
                                                std::size_t epochs = 500, double learning_rate = 0.1,
                                                std::uint64_t seed = 0, std::size_t settle_iters = 100,
                                                double relax = 0.5) {
    AlmeidaPinedaResult res;
    AlmeidaPinedaNetwork& net = res.net;
    net.num_units    = num_units;
    net.num_inputs   = num_inputs;
    net.output_units = output_units;
    net.W.assign(num_units, std::vector<double>(num_units, 0.0));

    std::mt19937_64                  rng(seed);
    std::normal_distribution<double> init(0.0, 0.3);
    for (std::size_t i = num_inputs; i < num_units; ++i)
        for (std::size_t j = 0; j < num_units; ++j)
            if (i != j) net.W[i][j] = init(rng);  // no self-loops

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        std::vector<std::vector<double>> grad(num_units, std::vector<double>(num_units, 0.0));
        double                           total_err = 0.0;

        for (std::size_t p = 0; p < inputs.size(); ++p) {
            std::vector<double> a;
            std::vector<double> x = detail::ap_settle(net, inputs[p], a, settle_iters, relax);

            // Error signal e_i at output units.
            std::vector<double> e(num_units, 0.0);
            for (std::size_t k = 0; k < output_units.size(); ++k) {
                const std::size_t u = output_units[k];
                const double      d = x[u] - targets[p][k];
                e[u]                = d;
                total_err += d * d;
            }

            // Adjoint relaxation: y_i = f'(a_i) (sum_j W_ji y_j + e_i).
            std::vector<double> y(num_units, 0.0);
            for (std::size_t it = 0; it < settle_iters; ++it)
                for (std::size_t i = num_inputs; i < num_units; ++i) {
                    double s = 0.0;
                    for (std::size_t j = num_inputs; j < num_units; ++j) s += net.W[j][i] * y[j];
                    const double fp   = 1.0 - std::tanh(a[i]) * std::tanh(a[i]);  // f'(a) for tanh
                    const double ynew = fp * (s + e[i]);
                    y[i]              = (1.0 - relax) * y[i] + relax * ynew;
                }

            // Accumulate gradient dE/dW_ij = y_i x_j (i is a free unit).
            for (std::size_t i = num_inputs; i < num_units; ++i)
                for (std::size_t j = 0; j < num_units; ++j)
                    if (i != j) grad[i][j] += y[i] * x[j];
        }

        for (std::size_t i = num_inputs; i < num_units; ++i)
            for (std::size_t j = 0; j < num_units; ++j) net.W[i][j] -= learning_rate * grad[i][j];

        res.error_history.push_back(total_err / (inputs.size() * output_units.size()));
    }
    return res;
}

/// Settle the network on an input and read off the output-unit activations.
inline std::vector<double> almeida_pineda_predict(const AlmeidaPinedaNetwork& net,
                                                  const std::vector<double>& input,
                                                  std::size_t settle_iters = 100, double relax = 0.5) {
    std::vector<double> a;
    std::vector<double> x = detail::ap_settle(net, input, a, settle_iters, relax);
    std::vector<double> out;
    for (std::size_t u : net.output_units) out.push_back(x[u]);
    return out;
}

}  // namespace datamunge::algorithms
