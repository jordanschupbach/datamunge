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
/// \f$y_i = f^{\prime}(a_i)\bigl(\sum_j W_{ji} y_j + e_i\bigr)\f$ and
/// \f$\partial E/\partial W_{ij} = y_i x_j\f$,
/// where \f$e_i = (x_i - t_i)\f$ for output units and 0 otherwise. This is the
/// recurrent analogue of backprop: the forward pass is a relaxation to equilibrium,
/// and the backward pass is *another* relaxation of the error signal -- both are fixed
/// points, so no unrolling through time is needed (unlike BPTT).
///
/// The pieces are exposed separately -- almeida_pineda_settle(), almeida_pineda_gradient() --
/// so the fixed points and gradients can be inspected; stats::AlmeidaPinedaClassifier fits a
/// classifier from a DataFrame.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief A recurrent network trained by Almeida-Pineda.
struct AlmeidaPinedaNetwork {
    std::size_t                      num_units   = 0;  ///< Total units.
    std::size_t                      num_inputs  = 0;  ///< Units 0..num_inputs-1 are clamped to the input.
    std::vector<std::size_t>         output_units;     ///< Which units carry the targets.
    std::vector<std::vector<double>> W;                ///< num_units x num_units recurrent weights; row i feeds unit i.
};

/// @brief Result of almeida_pineda_train().
struct AlmeidaPinedaResult {
    AlmeidaPinedaNetwork net;           ///< The trained network.
    std::vector<double>  error_history; ///< Mean squared output error after each epoch.
};

/// @brief How the forward and adjoint relaxations are run.
struct AlmeidaPinedaSettleOptions {
    std::size_t max_iterations{1000};  ///< Upper limit on relaxation sweeps.
    double tolerance{1e-10};           ///< Stop when the largest change in a sweep is below this; 0 runs all sweeps.
    double relax{0.5};                 ///< Damping \f$\rho\in(0,1]\f$: \f$x \leftarrow (1-\rho)x + \rho f(Wx)\f$.
};

/// @brief An equilibrium of the forward dynamics.
struct AlmeidaPinedaState {
    std::vector<double> x;          ///< Activations (inputs clamped).
    std::vector<double> a;          ///< Net inputs \f$a_i=\sum_j W_{ij}x_j\f$ at the final state.
    std::size_t iterations{0};      ///< Sweeps performed.
    std::vector<double> residuals;  ///< Largest change in each sweep.
    bool converged{false};          ///< true if the last change was below the tolerance.
};

/// @brief One pattern's error and exact gradient, from the forward and adjoint fixed points.
struct AlmeidaPinedaGradient {
    double error{0.0};                    ///< \f$E=\tfrac12\sum_k (x_k-t_k)^2\f$ over the output units.
    std::vector<std::vector<double>> dW;  ///< \f$\partial E/\partial W_{ij}=y_i x_j\f$ (zero on clamped rows and the diagonal).
    AlmeidaPinedaState forward;           ///< The forward equilibrium.
    std::vector<double> y;                ///< The adjoint equilibrium.
    std::size_t adjoint_iterations{0};    ///< Adjoint sweeps performed.
    std::vector<double> adjoint_residuals;///< Largest change in each adjoint sweep.
    bool converged{false};                ///< Both relaxations converged.
};

/// @brief A network with no self-loops and \f$N(0, \sigma^2)\f$ weights into every free unit.
/// @param num_units Total units.
/// @param num_inputs Clamped input units (indices 0..num_inputs-1).
/// @param output_units Indices of the output units.
/// @param init_sd Standard deviation \f$\sigma\f$ of the initial weights.
/// @param seed Random seed.
/// @throws std::invalid_argument if num_inputs >= num_units or an output index is not a free unit.
[[nodiscard]] AlmeidaPinedaNetwork almeida_pineda_network(std::size_t num_units, std::size_t num_inputs,
                                                          std::vector<std::size_t> output_units,
                                                          double init_sd = 0.3, std::uint64_t seed = 0);

/// @brief Relaxes the free units to a fixed point \f$x_i=\tanh(\sum_j W_{ij}x_j)\f$ with the
///        inputs clamped, using synchronous damped sweeps.
[[nodiscard]] AlmeidaPinedaState almeida_pineda_settle(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                                       const AlmeidaPinedaSettleOptions& options = {});

/// @brief The output error and its gradient for one pattern: settle forward, then settle the
///        adjoint \f$y_i=f'(a_i)\bigl(\sum_j W_{ji}y_j+e_i\bigr)\f$ and take \f$\partial E/\partial W_{ij}=y_ix_j\f$.
///
/// At a fixed point the adjoint iteration matrix \f$DW^\top\f$ is similar to the transpose of the
/// forward Jacobian \f$DW\f$ (\f$D=\operatorname{diag}f'(a)\f$), so the adjoint converges whenever the
/// forward relaxation is locally stable, at the same asymptotic rate.
/// @param net The network.
/// @param input Values of the clamped units.
/// @param target Targets, one per output unit.
/// @param options Relaxation settings for both passes.
[[nodiscard]] AlmeidaPinedaGradient almeida_pineda_gradient(const AlmeidaPinedaNetwork& net,
                                                            const std::vector<double>& input,
                                                            const std::vector<double>& target,
                                                            const AlmeidaPinedaSettleOptions& options = {});

/// @brief Train a recurrent network with Almeida-Pineda backpropagation (full-batch gradient descent).
///
/// \param inputs        Input patterns (each of length \p num_inputs).
/// \param targets       Target values for the \p output_units (each row aligned with \p output_units).
/// \param num_units     Total units (inputs + hidden + outputs).
/// \param num_inputs    Number of clamped input units (indices 0..num_inputs-1).
/// \param output_units  Indices of the output units.
/// \param epochs        Training epochs.
/// \param learning_rate Gradient step size.
/// \param seed          RNG seed for weight init.
/// \param settle_iters  Relaxation sweeps for the forward/adjoint fixed points.
/// \param relax         Relaxation damping in (0,1].
AlmeidaPinedaResult almeida_pineda_train(const std::vector<std::vector<double>>& inputs,
                                         const std::vector<std::vector<double>>& targets,
                                         std::size_t num_units, std::size_t num_inputs,
                                         const std::vector<std::size_t>& output_units,
                                         std::size_t epochs = 500, double learning_rate = 0.1,
                                         std::uint64_t seed = 0, std::size_t settle_iters = 100,
                                         double relax = 0.5);

/// @brief Settle the network on an input and read off the output-unit activations.
std::vector<double> almeida_pineda_predict(const AlmeidaPinedaNetwork& net, const std::vector<double>& input,
                                           std::size_t settle_iters = 100, double relax = 0.5);

}  // namespace datamunge::algorithms
