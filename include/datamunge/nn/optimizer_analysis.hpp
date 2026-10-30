#pragma once
/// @file optimizer_analysis.hpp
/// @brief Inspectable SGD/momentum updates and diagonal-quadratic stability experiments.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/nn/optim.hpp>
namespace datamunge::nn {
/// @brief Replays prescribed gradient/rate streams through the real Float64 tensor optimizer.
/// Supports SGD, Momentum and Nesterov, with fresh zero velocity. No gradient is recomputed
/// from the changing weights. All inputs are borrowed and unchanged; output owns numeric
/// snapshots, with no shared mutable state. Work/storage O(steps*coordinates).
/// @param initial Nonempty finite initial parameter vector.
/// @param gradients Finite steps-by-coordinates prescribed gradients; zero steps allowed.
/// @param rates One finite nonnegative rate, or one per step.
/// @param config Valid optimizer configuration; only the three stated kinds are accepted.
/// @return step/coordinate/rate/gradient/before/after/velocity table, zero-based coordinates,
/// one-based updates; velocity is zero for SGD and the updated gradient accumulator otherwise.
/// @throws std::invalid_argument Unsupported kind, invalid configuration, shape or finite domain.
/// @throws std::overflow_error A finite input stream produces a nonfinite parameter/velocity.
/// @par Example
/// @code
/// const auto
/// t=dm::nn::optimizer_gradient_trace({0},dm::linalg::DenseMatrix<double>(1,1,{1}),{.1});
/// std::print("{:.1f}\n",t.double_at("after",0)); // -0.1
/// @endcode
/// @par References
/// - TensorFlow ApplyMomentum convention:
/// https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/apply-momentum
/// @see optimizer_quadratic_trace(), Optimizer
[[nodiscard]] dstruct::DataFrame optimizer_gradient_trace(
    const std::vector<double>&         initial,
    const linalg::DenseMatrix<double>& gradients,
    const std::vector<double>&         rates,
    OptimizerConfig                    config = {});
/// @brief Runs tensor-autograd optimization of a diagonal quadratic, retaining every state.
/// Objective f(w)=sum(curvature_j*w_j^2)/2 has its known minimum at zero. This is a
/// deterministic full-gradient experiment, not stochastic minibatch training. Fresh
/// Float64 leaves/optimizer state are owned internally. Work/output storage O(steps*p).
/// @param curvature Positive finite diagonal Hessian entries, length p>=1.
/// @param initial Finite p-vector, unchanged.
/// @param steps Number of optimizer updates; zero returns the initial state only.
/// @param rate Finite nonnegative constant learning rate.
/// @param config Valid SGD/Momentum/Nesterov configuration.
/// @return step/coordinate/weight/gradient/velocity/loss table, including step zero;
/// loss is the same whole-objective value on each coordinate row of a step.
/// @throws std::invalid_argument Invalid kind/configuration, shapes or finite domains.
/// @throws std::overflow_error An update or objective becomes nonfinite.
/// @par Example
/// @code
/// const auto t=dm::nn::optimizer_quadratic_trace({2},{1},1,.1);
/// std::print("{:.1f}\n",t.double_at("weight",1)); // 0.8
/// @endcode
/// @par References
/// - B. T. Polyak (1964), Some methods of speeding up the convergence of iteration methods.
///   https://doi.org/10.1016/0041-5553(64)90137-5
/// @see optimizer_mode_radius(), optimizer_gradient_trace()
[[nodiscard]] dstruct::DataFrame optimizer_quadratic_trace(
    const std::vector<double>& curvature,
    const std::vector<double>& initial,
    std::size_t                steps,
    double                     rate,
    OptimizerConfig            config = {});
/// @brief Spectral radius of one scalar quadratic mode under a constant optimizer rate.
/// Returns |1-a| for SGD, or the largest root modulus of the two-state momentum map.
/// Here a=learning_rate*curvature. Radius below one implies asymptotic decay for this
/// fixed quadratic mode; radius one is a boundary, not a convergence guarantee. A
/// repeated-root transient or a poorly chosen initial state is not described by radius
/// alone. No assertion is made for nonquadratic/stochastic objectives. Work/storage O(1).
/// @param kind SGD, Momentum or Nesterov (the TensorFlow reparameterized convention).
/// @param rate_curvature Finite nonnegative dimensionless a.
/// @param momentum Finite velocity decay in [0,1), default .9; ignored by SGD's formula.
/// @return Nonnegative root modulus.
/// @throws std::invalid_argument Invalid kind, a or momentum.
/// @par Example
/// @code
/// std::print("{:.2f}\n",dm::nn::optimizer_mode_radius(dm::nn::OptimizerKind::Sgd,.1)); // 0.90
/// @endcode
/// @par References
/// - Polyak (1964). https://doi.org/10.1016/0041-5553(64)90137-5
/// - https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/apply-momentum
/// @see optimizer_quadratic_trace()
[[nodiscard]] double
optimizer_mode_radius(OptimizerKind kind, double rate_curvature, double momentum = .9);
} // namespace datamunge::nn
