#pragma once
/// @file rbf.hpp
/// @brief Native differentiable Gaussian radial-basis features for explicit tensor model training.
#include <cstddef>
#include <datamunge/nn/tensor.hpp>
/// @brief Canonical differentiable tensor computations.
namespace datamunge::nn {
/// @brief Build Gaussian radial-basis activations entirely with native tensor operations.
/// For input[N,D] and centers[K,D], returns exp(-0.5*sum_d(((input-center)/width)^2)) with shape[N,K]. Requires finite CPU Float32/Float64 tensors with matching dtype, positive D and a finite positive width representable in that dtype. Empty N or K yields an empty feature matrix. Native broadcast/subtraction/square/reduction/exp operations preserve gradients to both input and centers; width is a fixed scalar, not a learnable leaf. Checks actual normalized squared distances for overflow rather than passing infinite intermediates into backward arithmetic. O(N*K*D) tensor work and intermediate storage, independently capped. Arguments unchanged; the returned tensor owns its allocation and retained autograd references. Use explicit Tensor leaves, objective and optimizer loop to fit the output weights/bias; this function does not train a model.
/// @param input Finite CPU floating observations, shape[N,D].
/// @param centers Finite matching CPU floating centers, shape[K,D].
/// @param width Positive finite fixed Gaussian sigma.
/// @param max_intermediate_elements Complete N*K*D intermediate-element policy, default1000000; zero disables.
/// @return Differentiable Gaussian activation tensor[N,K].
/// @throws std::invalid_argument Undefined/invalid shapes, device/dtype, nonfinite values or width.
/// @throws std::length_error Complete intermediate policy exceeded.
/// @throws std::overflow_error Shape arithmetic or actual normalized squared distances are unrepresentable.
/// @par Example
/// @code
/// auto x=dm::nn::from_vector(std::vector<double>{0,1},{2,1});
/// auto c=dm::nn::from_vector(std::vector<double>{0},{1,1});
/// auto phi=dm::nn::rbf_features(x,c,1);
/// std::print("{} {}\n",phi.shape(),phi.at({0,0})); // [2, 1] 1
/// @endcode
/// @par References
/// - Broomhead and Lowe (1988), Multivariable Functional Interpolation and Adaptive Networks, https://apps.dtic.mil/sti/citations/ADA196234
/// @see matmul(), mse(), Optimizer
Tensor rbf_features(
    const Tensor& input,
    const Tensor& centers,
    double        width,
    std::size_t   max_intermediate_elements = 1000000);
} // namespace datamunge::nn
