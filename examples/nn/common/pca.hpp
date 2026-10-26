// datamunge::nn port of tflow_examples src/common/ml/pca.h.
// Principal components and kernel centering with nn ops.
// PCA (float64 in the examples) subtracts the training mean and computes a thin
// SVD. Variances divide by N-1. Components are columns; their signs are arbitrary.
// Whiten only components with positive variance. Kernel cross-centering uses
// training column/grand means and each query row's mean over training points;
// never center a validation/test kernel using its own column means.
#pragma once

#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace ml {
namespace nn = datamunge::nn;

struct PrincipalComponents {
  nn::Tensor mean, components, variances, scores;
  PrincipalComponents(const nn::Tensor& x, int retained) {
    const auto dimensions = x.size(1);
    if (retained <= 0 || retained > dimensions) throw std::invalid_argument("invalid retained PCA dimension");
    mean                = nn::mean(x, {0}, true);
    const auto svd      = nn::svd(x - mean);
    components          = nn::slice(svd.v, {0, 0}, {dimensions, retained});
    const double n      = double(x.size(0));
    variances           = nn::square(nn::slice(svd.s, {0}, {retained})) / (n - 1);
    scores              = Transform(x);
  }
  nn::Tensor Transform(const nn::Tensor& x) const { return nn::matmul(x - mean, components); }
  nn::Tensor Reconstruct(const nn::Tensor& projection) const {
    return mean + nn::matmul(projection, components, false, true);
  }
};

inline nn::Tensor CenterKernel(const nn::Tensor& kernel) {
  return kernel - nn::mean(kernel, {0}, true) - nn::mean(kernel, {1}, true) + nn::mean(kernel, {0, 1});
}

inline nn::Tensor CenterCrossKernel(const nn::Tensor& cross, const nn::Tensor& training_kernel) {
  return cross - nn::mean(training_kernel, {0}, true) - nn::mean(cross, {1}, true) + nn::mean(training_kernel, {0, 1});
}

}  // namespace ml
