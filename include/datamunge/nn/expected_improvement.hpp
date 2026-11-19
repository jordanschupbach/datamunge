#pragma once

/// @file expected_improvement.hpp
/// @brief Gaussian process acquisition on a finite set of candidates.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Rank finite candidates by Gaussian process expected improvement for minimization.
/// Uses an isotropic unit-variance RBF kernel and standardizes the observed losses.
/// Scores are in standardized loss units. Observed candidates receive -1.
/// @param grid Candidate coordinates [candidates,dimensions].
/// @param seen Distinct observed row indices, at least one.
/// @param losses Finite loss observations in seen order.
/// @param length_scale Positive RBF length scale, default 1.6.
/// @return Expected improvement for each candidate in grid order.
/// @throws std::invalid_argument For invalid shapes, indices, losses or scale.
/// @par Example
/// @code
/// auto scores=dm::nn::gaussian_expected_improvement(dm::nn::arange(0,4).reshape({4,1}),{0},{1});
/// @endcode
/// @par References
/// - Jones et al. (1998), Efficient Global Optimization of Expensive Black-Box Functions.
///   doi:10.1023/A:1008306431147
/// @see cholesky(), triangular_solve()
inline std::vector<double> gaussian_expected_improvement(const Tensor& grid,
    const std::vector<std::int64_t>& seen,const std::vector<double>& losses,double length_scale=1.6) {
  if (!grid.defined() || grid.dim()!=2 || grid.size(0)<1 || grid.size(1)<1 ||
      seen.empty() || seen.size()!=losses.size() || !std::isfinite(length_scale) || length_scale<=0)
    throw std::invalid_argument("Invalid expected improvement inputs");
  std::vector<bool> observed(grid.size(0));
  for (std::size_t i=0;i<seen.size();++i) {
    if (seen[i]<0 || seen[i]>=grid.size(0) || observed[seen[i]] || !std::isfinite(losses[i]))
      throw std::invalid_argument("Invalid observed candidate or loss");
    observed[seen[i]]=true;
  }
  NoGradGuard guard;
  const auto all=grid.to(DType::Float64);
  const auto fitted=take_rows(all,seen);
  /// Compute the unit-variance RBF matrix between row sets.
  const auto kernel=[&](const Tensor& a,const Tensor& b) {
    return exp(-.5*sum(square((expand_dims(a,1)-expand_dims(b,0))/length_scale),{2}));
  };
  const auto values=from_vector(losses,{static_cast<std::int64_t>(losses.size()),1});
  const auto center=mean(values);
  const auto scale=std::max(1e-4,sqrt(mean(square(values-center))).item());
  const auto chol=cholesky(kernel(fitted,fitted)+1e-6*eye(seen.size(),DType::Float64));
  const auto k=triangular_solve(chol,kernel(fitted,all));
  const auto z=triangular_solve(chol,(values-center)/scale);
  const auto means=matmul(transpose(k),z).to_vector<double>();
  const auto variance=(1-sum(square(k),{0})).to_vector<double>();
  const auto best=(*std::min_element(losses.begin(),losses.end())-center.item())/scale;
  std::vector<double> result(means.size(),-1);
  for (std::size_t i=0;i<result.size();++i) {
    if (observed[i]) continue;
    const auto sd=std::sqrt(std::max(1e-15,variance[i])), delta=best-means[i], ratio=delta/sd;
    result[i]=.5*delta*std::erfc(-ratio/std::numbers::sqrt2)+
        sd*std::exp(-.5*ratio*ratio)/std::sqrt(2*std::numbers::pi);
  }
  return result;
}

}  // namespace datamunge::nn
