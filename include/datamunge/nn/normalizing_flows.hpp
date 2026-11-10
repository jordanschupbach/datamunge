#pragma once
/// @file normalizing_flows.hpp
/// @brief Explicit banana observations and live Gaussian change-of-variables densities.
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/autograd.hpp"

namespace datamunge::nn::models {
/// @brief Internal finite matrix and two-coordinate validation for flow operations.
namespace flow_detail {
/// @brief Require a nonempty finite floating CPU matrix, optionally of fixed width.
/// @param x Float32/64 CPU[N,D], N,D>0, arbitrary strides.
/// @param width Required final width, or zero(default) to accept any positive D.
/// @throws std::invalid_argument For invalid storage, geometry, dtype or nonfinite values.
inline void Matrix(const Tensor& x, std::int64_t width=0) {
  if(!x.defined() || x.device().backend!=compute::Backend::CPU || x.dim()!=2 ||
     x.size(0)<=0 || x.size(1)<=0 || (width && x.size(1)!=width))
    throw std::invalid_argument("Flow operations require a positive floating CPU matrix");
  gan_detail::Finite(x);
}
} // namespace flow_detail

/// @brief Transform explicit standard-normal coordinates into detached banana observations.
///
/// For finite Float32/64 CPU standard[N,2], compute x0=u and
/// x1=v+curvature*(u*u-1) in Double, then cast once to dtype(default Float32).
/// Curvature is any finite real, including zero/negative values. This invertible
/// triangular map has determinant one. The caller supplies the complete noise
/// bank; this function does not sample, split rows, fit or retain RNG state.
/// Arbitrary strides accepted; result owns stopped contiguous storage and never
/// mutates/retains the input graph. O(N) work/storage; independent calls share no
/// mutable state. Double or final Float32 overflow is rejected without clipping.
/// @param standard Finite floating CPU[N,2] explicit normal-coordinate bank.
/// @param curvature Finite quadratic bend coefficient, default .5.
/// @param dtype Float32(default) or Float64 final observation dtype.
/// @return Owning detached finite CPU[N,2] banana coordinates.
/// @throws std::invalid_argument For invalid inputs/options or nonfinite computed coordinates.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto u=n::from_vector(std::vector<double>{0,0,2,1},{2,2});
/// auto x=n::models::BananaPoints(u);
/// // x is Float32 [[0,-.5],[2,2.5]]. No gradient is retained.
/// @endcode
/// @par References
/// - Original compact normalizing-flow teaching protocol: x0=u, x1=v+.5*(u*u-1),
/// with explicit Box-Muller Double coordinates; not an external dataset.
/// @see BananaLogProbability(), GaussianNoise(), NiceDensity()
inline Tensor BananaPoints(const Tensor& standard,double curvature=.5,DType dtype=DType::Float32) {
  flow_detail::Matrix(standard,2);
  if(!std::isfinite(curvature) || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("BananaPoints needs finite curvature and a floating output dtype");
  NoGradGuard guard;
  const auto v=cast(standard,DType::Float64);
  const auto u=slice_axis(v,1,0,1),w=slice_axis(v,1,1,2);
  auto out=cast(concat({u,w+curvature*(u*u-1.)},1),dtype).contiguous();
  gan_detail::Finite(out);
  return out;
}

/// @brief Evaluate the analytic unit-determinant banana log density in Double.
///
/// Finite Float32/64 CPU points[N,2] and finite curvature(default .5). Invert
/// u=x0, v=x1-curvature*(x0*x0-1), then return -log(2*pi)-.5*(u*u+v*v).
/// Uses the full Double log constant, independently of the model's optional
/// legacy rounded constant. Live first/higher input paths survive conversion
/// to Double and parent collection. Arbitrary strides, no mutation, sampling,
/// fitting or global state. O(N) work/storage; nonfinite results are rejected.
/// @param points Finite floating CPU[N,2] observations in coordinate units.
/// @param curvature Finite quadratic bend coefficient, default .5.
/// @return Owning live Float64 CPU[N] analytic log probabilities in nats.
/// @throws std::invalid_argument For invalid inputs/curvature or nonfinite results.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto x=dm::nn::from_vector(std::vector<double>{0,-.5},{1,2});
/// auto logp=dm::nn::models::BananaLogProbability(x);
/// // logp[0]=-log(2*pi), approximately -1.8378770664.
/// @endcode
/// @par References
/// - Original compact flow protocol's analytic triangular banana map with unit determinant.
/// @see BananaPoints(), StandardNormalLogProbability()
inline Tensor BananaLogProbability(const Tensor& points,double curvature=.5) {
  flow_detail::Matrix(points,2);
  if(!std::isfinite(curvature))throw std::invalid_argument("Banana curvature must be finite");
  const auto x=cast(points,DType::Float64);
  const auto u=reshape(slice_axis(x,1,0,1),{-1});
  const auto v=reshape(slice_axis(x,1,1,2),{-1})-curvature*(u*u-1.);
  auto out=-std::log(2*std::acos(-1.))-.5*(u*u+v*v);
  gan_detail::Finite(out);return out;
}

/// @brief Evaluate a factorized standard Gaussian log density without sampling.
///
/// Finite floating CPU latent[N,D], positive axes and arbitrary strides.
/// Return -.5*sum(z*z,axis1)-.5*D*log(2*pi), in the input Float32/64 dtype.
/// rounded_constant=false(default) uses a Double log constant before scalar
/// promotion; true first rounds log(2*pi) to Float32, even for Double inputs,
/// matching the original compact flow source. First/higher paths remain live;
/// no mutation, fitting, sampling or RNG. Owns its result/graph after input
/// collection. O(ND) work and O(ND+N) transient storage; finite overflow rejected.
/// @param latent Finite Float32/64 CPU[N,D] Gaussian coordinates.
/// @param rounded_constant False(default); true preserves the source Float32 log constant.
/// @return Owning live matching floating CPU[N] log probabilities in nats.
/// @throws std::invalid_argument For invalid inputs or nonfinite results.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto logp=dm::nn::models::StandardNormalLogProbability(dm::nn::zeros({3,2}));
/// // Three identical values approximately -1.837877, one per row.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE: Non-linear Independent Components
/// Estimation, https://arxiv.org/abs/1410.8516 . This compact model uses a Gaussian base.
/// @see NormalFlowDensity(), BananaLogProbability()
inline Tensor StandardNormalLogProbability(const Tensor& latent,bool rounded_constant=false) {
  flow_detail::Matrix(latent);
  double constant=std::log(2*std::acos(-1.));
  if(rounded_constant)constant=double(float(constant));
  auto out=-(.5*sum(square(latent),{1})+.5*double(latent.size(1))*constant);
  gan_detail::Finite(out);return out;
}

/// @brief Own live base/density fields and a mean negative log likelihood.
///
/// Three flat Tensor handles retain their applicable input/parameter graphs
/// and storage after parent collection. No model/store, optimizer or RNG is
/// retained separately; applicable learned leaves belong to the graph.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto z=dm::nn::zeros({2,2});
/// auto out=dm::nn::models::NormalFlowDensity(z,dm::nn::full({},0));
/// // base/log_probability[2], scalar loss approximately 1.837877 nats/row.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see NormalFlowDensity(), NiceDensity()
struct NormalFlowDensityOutput {
  Tensor base_log_probability; ///< Owning live matching Float32/64[N] standard Gaussian log density.
  Tensor log_probability;      ///< Owning live matching Float32/64[N] base+forward_logdet.
  Tensor loss;                 ///< Owning live matching floating scalar negative row mean, nats/row.
};

/// @brief Apply the Gaussian change-of-variables formula with an explicit forward log determinant.
///
/// Finite matching floating CPU latent[N,D] and forward_logdet scalar or[N].
/// Determinants must describe x->z, not its inverse. A scalar expresses a global
/// determinant and broadcasts to all rows; [1] is accepted only when N=1.
/// Return the Gaussian base log density, base+forward_logdet and its negative
/// mean over rows. Does not prove invertibility or derive a Jacobian; the caller
/// supplies a correct analytic/numerical determinant. First/higher applicable
/// latent/logdet/parameter paths remain live. No hidden no-grad, mutation,
/// optimizer, sampling or RNG. Arbitrary strides; O(ND) work/transient storage;
/// owning flat fields survive parent collection. Nonfinite outputs rejected.
/// @param latent Finite Float32/64 CPU[N,D], N,D>0, Gaussian coordinates.
/// @param forward_logdet Matching finite CPU scalar or[N] log absolute determinant in nats.
/// @param rounded_constant False(default); true uses the source Float32 log(2*pi) constant.
/// @return Owning NormalFlowDensityOutput with base/logp[N] and scalar mean NLL.
/// @throws std::invalid_argument For invalid inputs/determinant geometry or nonfinite results.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto z=n::zeros({2,2},n::DType::Float64);
/// auto out=n::models::NormalFlowDensity(z,n::full({},.2,n::DType::Float64));
/// // loss=log(2*pi)-.2, approximately 1.6378770664 nats/row.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see StandardNormalLogProbability(), NormalFlowDensityOutput, NiceInverse()
inline NormalFlowDensityOutput NormalFlowDensity(
    const Tensor& latent,const Tensor& forward_logdet,bool rounded_constant=false) {
  flow_detail::Matrix(latent);gan_detail::Finite(forward_logdet);
  if(forward_logdet.device().backend!=compute::Backend::CPU ||
     forward_logdet.dtype()!=latent.dtype() ||
     !(forward_logdet.dim()==0 || (forward_logdet.dim()==1 && forward_logdet.size(0)==latent.size(0))))
    throw std::invalid_argument("Flow determinant must be a matching scalar or one value per row");
  auto base=StandardNormalLogProbability(latent,rounded_constant);
  auto logp=base+forward_logdet;auto loss=-mean(logp,{0});
  gan_detail::Finite(logp);gan_detail::Finite(loss);
  return {base,logp,loss};
}
} // namespace datamunge::nn::models
