#pragma once
/// @file cached_spectral_dense.hpp
/// @brief Dense spectral maps with explicit cached columns and source-rounded initialization.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/spectral_cache.hpp"

namespace datamunge::nn {
/// @brief Apply a dense map using cached singular columns and a live signed weight denominator.
///
/// y=activate(x*W/max(sum(detach(v)*(W*detach(u))),double(1e-6f))+b).
/// Finite matching Float32/64 CPU x[N,in], positive axes, distinct matching stores,
/// nonempty name and positive units required. Creates name/W[in,units],b[units]
/// in parameters, and name/u[units,1],v[in,1] in state, in that order. W's uniform
/// limit is sqrt(6.f/float(in+units)); biases are zero; initial state columns are
/// 1.f/sqrt(float(length)). Constants preserve Float32 rounding in Float64 too.
/// All shape/product/input checks precede leaves; incompatible existing leaves,
/// invalid activation or nonfinite computation may fail later after earlier leaves
/// were created. Strides and applicable first/higher x/W/b paths stay live, including
/// sigma; both column paths stop. No initial power iteration, state advance, RNG
/// beyond missing leaf creation, fit, optimizer or implicit freeze is hidden.
/// Existing state is unchanged; exclude its store from optimizers despite its
/// ParameterStore flags. Inputs unchanged, output owns lifetime. Work O(N*in*units),
/// activations O(N*units) plus validation. Synchronize shared stores. The cached
/// estimate need not equal the exact operator norm; negative sigma uses the floor.
/// @param parameters Mutable floating weight/bias owner.
/// @param state Distinct matching cache owner, unchanged after initialization.
/// @param name Nonempty prefix for W,b,u,v, without an added layer suffix.
/// @param x Finite signed matching floating CPU[N,in] matrix with positive axes.
/// @param units Positive output width, with representable sums/products.
/// @param activation Output activation, default Activation::None (raw affine output).
/// @return Owning live floating[N,units] dense output.
/// @throws std::invalid_argument For invalid inputs/options/stores/geometry, incompatible leaves,
/// or nonfinite weights, bias, cache or computed output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1),s(0);
/// n::cached_spectral_dense(p,s,"layer",n::zeros({2,3}),4,n::Activation::Tanh);
/// n::advance_cached_spectral_state(p,s,16); // explicit warm-up outside the layer.
/// auto y=n::cached_spectral_dense(p,s,"layer",n::ones({2,3}),4,n::Activation::Tanh);
/// // y is[2,4]; subsequent scoring does not change existing u/v.
/// @endcode
/// @par References
/// - Miyato et al. (2018), Spectral Normalization for Generative Adversarial Networks,
///   https://arxiv.org/abs/1802.05957 . This explicit cached-column variant adds a
///   signed denominator floor and uses source-rounded initialization constants.
/// @see cached_spectral_kernel(), advance_cached_spectral_state(), spectral_norm_dense()
inline Tensor cached_spectral_dense(
    ParameterStore&    parameters,
    ParameterStore&    state,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       units,
    Activation         activation = Activation::None) {
  if (&parameters == &state || parameters.dtype() != state.dtype() || name.empty())
    throw std::invalid_argument(
        "Cached spectral dense needs distinct matching stores and a nonempty name");
  spectral_cache_detail::Finite(x);
  if (x.dim() != 2 || x.dtype() != parameters.dtype() || units <= 0)
    throw std::invalid_argument(
        "Cached spectral dense needs matching matrix input and positive units");
  const auto in = x.size(1), rows = x.size(0), limit = std::numeric_limits<std::int64_t>::max();
  if (in > limit - units || in > limit / units || rows > limit / units)
    throw std::invalid_argument("Cached spectral dense dimensions/products overflow");
  auto w =
      parameters.get(name + "/W", {in, units}, init::uniform(std::sqrt(6.f / float(in + units))));
  auto b = parameters.get(name + "/b", {units}, init::zeros());
  auto u = state.get(name + "/u", {units, 1}, init::constant(1.f / std::sqrt(float(units))));
  auto v = state.get(name + "/v", {in, 1}, init::constant(1.f / std::sqrt(float(in))));
  spectral_cache_detail::Finite(b);
  auto kernel = cached_spectral_kernel(w, u, v);
  auto y      = activate(matmul(x, kernel.weight) + b, activation);
  spectral_cache_detail::Finite(y);
  return y;
}
} // namespace datamunge::nn
