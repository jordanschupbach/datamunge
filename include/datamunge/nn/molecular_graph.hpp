/// @file
/// @brief Radial and angular features and compact triatomic graph regressors.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Own geometric features for the six directed edges of three atoms.
///
/// Edge order is sender first, then receiver, with self-edges excluded.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto xyz = n::tensor({0.,0.,0., 1.,0.,0., 0.,1.,0.}, {1,3,3});
/// const auto basis = n::triatomic_graph_basis(xyz);
/// // basis.radial.shape() == {1,6,6}.
/// @endcode
/// @see triatomic_graph_basis()
struct TriatomicGraphBasis {
  Tensor gaussian; ///< [B,6,6] Gaussian radial descriptors.
  Tensor radial; ///< [B,6,6] cutoff-weighted sine radial descriptors.
  Tensor angular; ///< [B,6,3] spherical Bessel and Legendre descriptors.
  Tensor cutoff; ///< [B,6,1] cosine cutoff multipliers.
};

/// @brief Compute distances and directional bases on a triatomic graph.
///
/// Uses six Gaussian centers starting at 0.5 and spaced by 0.55, six
/// sine radial frequencies, three angular terms, and cutoff radius 4.
/// The angular term for edge j→i uses the unique non-backtracking
/// incoming edge k→j. Coordinates and output share dtype. The basis
/// preserves coordinate gradients away from coincident positions.
///
/// @param xyz Finite Float32/64 [B,3,3] coordinates with distinct atoms in each row.
/// @return Directed-edge bases and cutoff multipliers.
/// @throws std::invalid_argument For incompatible coordinates or zero distances.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto xyz = n::tensor({0.,0.,0., 1.,0.,0., 0.,1.,0.}, {1,3,3});
/// const auto basis = n::triatomic_graph_basis(xyz);
/// // basis.angular.shape() == {1,6,3}.
/// @endcode
/// @par References
/// - Schütt et al. (2017), SchNet, https://arxiv.org/abs/1706.08566
/// - Klicpera, Groß, and Günnemann (2020), DimeNet, https://arxiv.org/abs/2003.03123
/// @see triatomic_energy()
inline TriatomicGraphBasis triatomic_graph_basis(const Tensor &xyz) {
  if (xyz.dim() != 3 || xyz.size(0) <= 0 || xyz.size(1) != 3 || xyz.size(2) != 3 ||
      (xyz.dtype() != DType::Float32 && xyz.dtype() != DType::Float64) ||
      !all(is_finite(xyz)).item<bool>())
    throw std::invalid_argument("triatomic_graph_basis: expected finite [B,3,3] coordinates");
  const auto sender = from_vector(std::vector<int>{0, 0, 1, 1, 2, 2});
  const auto receiver = from_vector(std::vector<int>{1, 2, 0, 2, 0, 1});
  const auto incoming = from_vector(std::vector<int>{4, 2, 5, 0, 3, 1});
  const auto difference = gather(xyz, receiver, 1) - gather(xyz, sender, 1);
  const auto distance = sqrt(sum(square(difference), {2}, true));
  if (!all(greater(distance, 0.)).item<bool>() || !all(is_finite(distance)).item<bool>())
    throw std::invalid_argument("triatomic_graph_basis: expected distinct atoms with finite distances");

  constexpr double pi = std::numbers::pi;
  const auto numbers = arange(0, 6, 1, xyz.dtype());
  const auto cutoff = where(less(distance, 4.), .5 * (1. + cos(pi / 4. * distance)), 0.);
  const auto gaussian = exp(-5. * square(distance - (.5 + .55 * numbers)));
  const auto radial = cutoff * sin((pi / 4.) * distance * (numbers + 1.)) / distance;

  const auto from = gather(distance, incoming, 1);
  const auto to_k = -gather(difference, incoming, 1);
  const auto angle = sum(difference * to_k, {2}, true) / (distance * from);
  const auto arg0 = (pi / 4.) * from;
  const auto arg1 = (4.493409457909064 / 4.) * from;
  const auto arg2 = (5.76345919689455 / 4.) * from;
  const auto j1 = [](const Tensor &x) { return (sin(x) / x - cos(x)) / x; };
  const auto angular = gather(cutoff, incoming, 1) * concat({
      sin(arg0) / arg0, std::sqrt(3.) * j1(arg1) * angle,
      std::sqrt(5.) * (3. * j1(arg2) / arg2 - sin(arg2) / arg2) *
          (.5 * (3. * square(angle) - 1.))}, 2);
  return {gaussian, radial, angular, cutoff};
}

/// @brief Predict a scalar from one SchNet-style or DimeNet-style interaction block.
///
/// Atom zero uses a distinct species and the other two atoms share a species.
/// The radial branch makes continuous filters for neighbor states. The
/// directional branch mixes non-backtracking triplet descriptors with
/// incoming edge states. A shared head predicts per-atom contributions,
/// which are summed. The store owns the learned parameters.
///
/// @param store Parameter store owning the projection weights.
/// @param name Layer path within the store.
/// @param basis Triatomic edge features as produced by triatomic_graph_basis().
/// @param width Positive channel count.
/// @param directional Use the directional branch if true, radial if false.
/// @return Live [B,1] predictions in the store dtype.
/// @throws std::invalid_argument For incompatible basis shapes or widths.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto xyz = n::tensor({0.,0.,0., 1.,0.,0., 0.,1.,0.}, {1,3,3});
/// const auto y = n::triatomic_energy(p, "model", n::triatomic_graph_basis(xyz), 8, true);
/// // y.shape() == {1,1}.
/// @endcode
/// @par References
/// - Schütt et al. (2017), SchNet, https://arxiv.org/abs/1706.08566
/// - Klicpera, Groß, and Günnemann (2020), DimeNet, https://arxiv.org/abs/2003.03123
/// @see triatomic_graph_basis()
inline Tensor triatomic_energy(ParameterStore &store, const std::string &name,
                               const TriatomicGraphBasis &basis, std::int64_t width,
                               bool directional) {
  const auto &descriptors = directional ? basis.radial : basis.gaussian;
  if (descriptors.dim() != 3 || descriptors.size(0) <= 0 || descriptors.size(1) != 6 ||
      descriptors.size(2) != 6 || basis.cutoff.shape() != Shape{descriptors.size(0), 6, 1} ||
      (directional && basis.angular.shape() != Shape{descriptors.size(0), 6, 3}) ||
      width <= 0)
    throw std::invalid_argument("triatomic_energy: incompatible basis or width");
  const auto dtype = store.dtype();
  const auto sender = from_vector(std::vector<int>{0, 0, 1, 1, 2, 2});
  const auto receiver = from_vector(std::vector<int>{1, 2, 0, 2, 0, 1});
  const auto incidence = transpose(one_hot(receiver, 3, 1., 0., dtype), {1, 0});
  const auto species = tensor({1., 0., 0.}, {3, 1}, dtype);
  const auto cutoff = basis.cutoff.to(dtype);
  const auto radial = dense(store, name + "/radial", descriptors.to(dtype), width, Activation::Tanh);
  Tensor atoms;
  if (directional) {
    const auto incoming = from_vector(std::vector<int>{4, 2, 5, 0, 3, 1});
    const auto roles = concat({gather(species, sender), gather(species, receiver)}, 1);
    const auto type = dense(store, name + "/type", roles, width);
    const auto edge = tanh(radial + type);
    const auto angular = dense(store, name + "/angular", basis.angular.to(dtype), width,
                               Activation::Tanh);
    const auto mixed = gather(edge * radial, incoming, 1) * angular;
    const auto interaction = dense(store, name + "/interaction", mixed, width, Activation::Tanh);
    atoms = matmul(incidence, (edge + interaction) * cutoff);
  } else {
    const auto type = dense(store, name + "/type", species, width, Activation::Tanh);
    const auto filter = dense(store, name + "/filter", radial, width);
    const auto messages = filter * gather(type, sender) * cutoff;
    const auto interaction =
        dense(store, name + "/interaction", matmul(incidence, messages), width, Activation::Tanh);
    atoms = type + interaction;
  }
  return sum(dense(store, name + "/head", atoms, 1), {1});
}

} // namespace datamunge::nn
