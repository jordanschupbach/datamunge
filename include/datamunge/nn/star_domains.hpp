#pragma once
/// @file star_domains.hpp
/// @brief Native three-domain image transforms and disjoint per-class selection with unused IDs.
#include <array>
#include <set>

#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/training.hpp"
namespace datamunge::nn::models {
/// @brief Transform each square grayscale row into its supplied original/inverted/rotated domain.
///
/// Finite signed Float32/64 CPU original[N,H,H,1], all axes positive, strides
/// allowed, and aligned Int32/64 CPU domain[N] in0..2. Domain0 is unchanged,
/// 1 is1-original,2 is90-degree rotation out(y,x)=original(H-1-x,y), with
/// no inversion on rotation. Validates every domain before one-hot/transform.
/// No unit clipping, fitting, RNG, parameter or optimizer. Applicable first/higher
/// paths remain live through selected rows; discrete domain IDs have no gradient.
/// Owning output retains storage/graph after inputs are collected; inputs unchanged.
/// O(N*H*H) native work/storage; synchronize external mutation. Known transforms
/// are diagnostic/data preparation only, never a StarGan objective target.
/// @param original Finite signed floating CPU[N,H,H,1], positive square grayscale geometry.
/// @param domain Integer CPU[N] domain IDs0..2, one per image.
/// @return Owning live same-shape/dtype transformed image.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto x=n::tensor({0.,.25,.5,.75},{1,2,2,1},n::DType::Float64);
/// auto rotated=n::models::StarDomainImages(x,n::full({1},2,n::DType::Int64));
/// // Rotated pixels are(.5,0,.75,.25); no intensity inversion is applied.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see SplitStarDomains(), StarGan()
inline Tensor StarDomainImages(const Tensor& original, const Tensor& domain) {
  gan_detail::Finite(original);
  if (original.dim() != 4 || original.size(1) != original.size(2) || original.size(3) != 1)
    throw std::invalid_argument("Star domains need positive square grayscale NHWC images");
  gan_detail::Labels(domain, original.size(0), 3);
  auto code    = reshape(domain, {original.size(0), 1, 1, 1});
  auto rotated = transpose(reverse(original, {1}), {0, 2, 1, 3});
  auto out     = where(equal(code, 0.), original, where(equal(code, 1.), 1. - original, rotated));
  gan_detail::Finite(out);
  return out;
}
/// @brief Own selected three-domain rows, exact originals, metadata and explicitly unused
/// identities.
///
/// Six flat fields retain storage and applicable image gather/transform graphs
/// beyond parent collection. domain/digit are Int32; source/unused preserve input
/// ID dtype. Selection order is the original global row order. No fitted state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto labels=n::cast(n::arange(0,30,1,n::DType::Float64)/3.,n::DType::Int32);
/// auto
/// pool=n::models::SplitStarDomains(n::ones({30,28,28,1}),labels,n::arange(0,30,1,n::DType::Int64),1);
/// // pool.image has30 rows; each class/domain has1 row; unused is empty.
/// @endcode
/// @see SplitStarDomains(), StarDomainImages()
struct StarDomains {
  Tensor image;    ///< Owning live selected/transformed[Nselected,28,28,1].
  Tensor original; ///< Owning exact selected untransformed images, diagnostic only.
  Tensor domain;   ///< Owning Int32[Nselected] observed-order domain IDs0..2.
  Tensor digit;    ///< Owning Int32[Nselected] digit metadata0..9, never a model input.
  Tensor source;   ///< Owning original source IDs[Nselected], input ID dtype preserved.
  Tensor unused;   ///< Owning IDs of unused rows[N-Nselected], input ID dtype preserved.
};
/// @brief Select disjoint original/inverted/rotated domains by each class occurrence order.
///
/// Finite unit Float32/64 CPU images[N,28,28,1], N>0 divisible by10;
/// Int32/64 CPU labels[N] in0..9 exactly balanced N/10 each; aligned Int32/64
/// nonnegative unique source_ids[N]. per_class positive and at most floor(N/30).
/// Take first per_class observations of each class for domain0, next per_class
/// for1, next for2; retain remaining IDs explicitly as unused. All selected and
/// unused IDs preserve original global order/dtype. Domain/digit outputs are Int32.
/// Default per_class30 selects900 fitting rows from1000; use6 for180/200 held-out.
/// All original selected pixels are retained exactly before native transformation.
/// No random pairing or per-label sort; labels only partition data/diagnostics.
/// Does not certify official source provenance, pixel uniqueness or cross-pool
/// independence: checked LoadSaliency supplies those checks. Validates metadata,
/// balance and unit pixels before gathers. No RNG, fitting, parameters or optimizer.
/// Inputs unchanged; owning fields keep applicable image paths. O(N) host metadata
/// and O(Nselected*784) native image work/storage; synchronize external mutation.
/// @param images Finite unit floating CPU[N,28,28,1], N positive/divisible by10.
/// @param labels Aligned integer CPU[N] class IDs0..9, exactly balanced.
/// @param source_ids Aligned integer CPU[N] nonnegative unique original identities.
/// @param per_class Positive selected rows per class per domain, default30; explicit6 for held-out.
/// @return Owning StarDomains with selected images/originals, domain/digit/source and unused IDs.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto labels=n::cast(n::arange(0,40,1,n::DType::Float64)/4.,n::DType::Int64);
/// auto
/// p=n::models::SplitStarDomains(n::zeros({40,28,28,1}),labels,n::arange(100,140,1,n::DType::Int64),1);
/// // 30 selected rows and10 explicitly unused IDs; first domains are0,1,2.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarDomains, StarDomainImages(), LoadSaliency()
inline StarDomains SplitStarDomains(
    const Tensor& images,
    const Tensor& labels,
    const Tensor& source_ids,
    std::int64_t  per_class = 30) {
  gan_detail::Finite(images);
  if (images.dim() != 4)
    throw std::invalid_argument("Star split needs rank-four NHWC images");
  auto n = images.size(0);
  if (images.dim() != 4 || images.shape() != Shape({n, 28, 28, 1}) || n % 10 || per_class <= 0 ||
      per_class > n / 30)
    throw std::invalid_argument("Star split needs balanced28x28 pools and enough per-class rows");
  gan_detail::Labels(labels, n, 10);
  if (!source_ids.defined() || source_ids.shape() != Shape({n}) ||
      (source_ids.dtype() != DType::Int32 && source_ids.dtype() != DType::Int64))
    throw std::invalid_argument("Star split needs aligned integer source IDs");
  for (double pixel : images.contiguous().to_vector<double>())
    if (pixel < 0 || pixel > 1)
      throw std::invalid_argument("Star split images must lie in[0,1]");
  auto classes = labels.contiguous().to_vector<std::int64_t>(),
       ids     = source_ids.contiguous().to_vector<std::int64_t>();
  std::array<std::int64_t, 10> seen{};
  std::set<std::int64_t>       unique;
  for (std::int64_t i = 0; i < n; ++i) {
    if (ids[i] < 0 || !unique.insert(ids[i]).second)
      throw std::invalid_argument("Star IDs must be nonnegative and unique");
    ++seen[classes[i]];
  }
  for (auto count : seen)
    if (count != n / 10)
      throw std::invalid_argument("Star classes must be balanced");
  seen.fill(0);
  std::vector<std::int64_t> rows, unused;
  std::vector<std::int32_t> codes;
  for (std::int64_t i = 0; i < n; ++i) {
    auto index = seen[classes[i]]++;
    if (index < 3 * per_class) {
      rows.push_back(i);
      codes.push_back(static_cast<std::int32_t>(index / per_class));
    } else
      unused.push_back(i);
  }
  auto original = take_rows(images, rows),
       domain   = from_vector(codes, {static_cast<std::int64_t>(rows.size())});
  return {
      StarDomainImages(original, domain),
      original,
      domain,
      cast(take_rows(labels, rows), DType::Int32),
      take_rows(source_ids, rows),
      take_rows(source_ids, unused)};
}
} // namespace datamunge::nn::models
