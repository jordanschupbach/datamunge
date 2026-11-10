#pragma once
/// @file cycle_domains.hpp
/// @brief Native rotation/inversion and disjoint per-class image domains for unpaired translation.
#include <array>
#include <set>

#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/training.hpp"

namespace datamunge::nn::models {
/// @brief Rotate grayscale NHWC images by90 degrees and invert intensities using native tensors.
///
/// Finite signed Float32/64 CPU[N,H,W,1], positive axes, strides allowed.
/// Default out(y,x)=1-image(H-1-x,y), shape[N,W,H,1]; inverse orientation is
/// out(y,x)=1-image(x,W-1-y). No unit-range clipping, quantization or fitted
/// state. The inverse orientation is the mathematical inverse, but cancellation
/// in1-(1-pixel) need not recover the original floating bits. Keep original
/// pixels when an exact known-mapping diagnostic requires them.
///
/// Uses reverse/transpose/subtraction, with applicable input gradients live.
/// Owning output retains storage/graph, inputs unchanged. No mutation, RNG,
/// parameter creation, optimizer or fitting. O(N*H*W) work/storage; synchronize
/// external input writes. This is a teaching-domain transform, not a loss term.
/// @param image Finite floating CPU[N,H,W,1] grayscale images, all axes positive.
/// @param inverse Use the inverse rotation/inversion orientation when true, default false.
/// @return Owning live rotated/inverted grayscale[N,W,H,1] in input dtype.
/// @throws std::invalid_argument For invalid geometry/dtype/axes/values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto x=n::tensor({0.,.25,.5,.75},{1,2,2,1},n::DType::Float64);
/// auto y=n::models::RotateInvertGrayscale(x);
/// // y is(.5,1,.25,.75); these binary fractions invert/rotate back exactly.
/// @endcode
/// @see SplitCycleDomains(), CycleGan()
inline Tensor RotateInvertGrayscale(const Tensor& image, bool inverse = false) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape().back() != 1)
    throw std::invalid_argument("Rotation/inversion needs positive grayscale NHWC images");
  auto rotated = inverse ? reverse(transpose(image, {0, 2, 1, 3}), {1})
                         : transpose(reverse(image, {1}), {0, 2, 1, 3});
  return 1. - rotated;
}

/// @brief Own disjoint A/original-B rows, transformed B and each domain's original source IDs.
///
/// Flat fields retain storage and applicable image-gather/transform paths after
/// parent/input collection. IDs preserve caller-provided dtype and provenance;
/// the split has no learned or mutable state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto ids=n::arange(0,20,1,n::DType::Int64);
/// auto labels=n::cast(n::arange(0,20,1,n::DType::Float64)/2.,n::DType::Int32);
/// auto d=n::models::SplitCycleDomains(n::ones({20,28,28,1}),labels,ids);
/// // Each domain has10 rows; A takes the first and B the second row of each class.
/// @endcode
/// @see SplitCycleDomains(), RotateInvertGrayscale()
struct CycleDomains {
  Tensor a;     ///< Owning live first-half-per-class images[N/2,28,28,1], in original row order.
  Tensor b;     ///< Owning live rotation/inversion of original_b[N/2,28,28,1].
  Tensor ids_a; ///< Owning original source IDs for A[N/2], input metadata dtype retained.
  Tensor ids_b; ///< Owning original source IDs for B[N/2], disjoint from ids_a.
  Tensor original_b; ///< Owning exact untransformed B rows, retained for diagnostics only.
};

/// @brief Split a balanced ten-class image pool into disjoint unpaired teaching domains.
///
/// Finite unit-range Float32/64 CPU images[N,28,28,1], N>=20 and divisible by20.
/// Matching integer Int32/64 labels[N] in0..9, with exactly N/10 rows per class;
/// integer source_ids[N] nonnegative/unique within this supplied pool. Metadata
/// dtypes may differ and are retained for IDs. Take the first N/20 observed rows
/// of every class for A, remaining rows for original_B, preserving global input
/// order in both gathers. B=RotateInvertGrayscale(original_B). No random split,
/// pairing, reorder by label, fitted transform or hidden cross-pool check.
///
/// Original IDs, not row position, prove disjointness within this pool. This
/// function does not certify official MNIST provenance, image uniqueness or
/// cross-pool independence; use checked LoadSaliency() before each source split.
/// Labels only build balanced partitions and are not network inputs. Keep
/// original_b for diagnostics; neither known target mapping nor labels enter
/// CycleGan's objective. Inputs unchanged; owning fields retain applicable
/// gather/transform paths. No parameter/RNG/optimizer/fitting state. O(N) host
/// metadata plus native image gathers/transform; synchronize external writes.
/// @param images Finite unit-range Float32/64 CPU[N,28,28,1], N divisible by20.
/// @param labels Integer CPU[N] class IDs0..9, every class observed exactly N/10 times.
/// @param source_ids Integer CPU[N] nonnegative unique caller-provided original identities.
/// @return Owning CycleDomains with equal disjoint halves and exact original_B pixels.
/// @throws std::invalid_argument For invalid geometry/dtypes/ranges/balance or repeated source IDs.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto ids=n::arange(0,40,1,n::DType::Int64);
/// auto labels=n::cast(n::arange(0,40,1,n::DType::Float64)/4.,n::DType::Int32);
/// auto out=n::models::SplitCycleDomains(n::zeros({40,28,28,1}),labels,ids);
/// // ids_a begins0,1,4,5; ids_b begins2,3,6,7. B is transformed from the second half.
/// @endcode
/// @see CycleDomains, RotateInvertGrayscale(), LoadSaliency(), CycleGan()
inline CycleDomains
SplitCycleDomains(const Tensor& images, const Tensor& labels, const Tensor& source_ids) {
  gan_detail::Finite(images);
  if (images.dim() != 4 || images.shape() != Shape({images.size(0), 28, 28, 1}) ||
      images.size(0) < 20 || images.size(0) % 20 != 0)
    throw std::invalid_argument(
        "Cycle domains need balanced positive28x28 grayscale rows divisible by20");
  const auto rows = images.size(0);
  for (const auto* metadata : {&labels, &source_ids})
    if (!metadata->defined() || metadata->shape() != Shape({rows}) ||
        (metadata->dtype() != DType::Int32 && metadata->dtype() != DType::Int64))
      throw std::invalid_argument("Cycle domains need aligned integer labels/source IDs");
  for (double pixel : images.contiguous().to_vector<double>())
    if (pixel < 0 || pixel > 1)
      throw std::invalid_argument("Cycle domain pixels must lie in[0,1]");
  const auto classes = labels.contiguous().to_vector<std::int64_t>(),
             ids     = source_ids.contiguous().to_vector<std::int64_t>();
  std::array<std::int64_t, 10> counts{};
  std::set<std::int64_t>       unique;
  for (std::int64_t i = 0; i < rows; ++i) {
    if (classes[i] < 0 || classes[i] >= 10 || ids[i] < 0 || !unique.insert(ids[i]).second)
      throw std::invalid_argument("Cycle class/source IDs must be valid and unique");
    ++counts[classes[i]];
  }
  for (auto count : counts)
    if (count != rows / 10)
      throw std::invalid_argument("Cycle classes must be exactly balanced");
  counts.fill(0);
  std::vector<std::int64_t> a, b;
  for (std::int64_t i = 0; i < rows; ++i)
    (counts[classes[i]]++ < rows / 20 ? a : b).push_back(i);
  auto original = take_rows(images, b);
  return {
      take_rows(images, a),
      RotateInvertGrayscale(original),
      take_rows(source_ids, a),
      take_rows(source_ids, b),
      original};
}
} // namespace datamunge::nn::models
