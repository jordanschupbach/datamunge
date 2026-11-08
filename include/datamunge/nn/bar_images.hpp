#pragma once
/// @file bar_images.hpp
/// @brief Seeded noisy bar tensors, true object masks and original-row shuffle order.
#include <datamunge/nn/tensor.hpp>
#include <cstdint>
#include <vector>
namespace datamunge::nn {
/// @brief Finite square-image bar geometry and independent clipped pixel noise.
/// Horizontal bars span every column; vertical bars span every row. Position is
/// the first occupied row/column. Both classes share the inclusive position range;
/// max_position=-1 selects side-thickness. Noise is added before clipping to [0,1].
/// Float32/Float64 output is supported. This is a controlled synthetic task, not
/// a natural-image dataset or a claim about translation-invariant predictions.
/// @par Example
/// @code
/// dm::nn::BarImageOptions o;o.noise_stddev=0;o.min_position=2;o.max_position=2;
/// const auto data=dm::nn::make_bar_images(1,7,o);
/// std::print("{}\n",data.images.size(0)); // 2
/// @endcode
/// @see make_bar_images(), BarImageBatch
struct BarImageOptions {
  std::int64_t side{12}; ///< Positive square side length.
  std::int64_t thickness{2}; ///< Positive bar thickness no greater than side.
  std::int64_t min_position{0}; ///< Inclusive first row/column of the bar.
  std::int64_t max_position{-1}; ///< Inclusive last allowed start, or -1 for side-thickness.
  double noise_stddev{0.08}; ///< Finite nonnegative independent Gaussian pixel-noise scale; zero skips noise draws.
  DType dtype{DType::Float32}; ///< Floating output dtype, Float32 or Float64.
};
/// @brief Owned balanced alternating bar images, true masks, metadata and reproducible row order.
/// Shapes use NHWC. Row 2i is horizontal/class zero and row 2i+1 is vertical/class
/// one. Masks identify the noiseless object, not a threshold on the noisy image.
/// shuffled_rows is a permutation drawn after generation from the same local
/// engine; splitting this order does not alter the original image identities.
/// All values own storage; changing an image does not change its independent mask.
/// @par Example
/// @code
/// const auto data=dm::nn::make_bar_images(2,20240815);
/// std::print("{} {}\n",data.labels.size(0),data.shuffled_rows.size()); // 4 4
/// @endcode
/// @see make_bar_images(), BarImageOptions
struct BarImageBatch {
  Tensor images; ///< Floating [2*per_class,side,side,1] clipped pixel values.
  Tensor labels; ///< Int64 [2*per_class], alternating 0=horizontal and 1=vertical.
  Tensor positions; ///< Int64 [2*per_class], actual sampled first occupied row/column.
  Tensor object_mask; ///< Same floating shape/dtype as images, exact zero/one noiseless bar membership.
  std::vector<std::int64_t> shuffled_rows; ///< Actual complete post-generation shuffle of original row ids.
};
/// @brief Generate balanced noisy bar tensors with a local seeded MT19937-64 stream.
/// Draws one position then row-major pixel noise for each alternating class row;
/// the normal distribution is reconstructed per image, preserving the original
/// report's draw/cache convention. Zero noise skips those draws. Samples are
/// clipped to [0,1]; masks retain true object membership. After all images, shuffle
/// ascending original row ids with the same engine. Distribution/shuffle mapping
/// may differ across standard libraries, so bitwise portability is not promised.
/// Positive per_class and valid geometry/range/finite noise/dtype are required.
/// Checks complete tensor dimensions before allocation. O(per_class*side^2) work
/// and owned storage; no global RNG state or fitting is involved.
/// @param per_class Positive examples in each of the two orientations.
/// @param seed Local 64-bit engine seed, default 20240815.
/// @param options Geometry, position/noise domain and output dtype; defaults match the original bar task.
/// @return Owned alternating images/labels/positions/masks and post-generation row permutation.
/// @throws std::invalid_argument If count, geometry, position range, noise or dtype is invalid.
/// @throws std::length_error If supported tensor/vector dimensions cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto data=dm::nn::make_bar_images(300);
/// std::print("{} {}\n",data.images.size(0),data.images.size(1)); // 600 12
/// @endcode
/// @see BarImageBatch, BarImageOptions
BarImageBatch make_bar_images(std::int64_t per_class,std::uint64_t seed=20240815,BarImageOptions options={});
}
