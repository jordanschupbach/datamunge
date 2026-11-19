#pragma once

/// @file chunked_attention.hpp
/// @brief Exact scaled dot-product attention with CUDA score recomputation.

#include <datamunge/nn/detail/chunked_attention_ptx.hpp>
#include <datamunge/nn/tensor.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Output and first derivatives of a weighted attention output sum.
/// @par Example
/// @code
/// const dm::nn::ChunkedAttentionResult empty_result{};
/// @endcode
struct ChunkedAttentionResult {
  Tensor output; ///< The normalized weighted values, one row per query.
  Tensor dq; ///< Derivative with respect to the queries.
  Tensor dk; ///< Derivative with respect to the keys.
  Tensor dv; ///< Derivative with respect to the values.
};

/// @brief Evaluate exact attention and its vector-Jacobian product on a CUDA context.
/// Scores are streamed with online softmax and recomputed in the backward pass.
/// Working storage grows linearly with the input and output sizes. Inputs
/// are expected to give finite dot products. This is an explicit first
/// derivative API suitable for numerical checks with an output adjoint.
/// @param context CUDA execution context.
/// @param q CPU Float64 query matrix [queries, width].
/// @param k CPU Float64 key matrix [keys, width].
/// @param v CPU Float64 value matrix [keys, value_width].
/// @param upstream CPU Float64 output adjoint [queries, value_width].
/// @param query_tile Number of rows per query tile, default 64.
/// @param key_tile Number of rows per key tile, default 96.
/// @return CPU Float64 output and derivatives, copied from CUDA after execution.
/// @throws std::invalid_argument Incompatible device, matrix geometry or tile size.
/// @throws std::runtime_error A CUDA program, allocation, dispatch or transfer failed.
/// @par Example
/// @code
/// const dm::compute::Context cuda({dm::compute::Backend::CUDA, 0});
/// const auto x = dm::nn::ones({4, 2}, dm::nn::DType::Float64);
/// const auto fit = dm::nn::cuda_chunked_attention(cuda, x, x, x, x, 2, 3);
/// @endcode
/// @par References
/// - Milakov and Gimelshein (2018), Online normalizer calculation for softmax,
///   https://arxiv.org/abs/1805.02867.
/// @see scaled_dot_product_attention()
inline ChunkedAttentionResult cuda_chunked_attention(
    const compute::Context& context, const Tensor& q, const Tensor& k,
    const Tensor& v, const Tensor& upstream, std::uint32_t query_tile = 64,
    std::uint32_t key_tile = 96) {
  const auto valid = [](const Tensor& value) {
    return value.defined() && value.dim() == 2 && value.numel() > 0 &&
           value.numel() <= std::numeric_limits<std::uint32_t>::max() &&
           value.dtype() == DType::Float64 && value.device().backend == compute::Backend::CPU;
  };
  if (context.device().backend != compute::Backend::CUDA || !query_tile || !key_tile ||
      !valid(q) || !valid(k) || !valid(v) || !valid(upstream) ||
      q.size(1) != k.size(1) || k.size(0) != v.size(0) ||
      upstream.size(0) != q.size(0) || upstream.size(1) != v.size(1))
    throw std::invalid_argument("cuda_chunked_attention: expected CUDA and compatible nonempty CPU Float64 matrices");

  const auto upload = [&](const Tensor& value) {
    auto buffer = context.allocate(value.numel() * sizeof(double));
    const auto contiguous = value.contiguous();
    context.upload(buffer, contiguous.data_ptr<double>(), buffer.size());
    return buffer;
  };
  const auto queries = static_cast<std::uint32_t>(q.size(0));
  const auto keys = static_cast<std::uint32_t>(k.size(0));
  const std::array args{queries, keys, static_cast<std::uint32_t>(q.size(1)),
                        static_cast<std::uint32_t>(v.size(1)), query_tile, key_tile};
  const auto query = upload(q);
  const auto key = upload(k);
  const auto value = upload(v);
  const auto dy = upload(upstream);
  const auto output = context.allocate(dy.size());
  const auto maximum = context.allocate(std::size_t(queries) * sizeof(double));
  const auto normalizer = context.allocate(maximum.size());
  const auto correction = context.allocate(maximum.size());
  const auto dq = context.allocate(query.size());
  const auto dk = context.allocate(key.size());
  const auto dv = context.allocate(value.size());
  const auto launch = [&](const char* entry, const std::vector<compute::Buffer>& buffers,
                          std::uint32_t rows, std::uint32_t tile) {
    tile = std::min(tile, rows);
    const std::uint32_t block = std::min(tile, std::uint32_t{128});
    const auto blocks_per_tile = 1 + (tile - 1) / block;
    const auto tiles = 1 + (rows - 1) / tile;
    if (tiles > std::uint32_t{2147483647} / blocks_per_tile)
      throw std::invalid_argument("cuda_chunked_attention: tile grid exceeds the CUDA launch limit");
    const auto kernel = context.program(
        {entry, {}, detail::chunked_attention_ptx, static_cast<std::uint32_t>(buffers.size()), sizeof(args), {block, 1, 1}});
    context.dispatch(kernel, buffers, args.data(), sizeof(args), {tiles * blocks_per_tile, 1, 1});
  };
  launch("chunked_forward", {query, key, value, output, maximum, normalizer}, queries, query_tile);
  launch("chunked_query_gradient",
         {query, key, value, dy, output, maximum, normalizer, dq, correction}, queries, query_tile);
  launch("chunked_key_value_gradient",
         {query, key, value, dy, maximum, normalizer, correction, dk, dv}, keys, key_tile);
  const auto download = [&](const compute::Buffer& buffer, const Shape& shape) {
    auto result = empty(shape, DType::Float64);
    context.download(result.data_ptr<double>(), buffer, buffer.size());
    return result;
  };
  return {download(output, upstream.shape()), download(dq, q.shape()),
          download(dk, k.shape()), download(dv, v.shape())};
}

} // namespace datamunge::nn
