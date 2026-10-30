// CIFAR-10/CIFAR-100 binary records to NHWC nn tensors (port of tflow's
// common/data/cifar.h). CIFAR-10 records have one label byte followed by 1024
// red, green and blue values. CIFAR-100 records begin with coarse then fine
// labels. Keep both, exposing the fine label as labels. File order determines
// example order. Reject incomplete/empty records and out-of-range labels
// before allocating.
//
// Unlike nn::load_cifar10/100 (float32 in [0, 1], fixed file names), this
// keeps the raw bytes: nn has no uint8 dtype, so pixels are int32 holding
// 0..255 and labels int32, as tflow's uint8/int32 tensors.
#pragma once

/// @file cifar.hpp
/// @brief Strict CIFAR binary decoding into raw NHWC tensors.

#include <datamunge/nn/ops.hpp>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/binary.hpp"

namespace datamunge::nn::data {

/// @brief Binary CIFAR record layout (not the Python pickle format).
enum class Cifar {
  k10, ///< One fine label byte then three 1024-byte channel planes.
  k100 ///< Coarse then fine label bytes, then the same image planes.
};

/// @brief Raw CIFAR image/label tensors with optional coarse labels.
///
/// Images are Int32 [N,32,32,3] in 0..255, labels Int32 [N].
/// coarse_labels is undefined for CIFAR-10 and Int32 [N] for CIFAR-100.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const d::CifarDataset empty;
/// const auto count = empty.size(); // 0
/// @endcode
/// @see ParseCifar()
struct CifarDataset {
  datamunge::nn::Tensor images;        ///< Int32 [N,32,32,3], raw values 0..255.
  datamunge::nn::Tensor labels;        ///< Int32 [N], fine labels for CIFAR-100.
  datamunge::nn::Tensor coarse_labels; ///< Int32 [N] for CIFAR-100; undefined for CIFAR-10.

  /// @brief Number of decoded records.
  /// @return Label count, or zero for an undefined label tensor.
  std::int64_t size() const { return labels.defined() ? labels.numel() : 0; }
};

/// @brief Decode validated CIFAR records in caller-supplied file order.
///
/// Checks labels and sizes before allocating output. No normalisation or autograd history.
/// @param files Nonempty list of nonempty byte buffers; each has whole records.
/// @param kind CIFAR-10 or CIFAR-100 binary layout.
/// @return Owned NHWC Int32 images, fine labels and optional coarse labels.
/// @throws std::invalid_argument On invalid format, lengths, labels or dimension overflow.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::Bytes record(3073, 0); record[0] = 7;
/// const auto batch = d::ParseCifar({record}, d::Cifar::k10);
/// @endcode
///
/// @par References
/// - Krizhevsky, Nair and Hinton, https://www.cs.toronto.edu/~kriz/cifar.html
/// @see LoadCifar()
inline CifarDataset ParseCifar(const std::vector<Bytes>& files, Cifar kind) {
  namespace nn = datamunge::nn;
  if (files.empty() || (kind != Cifar::k10 && kind != Cifar::k100))
    throw std::invalid_argument("need CIFAR files and valid format");
  const std::size_t labels = kind == Cifar::k100 ? 2 : 1, record = 3072 + labels;
  std::size_t       count = 0;
  for (const auto& file : files) {
    if (file.empty() || file.size() % record)
      throw std::invalid_argument("incomplete or empty CIFAR records");
    if (file.size() / record > std::numeric_limits<std::size_t>::max() - count)
      throw std::invalid_argument("CIFAR record count overflow");
    count += file.size() / record;
    for (std::size_t start = 0; start < file.size(); start += record) {
      if (file[start] >= (labels == 2 ? 20 : 10) || (labels == 2 && file[start + 1] >= 100))
        throw std::invalid_argument("CIFAR label out of range");
    }
  }
  const auto pixels = Product(count, 3072);
  if (pixels > std::size_t(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("CIFAR tensor too large");
  std::vector<std::int32_t> image(pixels), fine(count), coarse(labels == 2 ? count : 0);
  std::size_t               row = 0;
  for (const auto& file : files)
    for (std::size_t start = 0; start < file.size(); start += record, ++row) {
      fine[row] = file[start + labels - 1];
      if (labels == 2)
        coarse[row] = file[start];
      for (std::size_t pixel = 0; pixel < 1024; ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel)
          image[row * 3072 + pixel * 3 + channel] = file[start + labels + channel * 1024 + pixel];
    }
  const auto   n = static_cast<std::int64_t>(count);
  CifarDataset result;
  result.images = nn::from_vector(image, {n, 32, 32, 3});
  result.labels = nn::from_vector(fine, {n});
  if (labels == 2)
    result.coarse_labels = nn::from_vector(coarse, {n});
  return result;
}

/// @brief Read binary batch files and decode their records.
/// @param paths Files in desired concatenation order; bounded reads of 512 MiB each.
/// @param kind CIFAR binary format.
/// @return Same raw tensors as ParseCifar.
/// @throws std::runtime_error On file errors; std::invalid_argument on invalid records.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto batch = d::LoadCifar({"test_batch.bin"}, d::Cifar::k10);
/// @endcode
/// @see ParseCifar()
inline CifarDataset LoadCifar(const std::vector<std::filesystem::path>& paths, Cifar kind) {
  std::vector<Bytes> files;
  for (const auto& path : paths)
    files.push_back(ReadBinary(path));
  return ParseCifar(files, kind);
}

} // namespace datamunge::nn::data
