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

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/binary.hpp"

namespace data {

enum class Cifar { k10, k100 };

struct CifarDataset {
  datamunge::nn::Tensor images;         // int32 [N, 32, 32, 3], values 0..255
  datamunge::nn::Tensor labels;         // int32 [N] (fine labels for CIFAR-100)
  datamunge::nn::Tensor coarse_labels;  // int32 [N], CIFAR-100 only
  std::int64_t          size() const { return labels.defined() ? labels.numel() : 0; }
};

inline CifarDataset ParseCifar(const std::vector<Bytes>& files, Cifar kind) {
  namespace nn = datamunge::nn;
  if (files.empty() || (kind != Cifar::k10 && kind != Cifar::k100))
    throw std::invalid_argument("need CIFAR files and valid format");
  const std::size_t labels = kind == Cifar::k100 ? 2 : 1, record = 3072 + labels;
  std::size_t       count  = 0;
  for (const auto& file : files) {
    if (file.empty() || file.size() % record) throw std::invalid_argument("incomplete or empty CIFAR records");
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
      if (labels == 2) coarse[row] = file[start];
      for (std::size_t pixel = 0; pixel < 1024; ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel)
          image[row * 3072 + pixel * 3 + channel] = file[start + labels + channel * 1024 + pixel];
    }
  const auto    n = static_cast<std::int64_t>(count);
  CifarDataset result;
  result.images = nn::from_vector(image, {n, 32, 32, 3});
  result.labels = nn::from_vector(fine, {n});
  if (labels == 2) result.coarse_labels = nn::from_vector(coarse, {n});
  return result;
}

inline CifarDataset LoadCifar(const std::vector<std::filesystem::path>& paths, Cifar kind) {
  std::vector<Bytes> files;
  for (const auto& path : paths) files.push_back(ReadBinary(path));
  return ParseCifar(files, kind);
}

}  // namespace data
