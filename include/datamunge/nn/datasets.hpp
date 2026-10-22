#pragma once

/// @file datasets.hpp
/// @brief Loaders for common datasets from files on disk.
///
/// Loaders only read; downloading belongs to the caller (the examples repo
/// has fetch scripts). Images come back as float32 NHWC in [0, 1] and labels
/// as int64.

#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

struct ImageDataset {
  Tensor images;  // float32 [N, H, W, C] in [0, 1]
  Tensor labels;  // int64 [N]
};

/// Reads an IDX file (the MNIST format), gzip-compressed or not (".gz" is
/// detected from the magic bytes). ubyte/byte/short data become int32,
/// int data int32, float float32 and double float64. Validates the header
/// and the file size strictly.
Tensor read_idx(const std::string& path);

/// MNIST or Fashion-MNIST from `directory`, split "train" or "test", using
/// the standard file names (train-images-idx3-ubyte, ...; a ".gz" variant is
/// used when the plain file is missing). Images [N, 28, 28, 1].
ImageDataset load_mnist(const std::string& directory, const std::string& split);

/// CIFAR-10 binary version: data_batch_1..5.bin (train) or test_batch.bin
/// (test) in `directory`. Images [N, 32, 32, 3].
ImageDataset load_cifar10(const std::string& directory, const std::string& split);

/// CIFAR-100 binary version: train.bin or test.bin; `fine` picks the 100
/// fine labels instead of the 20 coarse ones.
ImageDataset load_cifar100(const std::string& directory, const std::string& split, bool fine = true);

/// A numeric CSV file as a float64 matrix [rows, columns]. With `header` the
/// first row is returned in `column_names`. Empty cells and "NA"/"nan" become NaN.
Tensor read_csv(const std::string& path, std::vector<std::string>* column_names = nullptr, bool header = true,
                char delimiter = ',');

}  // namespace datamunge::nn
