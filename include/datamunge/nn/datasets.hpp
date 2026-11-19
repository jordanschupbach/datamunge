#pragma once

/// @file datasets.hpp
/// @brief Loaders for common datasets from files on disk.
///
/// Loaders only read; callers use datasets::fetch() or
/// datasets::prepare_example_data() for native acquisition and preparation. Images come back as float32 NHWC in [0, 1] and labels
/// as int64.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/tensor.hpp"
#include "datamunge/nn/training.hpp"

namespace datamunge::nn {

struct ImageDataset {
  Tensor images;  // float32 [N, H, W, C] in [0, 1]
  Tensor labels;  // int64 [N]
};

/// @brief A fitting and validation split with paired source row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({5, 4, 4, 1}), n::arange(0, 5).to(n::DType::Int64)};
/// const auto split = n::split_image_holdout(data, 2, 7, 2);
/// const auto count = split.fit.labels.numel(); // 2
/// @endcode
/// @see split_image_holdout()
struct ImageHoldout {
  ImageDataset fit;                          ///< Owned, detached fitting images and labels.
  ImageDataset validation;                   ///< Owned, detached validation images and labels.
  std::vector<std::int64_t> fit_rows;         ///< Zero-based source indices for fitting.
  std::vector<std::int64_t> validation_rows;  ///< Zero-based source indices for validation.
};

/// @brief Shuffle labelled images and reserve a fixed number of validation rows.
///
/// Uses the seed and shuffle convention of BatchOrder. The validation rows are
/// the tail of the permutation. An optional fitting cap keeps the first rows of
/// the fitting pool, so changing the cap cannot change validation membership.
/// Gathers images and labels in the same order without changing their dtypes or
/// values. The split does not stratify label counts.
/// @param data Images [N,H,W,C] and corresponding labels [N], already loaded.
/// @param validation_rows Number of rows held out, between 1 and N-1.
/// @param seed Seed of the row permutation. Defaults to zero.
/// @param fit_limit Maximum fitting rows. Zero (default) uses all N-validation_rows rows.
/// @return The two detached datasets and their source indices.
/// @throws std::invalid_argument If tensor shapes, row counts or the fitting cap are invalid.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({10, 4, 4, 1}), n::arange(0, 10).to(n::DType::Int64)};
/// const auto split = n::split_image_holdout(data, 3, 2026);
/// const auto count = split.fit.images.size(0); // 7
/// @endcode
/// @see ImageHoldout, BatchOrder, take_rows()
inline ImageHoldout split_image_holdout(const ImageDataset& data, std::int64_t validation_rows,
                                        std::uint64_t seed = 0, std::int64_t fit_limit = 0) {
  if (!data.images.defined() || !data.labels.defined() || data.images.dim() != 4 ||
      data.labels.dim() != 1 || data.images.size(0) != data.labels.size(0) ||
      data.images.numel() == 0)
    throw std::invalid_argument("image holdout requires nonempty [N,H,W,C] images and [N] labels");
  const auto rows = data.images.size(0);
  if (validation_rows < 1 || validation_rows >= rows || fit_limit < 0 ||
      fit_limit > rows - validation_rows)
    throw std::invalid_argument("image holdout has invalid validation size or fitting cap");

  BatchOrder order(rows, rows, seed);
  order.begin_epoch();
  const auto permutation = order.next();
  const auto available = rows - validation_rows;
  const auto used = fit_limit == 0 ? available : fit_limit;

  ImageHoldout split;
  split.fit_rows.assign(permutation.begin(), permutation.begin() + used);
  split.validation_rows.assign(permutation.begin() + available, permutation.end());
  NoGradGuard guard;
  split.fit = {take_rows(data.images, split.fit_rows), take_rows(data.labels, split.fit_rows)};
  split.validation = {take_rows(data.images, split.validation_rows),
                      take_rows(data.labels, split.validation_rows)};
  return split;
}

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

/// One labelled file of an image folder.
struct ImageFile {
  std::string path;
  std::int64_t label = 0;
};

/// Result of scan_image_folder: class names (index = label) and files.
struct ImageFolder {
  std::vector<std::string> class_names;
  std::vector<ImageFile> images;
};

/// Scans `root` with one immediate subdirectory per class (the layout of
/// tf.keras.utils.image_dataset_from_directory). Class names are the sorted
/// subdirectory names unless `class_names` is given (e.g. the training classes
/// when scanning validation data, so a missing class cannot shift labels);
/// subdirectories not in that list throw std::invalid_argument, as do empty,
/// duplicate or path-like names. Files ending in .png/.jpg/.jpeg (any case)
/// are listed sorted by path per class; other files are ignored. Throws
/// std::invalid_argument when `root` is not a directory or holds no images.
ImageFolder scan_image_folder(const std::string& root, std::vector<std::string> class_names = {});

/// Decodes a PNG or JPEG file (chosen by extension, like TensorFlow's
/// DecodePng/DecodeJpeg with channels=3: grayscale expands to RGB, alpha is
/// dropped) and resizes it to [height, width, 3] float32 in [0, 1] with
/// bilinear half-pixel-centre resizing (aspect ratio is not preserved).
/// Throws std::invalid_argument for other extensions and std::runtime_error
/// (naming the file) for unreadable or corrupt images.
Tensor load_image_rgb(const std::string& path, std::int64_t height, std::int64_t width);

/// Decodes `rows` of `images` into a batch: images [rows, height, width, 3]
/// float32 in [0, 1] (see load_image_rgb) and their int64 labels. A corrupt
/// file throws rather than being skipped, so labels never misalign.
ImageDataset load_image_batch(const std::vector<ImageFile>& images, const std::vector<std::size_t>& rows,
                              std::int64_t height, std::int64_t width);

/// scan_image_folder(root, class_names) followed by load_image_batch over all
/// files in scan order. `folder`, when non-null, receives the scan (class names).
ImageDataset load_image_folder(const std::string& root, std::int64_t height, std::int64_t width,
                               std::vector<std::string> class_names = {}, ImageFolder* folder = nullptr);

}  // namespace datamunge::nn
