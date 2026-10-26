#pragma once

/// @file datasets.hpp
/// @brief Loaders for common datasets from files on disk.
///
/// Loaders only read; downloading belongs to the caller (the examples repo
/// has fetch scripts). Images come back as float32 NHWC in [0, 1] and labels
/// as int64.

#include <cstddef>
#include <cstdint>
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
