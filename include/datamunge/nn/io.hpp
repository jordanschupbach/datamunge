#pragma once

/// @file io.hpp
/// @brief NumPy .npy / .npz files.
///
/// Supported element types: float32, float64, int32, int64 and bool, little
/// or big endian, C or Fortran order. uint8, int8, int16 and uint16 arrays are
/// read into int32 tensors (nn has no narrow integer dtypes). Writing always
/// produces little-endian C-order version 1.0 files. .npz archives are read
/// whether their members are stored or deflate-compressed, and written
/// stored (uncompressed), which NumPy reads directly.

#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

using NamedTensors = std::vector<std::pair<std::string, Tensor>>;

void   save_npy(const std::string& path, const Tensor& tensor);
Tensor load_npy(const std::string& path);

/// In-memory forms of the same encodings.
std::vector<unsigned char> encode_npy(const Tensor& tensor);
Tensor                     decode_npy(const std::vector<unsigned char>& bytes);

/// Writes `arrays` as members "<name>.npy" (names may contain '/').
void save_npz(const std::string& path, const NamedTensors& arrays);
/// Members in archive order, with the ".npy" suffix removed.
NamedTensors load_npz(const std::string& path);

/// In-memory forms of the .npz encoding: encode_npz returns the bytes
/// save_npz writes; decode_npz reads an archive already in memory (same
/// rules as load_npz, errors without a file name).
std::vector<unsigned char> encode_npz(const NamedTensors& arrays);
NamedTensors               decode_npz(const std::vector<unsigned char>& bytes);

}  // namespace datamunge::nn
