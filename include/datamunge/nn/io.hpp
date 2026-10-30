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

/// @brief Encode a CPU tensor as an owned NumPy NPY version-one byte array.
/// Packs views in logical row-major order and writes a little-endian payload.
/// Does not mutate storage or record history. GPU input requires an explicit
/// Tensor::to(cpu_context) transfer before serialization.
/// @param tensor Defined CPU tensor with a supported built-in dtype.
/// @return Independent complete NPY document, without writing a file.
/// @throws std::logic_error Undefined tensor or GPU storage.
/// @throws std::runtime_error Unsupported dtype or header above 65535 bytes.
/// @par Example
/// @code
/// const auto x=dm::nn::from_vector(std::vector<std::int64_t>{9007199254740993LL});
/// const auto encoded=dm::nn::encode_npy(x);
/// const auto copy=dm::nn::decode_npy(encoded);
/// std::print("{}\n",copy.numel()); // 1
/// @endcode
/// @par References
/// - NumPy, NPY format version 1.0 specification:
///   https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html.
/// @see decode_npy(), save_npy(), Tensor::to()
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
