#pragma once
/// @file array_io.hpp
/// @brief Native IDX writing, explicit NPY layout/byte order and compressed NPZ serialization.
#include "datamunge/nn/io.hpp"

namespace datamunge::nn {
/// @brief Encode a CPU tensor in explicitly selected NPY storage order and byte order.
///
/// Preserves logical coordinates, dtype, scalar shapes and empty axes. Packs views
/// without mutation; copies O(numel*dtype_size) bytes. Bool payloads have no byte order.
/// @param tensor Defined CPU tensor, using the same dtypes as encode_npy().
/// @param fortran_order True selects column-major payload order; false selects row-major.
/// @param big_endian True selects big-endian multibyte elements; false selects little-endian.
/// @return Complete owned NPY version-one document.
/// @throws std::logic_error For undefined or GPU tensors.
/// @throws std::runtime_error For unsupported dtype or oversized header.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto x=dm::nn::tensor({0.,1.,2.,3.,4.,5.},{2,3});
/// const auto bytes=dm::nn::encode_npy(x,true,true);
/// const auto restored=dm::nn::decode_npy(bytes);
/// @endcode
/// @par References
/// - NumPy NPY format, https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
/// @see decode_npy(), save_npy()
std::vector<unsigned char> encode_npy(const Tensor& tensor, bool fortran_order, bool big_endian);

/// @brief Save a CPU tensor with explicit NPY layout and byte order.
/// @param path Destination file; parent directory must exist.
/// @param tensor Defined CPU tensor; views are copied in logical coordinate order.
/// @param fortran_order True for column-major storage, false for row-major.
/// @param big_endian True for big-endian multibyte elements, false for little-endian.
/// @throws std::logic_error For undefined or GPU tensors.
/// @throws std::runtime_error For encoding or file-write errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::save_npy("build/fortran.npy",dm::nn::tensor({1.,2.,3.,4.},{2,2}),true,false);
/// @endcode
/// @par References
/// - NumPy NPY format, https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
/// @see encode_npy(), load_npy()
void save_npy(const std::string& path, const Tensor& tensor, bool fortran_order, bool big_endian);

/// @brief Encode named CPU tensors as stored or DEFLATE-compressed NPZ members.
///
/// Uses deterministic native DEFLATE level6 and ZIP32 headers with CRC32. Preserves
/// array order and slash-containing names; names must be unique and nonempty.
/// No NumPy, Python, system archive program or optional dataset adapter is required.
/// @param arrays Ordered name/tensor pairs; an empty collection creates an empty archive.
/// @param compressed True selects ZIP method8; false selects uncompressed method0.
/// @return Complete owned NPZ bytes; no files are written.
/// @throws std::invalid_argument For empty/duplicate/oversized names.
/// @throws std::logic_error For undefined or GPU tensors.
/// @throws std::runtime_error If a member/archive exceeds ZIP32 limits or encoding fails.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto bytes=dm::nn::encode_npz({{"x",dm::nn::zeros({100})}},true);
/// const auto arrays=dm::nn::decode_npz(bytes);
/// @endcode
/// @par References
/// - NumPy NPZ format, https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
/// @see save_npz(), decode_npz()
std::vector<unsigned char> encode_npz(const NamedTensors& arrays, bool compressed);

/// @brief Save named CPU tensors as a stored or DEFLATE-compressed NPZ archive.
/// @param path Destination file; parent directory must exist.
/// @param arrays Ordered unique name/tensor pairs, possibly empty.
/// @param compressed True selects native DEFLATE; false stores uncompressed members.
/// @throws std::invalid_argument For invalid array names.
/// @throws std::logic_error For undefined or GPU tensors.
/// @throws std::runtime_error For ZIP32 limits, encoding or file-write errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::save_npz("build/compressed.npz",{{"x",dm::nn::zeros({100})}},true);
/// @endcode
/// @par References
/// - NumPy NPZ format, https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
/// @see encode_npz(), load_npz()
void save_npz(const std::string& path, const NamedTensors& arrays, bool compressed);

/// @brief Write numeric CPU tensor data in the big-endian IDX interchange format.
///
/// Writes shape dimensions followed by exact big-endian numeric values. Supports
/// Int32, Float32, Float64, and byte encoding; no mutation or graph recording.
/// A Bool tensor uses unsigned-byte IDX automatically. Scalar/rank>255 tensors,
/// dimensions above UINT32_MAX and Int64 input without byte encoding are rejected.
/// @param path Destination file; parent directory must exist.
/// @param tensor Defined CPU tensor of rank1..255 with positive dimensions.
/// @param unsigned_byte True requires finite integral values in0..255 and emits IDX
/// type0x08; false (default) preserves supported numeric dtype.
/// @throws std::invalid_argument For unsupported dtype, rank, shape or byte values.
/// @throws std::logic_error For undefined or GPU tensors.
/// @throws std::runtime_error For file-write errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::write_idx("build/labels.idx",dm::nn::tensor({7,3}).to(dm::nn::DType::Int32),true);
/// const auto labels=dm::nn::read_idx("build/labels.idx");
/// @endcode
/// @par References
/// - IDX format, https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
/// @see read_idx()
void write_idx(const std::string& path, const Tensor& tensor, bool unsigned_byte = false);
} // namespace datamunge::nn
