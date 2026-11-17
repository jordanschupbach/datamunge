#pragma once
/// @file safetensors.hpp
/// @brief Bounded, nonexecuting Float32 SafeTensors interchange for published weights.
#include <datamunge/nn/array_io.hpp>
namespace datamunge::nn {
/// @brief Load owning CPU Float32 arrays from a SafeTensors file.
///
/// Accepts contiguous F32 entries only, with little-endian payload and complete,
/// nonoverlapping coverage. Metadata is ignored. Header is bounded to16MiB and
/// payload to1GiB. No code, pickle or framework runtime is executed. The file is
/// read locally; callers authenticate downloaded files separately. Concurrent
/// reads are independent. Arrays retain storage after this function returns.
/// @param path Readable binary file path.
/// @return Named owning Float32 CPU tensors in lexicographic name order.
/// @throws std::invalid_argument For malformed header, unsupported dtype or invalid offsets/shapes.
/// @throws std::runtime_error For an unreadable or truncated file.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <fstream>
/// const std::string header=R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
/// std::ofstream out("build/example.safetensors",std::ios::binary);
/// for(int i=0;i<8;++i)out.put(char((std::uint64_t(header.size())>>(8*i))&255));
/// out<<header;for(unsigned char byte:{0,0,128,63})out.put(char(byte));out.close();
/// auto arrays=dm::nn::load_safetensors("build/example.safetensors"); // x=1.
/// @endcode
/// @par References
/// - Hugging Face, SafeTensors format specification,
/// https://github.com/huggingface/safetensors#format .
/// @see load_npz()
NamedTensors load_safetensors(const std::string& path);
} // namespace datamunge::nn
