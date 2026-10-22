// CMD: just run-nn numpy_weights
// Round-trip nn weights and numeric tensors with NumPy NPY/NPZ files.
//
// Port of tflow's numpy_weights. Preparation:
// python3 examples/nn/0_foundations/0.5_persistence_and_interoperability/numpy_interop.py prepare
// (run automatically when the fixtures are missing and python3 with NumPy is
// available). Read independent Python fixtures for byte order, Fortran layout,
// NPY v2/v3, uint8 and compressed/uncompressed NPZ. Load an affine model's
// weights into a ParameterStore, check predictions, and write NPY/NPZ files
// back from C++. Verify the reverse direction with the same script's `verify`
// action (also run automatically at the end when python3 is available).
//
// Cover float32/float64/int32/int64/bool, scalars and zero-sized axes, and
// reject object dtype, damaged lengths, bad magic and archive CRC failures.
// No pickle is read. Differences from tflow: nn has no uint8 dtype, so the
// C++ round trip uses bool where tflow used uint8, and NumPy's uint8 fixture
// is read into int32; TF variables/session become ParameterStore::load of the
// .npz plus nn ops. Fixtures live in output/nn/fixtures/numpy.

#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;

namespace {

const fs::path kScript = fs::path(__FILE__).parent_path() / "numpy_interop.py";

void Same(const Tensor& a, const Tensor& b) {
  check::Require(a.dtype() == b.dtype() && a.shape() == b.shape() && a.to_vector<double>() == b.to_vector<double>(),
                 "exact NumPy tensor round trip");
}

std::vector<unsigned char> ReadBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int RunScript(const std::string& action, const fs::path& root) {
  const std::string command = "python3 '" + kScript.string() + "' " + action + " --root '" + root.string() + "'";
  std::cout.flush();
  return std::system(command.c_str());
}

void Native(const fs::path& root) {
  const nn::NamedTensors arrays{
      {"double", nn::reshape(nn::arange(0, 6, 1, nn::DType::Float64), {2, 3})},
      {"int32", nn::from_vector(std::vector<std::int32_t>{-2, 0, 17})},
      {"int64", nn::scalar(static_cast<double>((std::int64_t{1} << 40) + 7), nn::DType::Int64)},
      {"bool", nn::tensor({0, 1, 1}, {3}, nn::DType::Bool)},
      {"float", nn::tensor({1.5, -.25})},
      {"empty", nn::zeros({2, 0, 3})},
  };
  check::Require(arrays[2].second.item<std::int64_t>() == (std::int64_t{1} << 40) + 7, "exact int64 scalar");
  for (const auto& [name, tensor] : arrays) Same(tensor, nn::decode_npy(nn::encode_npy(tensor)));
  const auto archive = root / "cpp_types.npz";
  nn::save_npz(archive.string(), arrays);
  const auto restored = nn::load_npz(archive.string());
  check::Require(restored.size() == arrays.size(), "NPZ member count");
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    check::Require(restored[i].first == arrays[i].first, "NPZ member names and order");
    Same(arrays[i].second, restored[i].second);
  }
  auto damaged = nn::encode_npy(arrays[0].second);
  damaged.pop_back();
  check::Throws([&] { nn::decode_npy(damaged); }, "truncated NPY must fail");
  damaged    = nn::encode_npy(arrays[0].second);
  damaged[0] = 0;
  check::Throws([&] { nn::decode_npy(damaged); }, "bad NPY magic must fail");
  auto zip = ReadBytes(archive);
  zip[30 + 10 + 1] ^= 1;  // inside double.npy's payload
  const auto broken = root / "cpp_types_damaged.npz";
  {
    std::ofstream out(broken, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(zip.data()), static_cast<std::streamsize>(zip.size()));
  }
  check::Throws([&] { nn::load_npz(broken.string()); }, "damaged NPZ CRC must fail");
  std::cout << "  five numeric dtypes, scalar/empty shapes and malformed-file checks passed\n";
}

void Python(const fs::path& root) {
  for (const auto name : {"fortran.npy", "version2.npy", "version3.npy"}) {
    const auto tensor = nn::load_npy((root / name).string());
    check::Require(tensor.dtype() == nn::DType::Float64 && tensor.shape() == nn::Shape({2, 3}),
                   "NumPy matrix dtype/shape");
    const auto values = tensor.to_vector<double>();
    for (int i = 0; i < 6; ++i) check::Near(values[i], i, 0, "NumPy Fortran/version interoperability");
  }
  const auto big = nn::load_npy((root / "big_endian.npy").string());
  check::Require(big.dtype() == nn::DType::Int32, "big-endian dtype");
  const std::vector<int> expected{1, 256, -7, 1 << 20, -(1 << 20), 42};
  check::Require(big.to_vector<int>() == expected, "byte swapping");
  const auto bytes = nn::load_npy((root / "bytes.npy").string());
  check::Require(bytes.dtype() == nn::DType::Int32 && bytes.to_vector<int>() == std::vector<int>{0, 128, 255},
                 "uint8 values widened to int32");
  check::Throws([&] { nn::load_npy((root / "object.npy").string()); },
                "object dtype must be rejected without reading pickle");
  const auto weights = nn::load_npz((root / "python_weights.npz").string());
  const auto compressed = nn::load_npz((root / "python_compressed.npz").string());
  check::Require(weights.size() == 2 && compressed.size() == 2, "two NPZ members");
  for (std::size_t i = 0; i < 2; ++i) Same(weights[i].second, compressed[i].second);

  nn::ParameterStore store;
  store.load((root / "python_weights.npz").string(), /*strict=*/false);
  const Tensor w = store.at("weights"), b = store.at("bias");
  check::Require(w.shape() == nn::Shape({2, 1}) && b.shape() == nn::Shape({1}), "NumPy weights shapes");
  const Tensor prediction = nn::matmul(nn::tensor({{2, -3}}), w) + b;
  check::Near(prediction.item(), 6.25, 0, "Python weights in nn C++");
  nn::save_npy((root / "cpp_weights.npy").string(), w);
  nn::save_npz((root / "cpp_weights.npz").string(), {{"weights", w}, {"bias", b}});
  std::cout << "  Python Fortran/big-endian/v2/v3/uint8 arrays and stored/DEFLATE NPZ passed\n"
               "  loaded parameters predict 6.25; wrote C++ NPY/NPZ for independent Python verification\n";
}

}  // namespace

int main() {
  try {
    const fs::path root = paths::OutputDir("fixtures/numpy");
    if (!fs::exists(root / "python_weights.npz") || !fs::exists(root / "bytes.npy")) RunScript("prepare", root);
    check::Require(fs::exists(root / "python_weights.npz"), "Run python3 " + kScript.string() + " prepare first");
    std::cout << "== NumPy file interoperability\n";
    Native(root);
    Python(root);
    const int verified = RunScript("verify", root);
    check::Require(verified == 0, "NumPy verification of the C++-written files failed");
    std::cout << "All C++ NumPy checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
