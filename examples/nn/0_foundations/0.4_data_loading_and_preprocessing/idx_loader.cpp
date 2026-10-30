// CMD: just run-nn idx_loader
// MNIST and Fashion-MNIST: strict IDX loading and nn preprocessing.
//
// Port of tflow's idx_loader. Create a tiny on-disk IDX pair with known
// bytes, load uint8 NHWC images and integer labels, shuffle matched
// image/label rows, then cast and divide by 255 with nn ops. Check malformed
// magic, truncated and trailing data, dimension overflow, count mismatches and
// label ranges before allocation.
//
// Also inspect cached real test sets under $DATAMUNGE_DATA_DIR/{mnist,
// fashion_mnist} (default ~/projects/tflow_examples/output/datasets; fetch
// there with python3 scripts/fetch_datasets.py both --split test). Optional
// root=PATH changes that root; dir=PATH loads a required IDX pair directly, and
// split=train|t10k selects its filename prefix. No network access occurs.
//
// Differences from tflow: nn::read_idx reads one IDX file of any element type
// (and gunzips ".gz" files itself), so the pair rules of tflow's ParseIdx
// (uint8 rank-3 images, uint8 rank-1 labels, nonzero dimensions, equal counts,
// labels < 10) are checked here in LoadPair. nn has no uint8 dtype: pixels
// come back as int32 holding 0..255. tflow's "bounded file read" check (a size
// limit on its byte reader) has no nn counterpart and is dropped; instead the
// real test set is also loaded from its .gz file and compared.

#include <algorithm>
#include <array>
#include <cstdint>
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
using Bytes  = std::vector<unsigned char>;
using nn::Tensor;

namespace {

struct Pair {
  Tensor images;  // int32 [N, H, W, 1] with byte values
  Tensor labels;  // int64 [N]
};

void Append32(Bytes& bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) bytes.push_back(static_cast<unsigned char>((value >> shift) & 255));
}

void WriteBinary(const fs::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  check::Require(bool(out), "write " + path.string());
}

/// The first four bytes of a (possibly gzip-compressed) IDX file, via the
/// same reader: a uint8 file's magic is 0x0000 08 <rank>.
void RequireUbyte(const fs::path& path, int rank) {
  std::ifstream in(path, std::ios::binary);
  unsigned char magic[4]{};
  in.read(reinterpret_cast<char*>(magic), 4);
  if (in && magic[0] == 0x1f && magic[1] == 0x8b) return;  // gzip: the payload is validated by read_idx
  if (!in || magic[0] || magic[1] || magic[2] != 0x08 || magic[3] != rank)
    throw std::invalid_argument("IDX requires uint8 rank-3 images and rank-1 labels: " + path.string());
}

/// An MNIST-style IDX pair with tflow's ParseIdx rules.
Pair LoadPair(const fs::path& images_path, const fs::path& labels_path, int classes = 10) {
  RequireUbyte(images_path, 3);
  RequireUbyte(labels_path, 1);
  Tensor images, labels;
  try {
    images = nn::read_idx(images_path.string());
    labels = nn::read_idx(labels_path.string());
  } catch (const std::runtime_error& e) {
    throw std::invalid_argument(e.what());
  }
  if (images.dim() != 3 || labels.dim() != 1) throw std::invalid_argument("IDX requires rank-3 images and rank-1 labels");
  for (auto d : images.shape())
    if (d == 0) throw std::invalid_argument("invalid IDX dimensions");
  if (images.size(0) != labels.size(0)) throw std::invalid_argument("IDX payload size/count mismatch");
  if (nn::max(labels).item() >= classes) throw std::invalid_argument("IDX label out of range");
  return {nn::reshape(images, {images.size(0), images.size(1), images.size(2), 1}), nn::cast(labels, nn::DType::Int64)};
}

void Fixtures() {
  Bytes images, labels;
  for (std::uint32_t value : {2051, 2, 2, 3}) Append32(images, value);
  for (std::uint32_t value : {2049, 2}) Append32(labels, value);
  const Bytes pixels{0, 51, 102, 153, 204, 255, 255, 204, 153, 102, 51, 0};
  images.insert(images.end(), pixels.begin(), pixels.end());
  labels.insert(labels.end(), {7, 3});
  const fs::path directory = paths::OutputDir("fixtures/idx");
  WriteBinary(directory / "images.idx", images);
  WriteBinary(directory / "labels.idx", labels);
  const auto dataset = LoadPair(directory / "images.idx", directory / "labels.idx");
  check::Require(dataset.images.shape() == nn::Shape({2, 2, 3, 1}), "IDX NHWC shape");
  const auto x = dataset.images.to_vector<int>();
  for (int i = 0; i < 12; ++i) check::Require(x[i] == pixels[i], "IDX pixel order");
  const auto y = dataset.labels.to_vector<int>();
  check::Require(y[0] == 7 && y[1] == 3, "IDX label order");
  const Tensor shuffled_x = nn::take_rows(dataset.images, {1, 0});
  const Tensor shuffled_y = nn::take_rows(dataset.labels, {1, 0});
  check::Require(shuffled_x.at({0, 0, 0, 0}) == 255 && shuffled_y.at({0}) == 3, "paired row gathering");

  const Tensor normalized = nn::cast(shuffled_x, nn::DType::Float32) / 255.0;
  const Tensor means      = nn::mean(normalized, {1, 2, 3});
  const auto   n = normalized.to_vector<double>(), sx = shuffled_x.to_vector<double>();
  for (int i = 0; i < 12; ++i) check::Near(n[i], sx[i] / 255., 1e-7, "pixel normalization");
  for (double m : means.to_vector<double>()) check::Near(m, .5, 1e-7, "per-image mean");

  std::size_t failures = 0;
  auto        reject   = [&](const Bytes& bad_x, const Bytes& bad_y) {
    WriteBinary(directory / "bad_images.idx", bad_x);
    WriteBinary(directory / "bad_labels.idx", bad_y);
    check::Throws([&] { LoadPair(directory / "bad_images.idx", directory / "bad_labels.idx"); },
                  "invalid IDX pair must fail");
    ++failures;
  };
  Bytes bx = images, by = labels;
  bx[3] = 0, reject(bx, labels);                                  // rank-0 image magic
  by = labels, by[3] = 0, reject(images, by);                     // rank-0 label magic
  bx = images, bx.pop_back(), reject(bx, labels);                 // truncated images
  bx = images, bx.push_back(0), reject(bx, labels);               // trailing image byte
  by = labels, by.pop_back(), reject(images, by);                 // truncated labels
  by = labels, by.push_back(0), reject(images, by);               // trailing label byte
  bx = images, bx[7] = 0, reject(bx, labels);                     // zero images
  bx = images, bx[11] = 0, reject(bx, labels);                    // zero rows
  by = labels, by[7] = 3, reject(images, by);                     // label count mismatch
  by = labels, by.back() = 10, reject(images, by);                // label out of range
  bx = images;
  for (int i = 4; i < 16; ++i) bx[i] = 255;                       // dimension overflow
  reject(bx, labels);
  reject({}, labels);
  reject(images, {});
  std::cout << "  2 images, shape [2,2,3,1], paired shuffle and normalization passed\n"
            << "  " << failures << " malformed IDX pairs rejected\n";
}

void Inspect(const fs::path& dir, const std::string& split) {
  const auto images = dir / (split + "-images-idx3-ubyte"), labels = dir / (split + "-labels-idx1-ubyte");
  const auto dataset = LoadPair(fs::exists(images) ? images : fs::path(images.string() + ".gz"),
                                fs::exists(labels) ? labels : fs::path(labels.string() + ".gz"));
  const auto n       = dataset.labels.size(0);
  std::array<std::int64_t, 10> counts{};
  for (auto label : dataset.labels.to_vector<std::int64_t>()) ++counts[static_cast<std::size_t>(label)];
  const Tensor pixels = nn::cast(dataset.images, nn::DType::Float64);
  const int    low = static_cast<int>(nn::min(pixels).item()), high = static_cast<int>(nn::max(pixels).item());
  std::cout << "  " << dir.filename().string() << '/' << split << ": " << n
            << " images, shape=" << nn::shape_str(dataset.images.shape())
            << ", normalized mean=" << nn::mean(pixels).item() / 255 << ", byte range=[" << low << ',' << high
            << "]\n    class counts:";
  for (auto count : counts) std::cout << ' ' << count;
  std::cout << '\n';
  if (fs::exists(images) && fs::exists(labels) && fs::exists(fs::path(images.string() + ".gz")) && fs::exists(fs::path(labels.string() + ".gz"))) {
    const auto gz = LoadPair(images.string() + ".gz", labels.string() + ".gz");
    check::Require(nn::all(nn::equal(gz.images, dataset.images)).item() == 1 &&
                       nn::all(nn::equal(gz.labels, dataset.labels)).item() == 1,
                   "gzip IDX matches the decompressed file");
    std::cout << "    .gz files decode to identical tensors\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path    root = paths::DataRoot(), required;
    std::string split = "t10k";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        at  = arg.find('=');
      const auto key = arg.substr(0, at), value = at == std::string::npos ? "" : arg.substr(at + 1);
      if (value.empty()) throw std::invalid_argument("expected key=value");
      if (key == "root")
        root = value;
      else if (key == "dir")
        required = value;
      else if (key == "split" && (value == "train" || value == "t10k"))
        split = value;
      else
        throw std::invalid_argument("unknown IDX option: " + arg);
    }
    std::cout << "== IDX fixture and validation checks\n";
    Fixtures();
    std::cout << "\n== Real IDX datasets\n";
    if (!required.empty())
      Inspect(required, split);
    else
      for (const std::string name : {"mnist", "fashion_mnist"}) {
        dataset_assets::EnsureMnist((root / name).string(), name == "fashion_mnist", split == "train" ? "train" : "test");
        Inspect(root / name, split);
      }
    std::cout << "All IDX loader checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
