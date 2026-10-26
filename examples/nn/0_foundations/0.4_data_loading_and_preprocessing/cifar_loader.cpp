// CMD: just run-nn cifar_loader
// CIFAR-10/CIFAR-100 binary decoding, planar-to-NHWC conversion and labels.
//
// Port of tflow's cifar_loader. Write original-format records with
// deterministic pixel patterns, read them from disk, and check every output
// coordinate. CIFAR-100 exposes both fine and coarse labels. Multiple CIFAR-10
// batch files concatenate in caller order. Normalize the pixels with nn ops
// and verify channel means. Reject empty, truncated and trailing records,
// invalid labels and bad formats.
//
// Optional format=10|100 and repeated file=PATH arguments load real extracted
// binary batches (the CIFAR-10 cache lives in
// $DATAMUNGE_DATA_DIR/cifar10/data_batch_*.bin). Download the binary archives
// from the dataset authors at https://www.cs.toronto.edu/~kriz/cifar.html;
// Python-pickle files are different. Real CIFAR-10 files are also compared
// with the library loader nn::load_cifar10 (float32 / 255).
//
// Differences from TF: nn has no uint8 dtype, so common/cifar.hpp returns the
// pixels as int32 holding 0..255 (labels int32 as in tflow). Fixtures are
// written under output/nn/fixtures/ (paths::OutputDir).

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/cifar.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

data::Bytes Record(data::Cifar kind, int index) {
  data::Bytes bytes;
  if (kind == data::Cifar::k100) bytes.push_back(4 + index);
  bytes.push_back(kind == data::Cifar::k100 ? 73 + index : 3 + index);
  for (int channel = 0; channel < 3; ++channel)
    for (int pixel = 0; pixel < 1024; ++pixel) bytes.push_back((index * 17 + channel * 61 + pixel) % 256);
  return bytes;
}

void Fixtures(data::Cifar kind) {
  const auto a = Record(kind, 0), b = Record(kind, 1);
  const auto dir = paths::OutputDir(kind == data::Cifar::k10 ? "fixtures/cifar10" : "fixtures/cifar100");
  data::WriteBinary(dir / "batch1.bin", a);
  data::WriteBinary(dir / "batch2.bin", b);
  const auto dataset = data::LoadCifar({dir / "batch1.bin", dir / "batch2.bin"}, kind);
  check::Require(dataset.images.shape() == nn::Shape({2, 32, 32, 3}), "CIFAR NHWC shape");
  const auto images = dataset.images.to_vector<int>(), labels = dataset.labels.to_vector<int>();
  for (int row = 0; row < 2; ++row) {
    check::Require(labels[row] == (kind == data::Cifar::k100 ? 73 : 3) + row, "fine label/order");
    if (kind == data::Cifar::k100)
      check::Require(dataset.coarse_labels.to_vector<int>()[row] == 4 + row, "coarse label/order");
    for (int pixel = 0; pixel < 1024; ++pixel)
      for (int channel = 0; channel < 3; ++channel)
        check::Require(images[row * 3072 + pixel * 3 + channel] == (row * 17 + channel * 61 + pixel) % 256,
                       "channel-major bytes to NHWC");
  }
  const Tensor normalized = nn::div(nn::cast(dataset.images, nn::DType::Float32), 255.0);
  const auto   means      = nn::mean(normalized, {0, 1, 2}).to_vector<float>();
  for (int i = 0; i < 3; ++i) check::Near(means[i], .5, 2e-6, "channel mean");
  std::size_t failures = 0;
  auto        reject   = [&](const data::Bytes& bytes) {
    bool rejected = false;
    try {
      data::ParseCifar({bytes}, kind);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check::Require(rejected, "invalid CIFAR records must fail");
    ++failures;
  };
  reject({});
  auto bad = a;
  bad.pop_back();
  reject(bad);
  bad = a;
  bad.push_back(0);
  reject(bad);
  bad    = a;
  bad[0] = kind == data::Cifar::k100 ? 20 : 10;
  reject(bad);
  if (kind == data::Cifar::k100) {
    bad    = a;
    bad[1] = 100;
    reject(bad);
  }
  std::cout << "  CIFAR-" << (kind == data::Cifar::k10 ? 10 : 100)
            << ": 2 files -> [2,32,32,3], all 6144 pixels and labels verified\n"
            << "    normalized channel means=(0.5,0.5,0.5), " << failures << " malformed records rejected\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    data::Cifar                        kind = data::Cifar::k10;
    std::vector<std::filesystem::path> files;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        at  = arg.find('=');
      const auto key = arg.substr(0, at), value = at == std::string::npos ? "" : arg.substr(at + 1);
      if (key == "file" && !value.empty())
        files.push_back(value);
      else if (key == "format" && (value == "10" || value == "100"))
        kind = value == "10" ? data::Cifar::k10 : data::Cifar::k100;
      else
        throw std::invalid_argument("expected format=10|100 or file=PATH");
    }
    std::cout << "== CIFAR binary loader checks\n";
    Fixtures(data::Cifar::k10);
    Fixtures(data::Cifar::k100);
    if (!files.empty()) {
      const auto                dataset = data::LoadCifar(files, kind);
      std::vector<std::int64_t> histogram(kind == data::Cifar::k10 ? 10 : 100);
      for (int label : dataset.labels.to_vector<int>()) ++histogram[label];
      std::cout << "  loaded " << files.size() << " real file(s), shape=[" << dataset.images.size(0)
                << ",32,32,3]\n    fine-label counts:";
      for (auto n : histogram) std::cout << ' ' << n;
      std::cout << '\n';
      // A single standard test_batch.bin can be checked against the library loader.
      if (kind == data::Cifar::k10 && files.size() == 1 && files[0].filename() == "test_batch.bin") {
        const auto lib = nn::load_cifar10(files[0].parent_path().string(), "test");
        const auto ours = nn::div(nn::cast(dataset.images, nn::DType::Float32), 255.0);
        check::Require(nn::equal(ours, lib.images).to_vector<int>() == std::vector<int>(ours.numel(), 1) &&
                           lib.labels.to_vector<int>() == dataset.labels.to_vector<int>(),
                       "nn::load_cifar10 agrees with the byte decoder");
        std::cout << "    identical to nn::load_cifar10 (pixels / 255, labels)\n";
      }
    }
    std::cout << "All CIFAR loader checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
