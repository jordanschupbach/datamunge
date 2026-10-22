// Tests for nn dataset loaders: IDX (MNIST), CIFAR binary and CSV.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

// gzip (Python gzip.GzipFile with FNAME "images.idx", mtime 0) of an IDX
// ubyte array of shape [2, 2, 3] holding 10..21.
const char kGz[] =
    "\x1f\x8b\x08\x08\x00\x00\x00\x00\x02\xff\x69\x6d\x61\x67\x65\x73\x2e\x69\x64\x78\x00\x63\x60\xe0\x60\x66\x60\x60\x60\x82\x62\x66"
    "\x2e\x6e\x1e\x5e\x3e\x7e\x01\x41\x21\x61\x11\x51\x00\x7f\x16\x74\x55\x1c\x00\x00\x00";

namespace fs = std::filesystem;

fs::path temp_dir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() / ("datamunge_nn_datasets_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void write_bytes(const fs::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

/// IDX ubyte file with the given dims and payload.
std::vector<unsigned char> idx_ubyte(const std::vector<std::uint32_t>& dims, const std::vector<unsigned char>& data) {
  std::vector<unsigned char> b = {0, 0, 0x08, static_cast<unsigned char>(dims.size())};
  for (auto d : dims)
    for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<unsigned char>((d >> s) & 0xff));
  b.insert(b.end(), data.begin(), data.end());
  return b;
}

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

}  // namespace

TEST(NnDatasets, ReadIdxTypes) {
  const auto dir = temp_dir("idx_types");
  write_bytes(dir / "u8", idx_ubyte({2, 3}, {0, 1, 2, 253, 254, 255}));
  const Tensor u8 = nn::read_idx((dir / "u8").string());
  EXPECT_EQ(u8.dtype(), DType::Int32);
  EXPECT_EQ(u8.shape(), (nn::Shape{2, 3}));
  EXPECT_EQ(vals(u8), (std::vector<double>{0, 1, 2, 253, 254, 255}));

  // Signed byte, big-endian short, int, float and double.
  write_bytes(dir / "i8", {0, 0, 0x09, 1, 0, 0, 0, 2, 0xff, 0x05});
  EXPECT_EQ(vals(nn::read_idx((dir / "i8").string())), (std::vector<double>{-1, 5}));
  write_bytes(dir / "i16", {0, 0, 0x0B, 1, 0, 0, 0, 2, 0xff, 0xfe, 0x01, 0x2c});
  EXPECT_EQ(vals(nn::read_idx((dir / "i16").string())), (std::vector<double>{-2, 300}));
  write_bytes(dir / "i32", {0, 0, 0x0C, 1, 0, 0, 0, 1, 0x80, 0, 0, 0});
  EXPECT_EQ(vals(nn::read_idx((dir / "i32").string())), (std::vector<double>{-2147483648.0}));
  write_bytes(dir / "f32", {0, 0, 0x0D, 1, 0, 0, 0, 1, 0x3f, 0xc0, 0, 0});
  const Tensor f32 = nn::read_idx((dir / "f32").string());
  EXPECT_EQ(f32.dtype(), DType::Float32);
  EXPECT_EQ(vals(f32), (std::vector<double>{1.5}));
  write_bytes(dir / "f64", {0, 0, 0x0E, 1, 0, 0, 0, 1, 0xc0, 0x04, 0, 0, 0, 0, 0, 0});
  const Tensor f64 = nn::read_idx((dir / "f64").string());
  EXPECT_EQ(f64.dtype(), DType::Float64);
  EXPECT_EQ(vals(f64), (std::vector<double>{-2.5}));
  fs::remove_all(dir);
}

TEST(NnDatasets, ReadIdxGzip) {
  const auto dir = temp_dir("idx_gzip");
  write_bytes(dir / "images.gz", std::vector<unsigned char>(kGz, kGz + sizeof(kGz) - 1));
  const Tensor t = nn::read_idx((dir / "images.gz").string());
  EXPECT_EQ(t.shape(), (nn::Shape{2, 2, 3}));
  EXPECT_EQ(t.at({1, 1, 2}), 21);
  EXPECT_EQ(t.at({0, 0, 0}), 10);
  fs::remove_all(dir);
}

TEST(NnDatasets, ReadIdxRejectsBadFiles) {
  const auto dir = temp_dir("idx_bad");
  write_bytes(dir / "magic", {1, 0, 0x08, 1, 0, 0, 0, 1, 7});
  EXPECT_THROW(nn::read_idx((dir / "magic").string()), std::runtime_error);
  write_bytes(dir / "type", {0, 0, 0x07, 1, 0, 0, 0, 1, 7});
  EXPECT_THROW(nn::read_idx((dir / "type").string()), std::runtime_error);
  write_bytes(dir / "short", idx_ubyte({2, 3}, {1, 2, 3, 4, 5}));
  EXPECT_THROW(nn::read_idx((dir / "short").string()), std::runtime_error);
  write_bytes(dir / "long", idx_ubyte({1}, {1, 2}));
  EXPECT_THROW(nn::read_idx((dir / "long").string()), std::runtime_error);
  write_bytes(dir / "header", {0, 0, 0x08, 3, 0, 0});
  EXPECT_THROW(nn::read_idx((dir / "header").string()), std::runtime_error);
  // A gzip file whose payload was cut short.
  std::vector<unsigned char> gz(kGz, kGz + sizeof(kGz) - 1);
  gz.resize(gz.size() - 12);
  write_bytes(dir / "cut.gz", gz);
  EXPECT_THROW(nn::read_idx((dir / "cut.gz").string()), std::runtime_error);
  EXPECT_THROW(nn::read_idx((dir / "missing").string()), std::runtime_error);
  fs::remove_all(dir);
}

TEST(NnDatasets, LoadMnistSynthetic) {
  const auto                 dir = temp_dir("mnist");
  std::vector<unsigned char> pixels(3 * 28 * 28, 0);
  pixels[0]             = 255;  // image 0, (0, 0)
  pixels[28 * 28 + 29]  = 51;   // image 1, (1, 1)
  write_bytes(dir / "train-images-idx3-ubyte", idx_ubyte({3, 28, 28}, pixels));
  write_bytes(dir / "train-labels-idx1-ubyte", idx_ubyte({3}, {5, 0, 9}));
  const auto train = nn::load_mnist(dir.string(), "train");
  EXPECT_EQ(train.images.dtype(), DType::Float32);
  EXPECT_EQ(train.images.shape(), (nn::Shape{3, 28, 28, 1}));
  EXPECT_EQ(train.labels.dtype(), DType::Int64);
  EXPECT_EQ(vals(train.labels), (std::vector<double>{5, 0, 9}));
  EXPECT_FLOAT_EQ(train.images.at({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(train.images.at({1, 1, 1, 0}), 0.2f);
  EXPECT_FLOAT_EQ(train.images.at({2, 5, 5, 0}), 0.0f);

  // Test split from .gz only; mismatched counts are rejected.
  write_bytes(dir / "t10k-labels-idx1-ubyte", idx_ubyte({2}, {1, 2}));
  write_bytes(dir / "t10k-images-idx3-ubyte", idx_ubyte({3, 28, 28}, pixels));
  EXPECT_THROW(nn::load_mnist(dir.string(), "test"), std::runtime_error);
  EXPECT_THROW(nn::load_mnist(dir.string(), "validation"), std::invalid_argument);
  EXPECT_THROW(nn::load_mnist((dir / "nope").string(), "train"), std::runtime_error);
  fs::remove_all(dir);
}

TEST(NnDatasets, LoadCifarConvertsPlanarToHwc) {
  const auto dir = temp_dir("cifar");
  // Two records: label, then R plane, G plane, B plane (32x32 each).
  auto record = [](unsigned char label, unsigned char r, unsigned char g, unsigned char b) {
    std::vector<unsigned char> rec{label};
    for (unsigned char v : {r, g, b})
      for (int i = 0; i < 1024; ++i) rec.push_back(v);
    return rec;
  };
  std::vector<unsigned char> batch;
  for (int k = 1; k <= 5; ++k) {
    auto a = record(static_cast<unsigned char>(k), 255, 0, 51);
    auto b = record(static_cast<unsigned char>(k + 1), 0, 102, 0);
    b[1 + 1024 + 32 * 2 + 3] = 255;  // G plane, y=2, x=3 of the second record
    std::vector<unsigned char> file = a;
    file.insert(file.end(), b.begin(), b.end());
    write_bytes(dir / ("data_batch_" + std::to_string(k) + ".bin"), file);
  }
  const auto train = nn::load_cifar10(dir.string(), "train");
  EXPECT_EQ(train.images.shape(), (nn::Shape{10, 32, 32, 3}));
  EXPECT_EQ(vals(train.labels), (std::vector<double>{1, 2, 2, 3, 3, 4, 4, 5, 5, 6}));
  EXPECT_FLOAT_EQ(train.images.at({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(train.images.at({0, 7, 9, 1}), 0.0f);
  EXPECT_FLOAT_EQ(train.images.at({0, 7, 9, 2}), 0.2f);
  EXPECT_FLOAT_EQ(train.images.at({1, 2, 3, 1}), 1.0f);
  EXPECT_FLOAT_EQ(train.images.at({1, 3, 2, 1}), 0.4f);
  EXPECT_THROW(nn::load_cifar10(dir.string(), "test"), std::runtime_error);  // no test_batch.bin

  // CIFAR-100: coarse and fine label bytes.
  std::vector<unsigned char> rec100{3, 77};
  for (int i = 0; i < 3072; ++i) rec100.push_back(static_cast<unsigned char>(i % 256));
  write_bytes(dir / "train.bin", rec100);
  EXPECT_EQ(vals(nn::load_cifar100(dir.string(), "train").labels), (std::vector<double>{77}));
  EXPECT_EQ(vals(nn::load_cifar100(dir.string(), "train", false).labels), (std::vector<double>{3}));
  // Truncated file.
  rec100.pop_back();
  write_bytes(dir / "test.bin", rec100);
  EXPECT_THROW(nn::load_cifar100(dir.string(), "test"), std::runtime_error);
  fs::remove_all(dir);
}

TEST(NnDatasets, ReadCsv) {
  const auto dir = temp_dir("csv");
  write_text(dir / "a.csv", "x,\"y, quoted\",z\r\n1,2.5,-3e2\r\n4,,NA\n\"7\",8,nan\n");
  std::vector<std::string> names;
  const Tensor             t = nn::read_csv((dir / "a.csv").string(), &names);
  EXPECT_EQ(names, (std::vector<std::string>{"x", "y, quoted", "z"}));
  EXPECT_EQ(t.dtype(), DType::Float64);
  EXPECT_EQ(t.shape(), (nn::Shape{3, 3}));
  EXPECT_EQ(t.at({0, 2}), -300);
  EXPECT_TRUE(std::isnan(t.at({1, 1})));
  EXPECT_TRUE(std::isnan(t.at({1, 2})));
  EXPECT_EQ(t.at({2, 0}), 7);

  write_text(dir / "b.tsv", "1\t2\n3\t4");
  const Tensor tsv = nn::read_csv((dir / "b.tsv").string(), nullptr, false, '\t');
  EXPECT_EQ(vals(tsv), (std::vector<double>{1, 2, 3, 4}));

  write_text(dir / "ragged.csv", "a,b\n1,2\n3\n");
  EXPECT_THROW(nn::read_csv((dir / "ragged.csv").string()), std::runtime_error);
  write_text(dir / "text.csv", "a\nhello\n");
  EXPECT_THROW(nn::read_csv((dir / "text.csv").string()), std::runtime_error);
  write_text(dir / "quote.csv", "a\n\"1\n");
  EXPECT_THROW(nn::read_csv((dir / "quote.csv").string()), std::runtime_error);
  fs::remove_all(dir);
}

TEST(NnDatasets, RealMnistWhenAvailable) {
  const char* dir = std::getenv("DATAMUNGE_MNIST_DIR");
  if (!dir) GTEST_SKIP() << "set DATAMUNGE_MNIST_DIR to run";
  const auto train = nn::load_mnist(dir, "train");
  const auto test  = nn::load_mnist(dir, "test");
  EXPECT_EQ(train.images.shape(), (nn::Shape{60000, 28, 28, 1}));
  EXPECT_EQ(test.images.shape(), (nn::Shape{10000, 28, 28, 1}));
  EXPECT_EQ(train.labels.at({0}), 5);
  EXPECT_EQ(test.labels.at({0}), 7);
  EXPECT_LE(nn::max(train.images).item(), 1.0);
  EXPECT_GE(nn::min(train.images).item(), 0.0);
}

TEST(NnDatasets, RealCifar10WhenAvailable) {
  const char* dir = std::getenv("DATAMUNGE_CIFAR10_DIR");
  if (!dir) GTEST_SKIP() << "set DATAMUNGE_CIFAR10_DIR to run";
  const auto train = nn::load_cifar10(dir, "train");
  const auto test  = nn::load_cifar10(dir, "test");
  EXPECT_EQ(train.images.shape(), (nn::Shape{50000, 32, 32, 3}));
  EXPECT_EQ(test.images.shape(), (nn::Shape{10000, 32, 32, 3}));
  EXPECT_EQ(train.labels.at({0}), 6);  // frog
  EXPECT_EQ(test.labels.at({0}), 3);   // cat
}
