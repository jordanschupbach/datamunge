// Tests for nn::io: .npy / .npz reading and writing.

#include <gtest/gtest.h>

#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

// Fixtures written by NumPy 2.3 (np.save / np.savez_compressed).
const char kNpz[] =
    "\x50\x4b\x03\x04\x2d\x00\x00\x00\x08\x00\x00\x00\x21\x00\x0b\xc0\xb0\x76\xff\xff\xff\xff\xff\xff\xff\xff\x05\x00\x14\x00\x77\x2e"
    "\x6e\x70\x79\x01\x00\x10\x00\x98\x00\x00\x00\x00\x00\x00\x00\x57\x00\x00\x00\x00\x00\x00\x00\x9b\xec\x17\xea\x1b\x10\xc9\xc8\x50"
    "\xc6\x50\xad\x9e\x92\x5a\x9c\x5c\xa4\x6e\xa5\xa0\x6e\x93\x66\xa2\xae\xa3\xa0\x9e\x96\x5f\x54\x52\x94\x98\x17\x9f\x5f\x94\x92\x0a"
    "\x12\x77\x4b\xcc\x29\x4e\x05\x8a\x17\x67\x24\x16\xa4\x02\xf9\x1a\x46\x3a\x0a\xc6\x9a\x3a\x0a\xb5\x0a\x64\x03\x2e\x06\x08\xb0\x67"
    "\x60\x68\x00\xe2\x03\x40\xcc\xe0\xc0\xc0\xa0\xe0\x00\x00\x50\x4b\x03\x04\x2d\x00\x00\x00\x08\x00\x00\x00\x21\x00\x74\xa8\xf3\x30"
    "\xff\xff\xff\xff\xff\xff\xff\xff\x0b\x00\x14\x00\x6c\x61\x79\x65\x72\x2f\x62\x2e\x6e\x70\x79\x01\x00\x10\x00\x98\x00\x00\x00\x00"
    "\x00\x00\x00\x4e\x00\x00\x00\x00\x00\x00\x00\x9b\xec\x17\xea\x1b\x10\xc9\xc8\x50\xc6\x50\xad\x9e\x92\x5a\x9c\x5c\xa4\x6e\xa5\xa0"
    "\x6e\x93\x69\xa1\xae\xa3\xa0\x9e\x96\x5f\x54\x52\x94\x98\x17\x9f\x5f\x94\x92\x0a\x12\x77\x4b\xcc\x29\x4e\x05\x8a\x17\x67\x24\x16"
    "\xa4\x02\xf9\x1a\xc6\x3a\x9a\x3a\x0a\xb5\x0a\x14\x00\x2e\x46\x06\x08\xf8\xf7\x1f\x02\x98\xa1\x7c\x00\x50\x4b\x03\x04\x2d\x00\x00"
    "\x00\x08\x00\x00\x00\x21\x00\xee\x19\xbb\x5d\xff\xff\xff\xff\xff\xff\xff\xff\x05\x00\x14\x00\x6d\x2e\x6e\x70\x79\x01\x00\x10\x00"
    "\x82\x00\x00\x00\x00\x00\x00\x00\x46\x00\x00\x00\x00\x00\x00\x00\x9b\xec\x17\xea\x1b\x10\xc9\xc8\x50\xc6\x50\xad\x9e\x92\x5a\x9c"
    "\x5c\xa4\x6e\xa5\xa0\x5e\x93\x64\xa8\xae\xa3\xa0\x9e\x96\x5f\x54\x52\x94\x98\x17\x9f\x5f\x94\x92\x0a\x12\x77\x4b\xcc\x29\x4e\x05"
    "\x8a\x17\x67\x24\x16\xa4\x02\xf9\x1a\x46\x3a\x9a\x3a\x0a\xb5\x0a\x14\x00\x2e\x46\x06\x00\x50\x4b\x01\x02\x2d\x03\x2d\x00\x00\x00"
    "\x08\x00\x00\x00\x21\x00\x0b\xc0\xb0\x76\x57\x00\x00\x00\x98\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01"
    "\x00\x00\x00\x00\x77\x2e\x6e\x70\x79\x50\x4b\x01\x02\x2d\x03\x2d\x00\x00\x00\x08\x00\x00\x00\x21\x00\x74\xa8\xf3\x30\x4e\x00\x00"
    "\x00\x98\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x8e\x00\x00\x00\x6c\x61\x79\x65\x72\x2f\x62\x2e\x6e"
    "\x70\x79\x50\x4b\x01\x02\x2d\x03\x2d\x00\x00\x00\x08\x00\x00\x00\x21\x00\xee\x19\xbb\x5d\x46\x00\x00\x00\x82\x00\x00\x00\x05\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x19\x01\x00\x00\x6d\x2e\x6e\x70\x79\x50\x4b\x05\x06\x00\x00\x00\x00\x03\x00\x03"
    "\x00\x9f\x00\x00\x00\x96\x01\x00\x00\x00\x00";
const char kBe[] =
    "\x93\x4e\x55\x4d\x50\x59\x01\x00\x76\x00\x7b\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x3e\x66\x38\x27\x2c\x20\x27\x66\x6f\x72\x74"
    "\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x46\x61\x6c\x73\x65\x2c\x20\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28\x33\x2c\x29"
    "\x2c\x20\x7d\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x0a"
    "\x3f\xf8\x00\x00\x00\x00\x00\x00\xc0\x02\x00\x00\x00\x00\x00\x00\x40\x08\x00\x00\x00\x00\x00\x00";
const char kFo[] =
    "\x93\x4e\x55\x4d\x50\x59\x01\x00\x76\x00\x7b\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x3c\x69\x34\x27\x2c\x20\x27\x66\x6f\x72\x74"
    "\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x54\x72\x75\x65\x2c\x20\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28\x32\x2c\x20\x33"
    "\x29\x2c\x20\x7d\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x0a"
    "\x00\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x02\x00\x00\x00\x05\x00\x00\x00";
const char kU8[] =
    "\x93\x4e\x55\x4d\x50\x59\x01\x00\x76\x00\x7b\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x7c\x75\x31\x27\x2c\x20\x27\x66\x6f\x72\x74"
    "\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x46\x61\x6c\x73\x65\x2c\x20\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28\x32\x2c\x20"
    "\x32\x29\x2c\x20\x7d\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x0a"
    "\x00\xff\x07\x80";
const char kI2[] =
    "\x93\x4e\x55\x4d\x50\x59\x01\x00\x76\x00\x7b\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x3c\x69\x32\x27\x2c\x20\x27\x66\x6f\x72\x74"
    "\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x46\x61\x6c\x73\x65\x2c\x20\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28\x32\x2c\x29"
    "\x2c\x20\x7d\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20"
    "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x0a"
    "\xfd\xff\x2c\x01";

template <std::size_t N>
std::vector<unsigned char> bytes_of(const char (&literal)[N]) {
  return std::vector<unsigned char>(literal, literal + N - 1);
}

std::string temp_path(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("datamunge_nn_io_" + name)).string();
}

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

void expect_same(const Tensor& a, const Tensor& b) {
  EXPECT_EQ(a.dtype(), b.dtype());
  EXPECT_EQ(a.shape(), b.shape());
  EXPECT_EQ(vals(a), vals(b));
}

}  // namespace

TEST(NnIo, NpyRoundTripEveryDtype) {
  const std::vector<Tensor> cases = {
      nn::tensor({1.5, -2.25, 3e30, -0.0}, {2, 2}, DType::Float32),
      nn::tensor({1.0 / 3, -1e-300, 2.5}, {3}, DType::Float64),
      nn::from_vector(std::vector<std::int32_t>{-7, 0, 2147483647, -2147483647 - 1}),
      nn::from_vector(std::vector<std::int64_t>{-1, 9007199254740993LL}, {2, 1}),
      nn::cast(nn::tensor({1, 0, 1, 1, 0, 0}, {1, 2, 3}), DType::Bool),
      nn::scalar(42.5, DType::Float64),
      nn::zeros({0, 3}),
  };
  for (const auto& t : cases) {
    const Tensor back = nn::decode_npy(nn::encode_npy(t));
    expect_same(back, t);
  }
}

TEST(NnIo, NpyHeaderIsAlignedAndReadableByNumpyConventions) {
  const auto bytes = nn::encode_npy(nn::zeros({2, 3}));
  ASSERT_GE(bytes.size(), 10u);
  EXPECT_EQ(bytes[0], 0x93);
  EXPECT_EQ(std::string(bytes.begin() + 1, bytes.begin() + 6), "NUMPY");
  EXPECT_EQ(bytes[6], 1);
  const std::size_t header = bytes[8] | (bytes[9] << 8);
  EXPECT_EQ((10 + header) % 64, 0u);
  const std::string text(bytes.begin() + 10, bytes.begin() + 10 + static_cast<std::ptrdiff_t>(header));
  EXPECT_NE(text.find("'descr': '<f4'"), std::string::npos);
  EXPECT_NE(text.find("'shape': (2, 3)"), std::string::npos);
  EXPECT_EQ(text.back(), '\n');
  // 1-D shapes keep the trailing comma, scalars are ().
  const auto one = nn::encode_npy(nn::zeros({4}));
  EXPECT_NE(std::string(one.begin(), one.end()).find("(4,)"), std::string::npos);
  const auto zero = nn::encode_npy(nn::scalar(1.0));
  EXPECT_NE(std::string(zero.begin(), zero.end()).find("'shape': ()"), std::string::npos);
}

TEST(NnIo, NpyWritesNonContiguousViewsInLogicalOrder) {
  const Tensor x    = nn::reshape(nn::arange(0, 6), {2, 3});
  const Tensor view = nn::transpose(x);
  ASSERT_FALSE(view.is_contiguous());
  const Tensor back = nn::decode_npy(nn::encode_npy(view));
  EXPECT_EQ(back.shape(), (nn::Shape{3, 2}));
  EXPECT_EQ(vals(back), (std::vector<double>{0, 3, 1, 4, 2, 5}));
  const Tensor sliced = nn::slice_axis(x, 1, 0, 3, 2);
  EXPECT_EQ(vals(nn::decode_npy(nn::encode_npy(sliced))), (std::vector<double>{0, 2, 3, 5}));
}

TEST(NnIo, NumpyWrittenBigEndianFortranAndNarrowInts) {
  const Tensor be = nn::decode_npy(bytes_of(kBe));
  EXPECT_EQ(be.dtype(), DType::Float64);
  EXPECT_EQ(vals(be), (std::vector<double>{1.5, -2.25, 3.0}));

  const Tensor fo = nn::decode_npy(bytes_of(kFo));
  EXPECT_EQ(fo.dtype(), DType::Int32);
  EXPECT_EQ(fo.shape(), (nn::Shape{2, 3}));
  EXPECT_EQ(vals(fo), (std::vector<double>{0, 1, 2, 3, 4, 5}));

  const Tensor u8 = nn::decode_npy(bytes_of(kU8));
  EXPECT_EQ(u8.dtype(), DType::Int32);
  EXPECT_EQ(vals(u8), (std::vector<double>{0, 255, 7, 128}));

  const Tensor i2 = nn::decode_npy(bytes_of(kI2));
  EXPECT_EQ(i2.dtype(), DType::Int32);
  EXPECT_EQ(vals(i2), (std::vector<double>{-3, 300}));
}

TEST(NnIo, NpyRejectsMalformedInput) {
  EXPECT_THROW(nn::decode_npy({1, 2, 3}), std::runtime_error);
  auto truncated = nn::encode_npy(nn::zeros({4}));
  truncated.pop_back();
  EXPECT_THROW(nn::decode_npy(truncated), std::runtime_error);
  auto bad_magic = nn::encode_npy(nn::zeros({4}));
  bad_magic[1]   = 'X';
  EXPECT_THROW(nn::decode_npy(bad_magic), std::runtime_error);
  EXPECT_THROW(nn::load_npy(temp_path("does_not_exist.npy")), std::runtime_error);
}

TEST(NnIo, NpyFileRoundTrip) {
  const std::string path = temp_path("roundtrip.npy");
  const Tensor      t    = nn::tensor({{1, 2, 3}, {4, 5, 6}}, DType::Float64);
  nn::save_npy(path, t);
  expect_same(nn::load_npy(path), t);
  std::remove(path.c_str());
}

TEST(NnIo, NpzRoundTripKeepsOrderAndSlashNames) {
  const std::string     path = temp_path("roundtrip.npz");
  const nn::NamedTensors arrays = {
      {"dense0/W", nn::reshape(nn::arange(0, 6, 1, DType::Float32), {2, 3})},
      {"dense0/b", nn::zeros({3})},
      {"step", nn::scalar(17, DType::Int64)},
      {"mask", nn::cast(nn::tensor({1, 0}), DType::Bool)},
  };
  nn::save_npz(path, arrays);
  const auto back = nn::load_npz(path);
  ASSERT_EQ(back.size(), arrays.size());
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    EXPECT_EQ(back[i].first, arrays[i].first);
    expect_same(back[i].second, arrays[i].second);
  }
  std::remove(path.c_str());
}

TEST(NnIo, NpzEmptyArchive) {
  const std::string path = temp_path("empty.npz");
  nn::save_npz(path, {});
  EXPECT_TRUE(nn::load_npz(path).empty());
  std::remove(path.c_str());
}

TEST(NnIo, NumpyCompressedNpzLoads) {
  // np.savez_compressed: deflate members with ZIP64 extra fields.
  const std::string path = temp_path("numpy_compressed.npz");
  write_bytes(path, bytes_of(kNpz));
  const auto arrays = nn::load_npz(path);
  ASSERT_EQ(arrays.size(), 3u);
  EXPECT_EQ(arrays[0].first, "w");
  EXPECT_EQ(arrays[0].second.dtype(), DType::Float32);
  EXPECT_EQ(arrays[0].second.shape(), (nn::Shape{2, 3}));
  EXPECT_EQ(vals(arrays[0].second), (std::vector<double>{0, 0.5, 1, 1.5, 2, 2.5}));
  EXPECT_EQ(arrays[1].first, "layer/b");
  EXPECT_EQ(arrays[1].second.dtype(), DType::Int64);
  EXPECT_EQ(vals(arrays[1].second), (std::vector<double>{1, -2, 3}));
  EXPECT_EQ(arrays[2].first, "m");
  EXPECT_EQ(arrays[2].second.dtype(), DType::Bool);
  EXPECT_EQ(vals(arrays[2].second), (std::vector<double>{1, 0}));
  std::remove(path.c_str());
}

TEST(NnIo, NpzDetectsCorruption) {
  const std::string path = temp_path("corrupt.npz");
  nn::save_npz(path, {{"x", nn::arange(0, 32)}});
  std::vector<unsigned char> bytes;
  {
    std::ifstream in(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
  }
  bytes[120] ^= 0xff;  // inside the member data
  write_bytes(path, bytes);
  EXPECT_THROW(nn::load_npz(path), std::runtime_error);
  write_bytes(path, {'n', 'o', 't', ' ', 'a', ' ', 'z', 'i', 'p'});
  EXPECT_THROW(nn::load_npz(path), std::runtime_error);
  std::remove(path.c_str());
}
