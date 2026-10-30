/// @file array_io.cpp
/// @brief Byte-level native NPY layout, NPZ compression and IDX writer regressions.
#include <gtest/gtest.h>

#include <bit>
#include <datamunge/nn/array_io.hpp>
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
namespace {
namespace n = dm::nn;
std::size_t payload(const std::vector<unsigned char>& bytes) {
  return 10 + bytes[8] + (std::size_t(bytes[9]) << 8);
}
TEST(ArrayIo, FortranPayloadMatchesSpecifiedCoordinateOrder) {
  const auto        x     = n::tensor({0., 1., 2., 3., 4., 5.}, {2, 3}, n::DType::Float64);
  const auto        bytes = n::encode_npy(x, true, false);
  const std::string header(bytes.begin() + 10, bytes.begin() + payload(bytes));
  EXPECT_NE(header.find("'fortran_order': True"), std::string::npos);
  const std::vector<double> expected{0, 3, 1, 4, 2, 5};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    std::uint64_t word = 0;
    for (int k = 0; k < 8; ++k)
      word |= std::uint64_t(bytes[payload(bytes) + i * 8 + k]) << (8 * k);
    EXPECT_EQ(std::bit_cast<double>(word), expected[i]);
  }
  EXPECT_EQ(n::decode_npy(bytes).to_vector<double>(), x.to_vector<double>());
}
TEST(ArrayIo, BigEndianPayloadPreservesExactLargeIntegersAndViews) {
  const auto x = n::from_vector(std::vector<std::int64_t>{0x0102030405060708LL, -2, 3, 4}, {2, 2});
  const auto bytes = n::encode_npy(x, false, true);
  const auto at    = payload(bytes);
  EXPECT_EQ(
      (std::vector<unsigned char>(bytes.begin() + at, bytes.begin() + at + 8)),
      (std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(n::decode_npy(bytes).to_vector<std::int64_t>(), x.to_vector<std::int64_t>());
  const auto view = n::transpose(x);
  EXPECT_EQ(
      n::decode_npy(n::encode_npy(view, true, true)).to_vector<std::int64_t>(),
      view.to_vector<std::int64_t>());
}
TEST(ArrayIo, ScalarEmptyAndBoolLayoutsRemainWellDefined) {
  for (const auto& x :
       {n::full({},3.), n::zeros({2, 0, 3}), n::tensor({0, 1, 1, 0}, {2, 2}, n::DType::Bool)})
    for (bool f : {false, true})
      for (bool b : {false, true}) {
        const auto value = n::decode_npy(n::encode_npy(x, f, b));
        EXPECT_EQ(value.shape(), x.shape());
        EXPECT_EQ(value.dtype(), x.dtype());
        EXPECT_EQ(value.to_vector<double>(), x.to_vector<double>());
      }
}
TEST(ArrayIo, CompressedNpzUsesDeflateAndPreservesAllTypedArrays) {
  const n::NamedTensors values{
      {"layer/weights", n::zeros({2048})},
      {"counter", n::from_vector(std::vector<std::int64_t>{9007199254740993LL}, {})},
      {"empty", n::zeros({2, 0})}};
  const auto bytes = n::encode_npz(values, true);
  ASSERT_GT(bytes.size(), 10u);
  EXPECT_EQ(bytes[8], 8);
  EXPECT_LT(bytes.size(), n::encode_npz(values).size());
  const auto decoded = n::decode_npz(bytes);
  ASSERT_EQ(decoded.size(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(decoded[i].first, values[i].first);
    EXPECT_EQ(n::encode_npy(decoded[i].second), n::encode_npy(values[i].second));
  }
  EXPECT_TRUE(n::decode_npz(n::encode_npz({}, true)).empty());
  EXPECT_THROW(
      n::encode_npz({{"x", n::full({},1.)}, {"x", n::full({},2.)}}, true), std::invalid_argument);
}
TEST(ArrayIo, IdxUnsignedBytesMatchOriginalSpecificationAndReader) {
  const auto path =
      (std::filesystem::temp_directory_path() / "dm-native-array-labels.idx").string();
  n::write_idx(path, n::tensor({7, 3}).to(n::DType::Int32), true);
  std::ifstream                    in(path, std::ios::binary);
  const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(in), {}};
  EXPECT_EQ(bytes, (std::vector<unsigned char>{0, 0, 8, 1, 0, 0, 0, 2, 7, 3}));
  EXPECT_EQ(n::read_idx(path).to_vector<int>(), (std::vector<int>{7, 3}));
  std::filesystem::remove(path);
}
TEST(ArrayIo, IdxTypedWritesAndRejectedInputs) {
  const auto path =
      (std::filesystem::temp_directory_path() / "dm-native-array-values.idx").string();
  for (auto type : {n::DType::Int32, n::DType::Float32, n::DType::Float64}) {
    const auto x = n::tensor({-2., 1., 3., 4.}, {2, 2}, type);
    n::write_idx(path, x);
    const auto y = n::read_idx(path);
    EXPECT_EQ(y.shape(), x.shape());
    EXPECT_EQ(y.dtype(), type);
    EXPECT_EQ(y.to_vector<double>(), x.to_vector<double>());
  }
  EXPECT_THROW(n::write_idx(path, n::tensor({256}), true), std::invalid_argument);
  EXPECT_THROW(n::write_idx(path, n::tensor({.5}), true), std::invalid_argument);
  EXPECT_THROW(n::write_idx(path, n::full({},1.)), std::invalid_argument);
  EXPECT_THROW(n::write_idx(path, n::zeros({2, 0})), std::invalid_argument);
  EXPECT_THROW(n::write_idx(path, n::tensor({1}).to(n::DType::Int64)), std::invalid_argument);
  std::filesystem::remove(path);
}
} // namespace
