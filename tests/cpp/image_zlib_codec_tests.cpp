#include <gtest/gtest.h>

#include <datamunge/image/detail/adler32.hpp>
#include <datamunge/image/detail/crc32.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>

#include <random>
#include <string>

using namespace datamunge::image::detail;

namespace {
std::vector<unsigned char> to_bytes(const std::string& s) { return std::vector<unsigned char>(s.begin(), s.end()); }
} // namespace

TEST(Crc32, MatchesThePublishedIsoHdlcTestVector) {
    const std::string check = "123456789";
    EXPECT_EQ(crc32_update(0, reinterpret_cast<const unsigned char*>(check.data()), check.size()), 0xCBF43926u);
}

TEST(Crc32, IncrementalUpdateMatchesAWholeBufferComputation) {
    const std::string a = "Hello, ";
    const std::string b = "world!";
    const std::uint32_t incremental = crc32_update(crc32_update(0, reinterpret_cast<const unsigned char*>(a.data()), a.size()),
                                                     reinterpret_cast<const unsigned char*>(b.data()), b.size());
    const std::string whole = a + b;
    const std::uint32_t direct = crc32_update(0, reinterpret_cast<const unsigned char*>(whole.data()), whole.size());
    EXPECT_EQ(incremental, direct);
}

TEST(Adler32, MatchesAPublishedTestVector) {
    const std::string wiki = "Wikipedia";
    EXPECT_EQ(adler32(reinterpret_cast<const unsigned char*>(wiki.data()), wiki.size()), 0x11E60398u);
}

TEST(Adler32, EmptyInputIsOne) { EXPECT_EQ(adler32(nullptr, 0), 1u); }

class ZlibCodecRoundTrip : public ::testing::TestWithParam<std::string> {};

TEST_P(ZlibCodecRoundTrip, CompressThenDecompressReproducesTheOriginalExactly) {
    const auto original = to_bytes(GetParam());
    const auto compressed = zlib_compress(original);
    const auto decompressed = zlib_decompress(compressed, original.size());
    EXPECT_EQ(decompressed, original);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, ZlibCodecRoundTrip,
                          ::testing::Values("", "x", std::string(200000, 'a'), std::string("the quick brown fox ").substr(0, 21)));

TEST(ZlibCodec, RoundTripsPseudoRandomBinaryData) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<unsigned char> original(50000);
    for (auto& b : original) b = static_cast<unsigned char>(byte_dist(rng));

    const auto compressed = zlib_compress(original);
    const auto decompressed = zlib_decompress(compressed, original.size());
    EXPECT_EQ(decompressed, original);
}

TEST(ZlibCodec, DataLargerThanOneStoredBlockRoundTrips) {
    // A single stored block is capped at 65535 bytes -- this exercises the multi-block chunking
    // path in zlib_compress() (and the multi-block loop in zlib_decompress()).
    std::vector<unsigned char> original(200000);
    for (std::size_t i = 0; i < original.size(); ++i) original[i] = static_cast<unsigned char>(i % 251);

    const auto compressed = zlib_compress(original);
    const auto decompressed = zlib_decompress(compressed, original.size());
    EXPECT_EQ(decompressed, original);
}

TEST(ZlibCodec, DecompressRejectsATruncatedStream) {
    const auto original = to_bytes(std::string(1000, 'z'));
    auto compressed = zlib_compress(original);
    compressed.resize(compressed.size() / 2); // chop it in half
    EXPECT_THROW(zlib_decompress(compressed, original.size()), std::runtime_error);
}

TEST(ZlibCodec, DecompressRejectsACorruptedHeader) {
    auto compressed = zlib_compress(to_bytes("hello"));
    compressed[0] = 0xFF; // corrupt the CMF byte (compression method nibble)
    EXPECT_THROW(zlib_decompress(compressed, 5), std::runtime_error);
}

TEST(ZlibCodec, DecompressRejectsAnAdler32Mismatch) {
    auto compressed = zlib_compress(to_bytes("hello world"));
    compressed.back() ^= 0xFF; // flip a bit in the Adler-32 trailer
    EXPECT_THROW(zlib_decompress(compressed, 11), std::runtime_error);
}

TEST(ZlibCodec, DecompressRejectsAWrongExpectedSize) {
    const auto compressed = zlib_compress(to_bytes("hello world"));
    EXPECT_THROW(zlib_decompress(compressed, 999), std::runtime_error);
}
