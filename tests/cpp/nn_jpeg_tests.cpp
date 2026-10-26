// Tests for the JPEG decoder (datamunge/image/jpeg.hpp), the read_image
// dispatcher and the nn image-folder loader (nn/datasets.hpp).
//
// Fixtures in tests/cpp/data/jpeg/ are a 37x29 crop encoded by libjpeg-turbo's
// cjpeg (4:2:0, 4:4:4, 4:2:2, 4:4:0, 4:1:1, progressive, restart markers,
// grayscale, Adobe RGB) and by macOS sips (Adobe CMYK, ICC profile stripped).
// Expected pixels are FNV-1a hashes of TensorFlow 2.20's DecodeJpeg output
// (tf.io.decode_jpeg), with dct_method "" (INTEGER_FAST) and INTEGER_ACCURATE.

#include <gtest/gtest.h>

#include <cstdint>
#include <datamunge/image/bmp.hpp>
#include <datamunge/image/jpeg.hpp>
#include <datamunge/image/netpbm.hpp>
#include <datamunge/image/png.hpp>
#include <datamunge/image/read_image.hpp>
#include <datamunge/nn/datasets.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace img = datamunge::image;
namespace nn  = datamunge::nn;
namespace fs  = std::filesystem;
using Bytes   = std::vector<std::uint8_t>;

namespace {

fs::path data_dir() { return fs::path(__FILE__).parent_path() / "data" / "jpeg"; }

Bytes load(const std::string& name) {
  std::ifstream in(data_dir() / name, std::ios::binary);
  if (!in) throw std::runtime_error("missing fixture " + (data_dir() / name).string());
  return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::uint64_t fnv1a(const std::uint8_t* p, std::size_t n) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ULL;
  return h;
}

fs::path temp_dir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() / ("datamunge_nn_jpeg_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void write_bytes(const fs::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

constexpr img::JpegDctMethod kFast     = img::JpegDctMethod::IntegerFast;
constexpr img::JpegDctMethod kAccurate = img::JpegDctMethod::IntegerAccurate;

struct TfCase {
  const char* file;
  img::JpegDctMethod dct;
  int channels;  // DecodeJpeg channels attribute
  int width, height, out_channels;
  std::uint64_t hash;
};

// FNV-1a of TF 2.20 DecodeJpeg's uint8 output.
const TfCase kTfCases[] = {
    {"base_411.jpg", kAccurate, 0, 37, 29, 3, 0x49e4e892f8d96abdULL},
    {"base_411.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"base_411.jpg", kAccurate, 3, 37, 29, 3, 0x49e4e892f8d96abdULL},
    {"base_411.jpg", kFast, 0, 37, 29, 3, 0xd9d0f170ff8c6812ULL},
    {"base_411.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"base_411.jpg", kFast, 3, 37, 29, 3, 0xd9d0f170ff8c6812ULL},
    {"base_420.jpg", kAccurate, 0, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"base_420.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"base_420.jpg", kAccurate, 3, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"base_420.jpg", kFast, 0, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"base_420.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"base_420.jpg", kFast, 3, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"base_422.jpg", kAccurate, 0, 37, 29, 3, 0x8b587404c65a0b79ULL},
    {"base_422.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"base_422.jpg", kAccurate, 3, 37, 29, 3, 0x8b587404c65a0b79ULL},
    {"base_422.jpg", kFast, 0, 37, 29, 3, 0x0d16fba4d27bd794ULL},
    {"base_422.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"base_422.jpg", kFast, 3, 37, 29, 3, 0x0d16fba4d27bd794ULL},
    {"base_440.jpg", kAccurate, 0, 37, 29, 3, 0x6821109d3705023bULL},
    {"base_440.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"base_440.jpg", kAccurate, 3, 37, 29, 3, 0x6821109d3705023bULL},
    {"base_440.jpg", kFast, 0, 37, 29, 3, 0x980d8b6937f80bbfULL},
    {"base_440.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"base_440.jpg", kFast, 3, 37, 29, 3, 0x980d8b6937f80bbfULL},
    {"base_444.jpg", kAccurate, 0, 37, 29, 3, 0xf1dea3b1e128f901ULL},
    {"base_444.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"base_444.jpg", kAccurate, 3, 37, 29, 3, 0xf1dea3b1e128f901ULL},
    {"base_444.jpg", kFast, 0, 37, 29, 3, 0xba98ba009e140b39ULL},
    {"base_444.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"base_444.jpg", kFast, 3, 37, 29, 3, 0xba98ba009e140b39ULL},
    {"cmyk.jpg", kAccurate, 0, 37, 29, 3, 0xfdb7d0473d65489fULL},
    {"cmyk.jpg", kAccurate, 3, 37, 29, 3, 0xfdb7d0473d65489fULL},
    {"cmyk.jpg", kFast, 0, 37, 29, 3, 0xa384ff8f006142cdULL},
    {"cmyk.jpg", kFast, 3, 37, 29, 3, 0xa384ff8f006142cdULL},
    {"gray.jpg", kAccurate, 0, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"gray.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"gray.jpg", kAccurate, 3, 37, 29, 3, 0x2f50d97fa9df621fULL},
    {"gray.jpg", kFast, 0, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"gray.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"gray.jpg", kFast, 3, 37, 29, 3, 0x3c69873999cb7e0cULL},
    {"gray_prog.jpg", kAccurate, 0, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"gray_prog.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"gray_prog.jpg", kAccurate, 3, 37, 29, 3, 0x2f50d97fa9df621fULL},
    {"gray_prog.jpg", kFast, 0, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"gray_prog.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"gray_prog.jpg", kFast, 3, 37, 29, 3, 0x3c69873999cb7e0cULL},
    {"prog_420.jpg", kAccurate, 0, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"prog_420.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"prog_420.jpg", kAccurate, 3, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"prog_420.jpg", kFast, 0, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"prog_420.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"prog_420.jpg", kFast, 3, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"rgb.jpg", kAccurate, 0, 37, 29, 3, 0x6c7e1c2eda9fe342ULL},
    {"rgb.jpg", kAccurate, 1, 37, 29, 1, 0x8a8d092d3c051c88ULL},
    {"rgb.jpg", kAccurate, 3, 37, 29, 3, 0x6c7e1c2eda9fe342ULL},
    {"rgb.jpg", kFast, 0, 37, 29, 3, 0x561327ee91bdcfdbULL},
    {"rgb.jpg", kFast, 1, 37, 29, 1, 0x6427c19405335e54ULL},
    {"rgb.jpg", kFast, 3, 37, 29, 3, 0x561327ee91bdcfdbULL},
    {"rst_420.jpg", kAccurate, 0, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"rst_420.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"rst_420.jpg", kAccurate, 3, 37, 29, 3, 0x14f1e35553f2ff16ULL},
    {"rst_420.jpg", kFast, 0, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"rst_420.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"rst_420.jpg", kFast, 3, 37, 29, 3, 0x8b8eb2fd0e34bc6aULL},
    {"rst_prog_422.jpg", kAccurate, 0, 37, 29, 3, 0x8b587404c65a0b79ULL},
    {"rst_prog_422.jpg", kAccurate, 1, 37, 29, 1, 0x8b99dd1367632987ULL},
    {"rst_prog_422.jpg", kAccurate, 3, 37, 29, 3, 0x8b587404c65a0b79ULL},
    {"rst_prog_422.jpg", kFast, 0, 37, 29, 3, 0x0d16fba4d27bd794ULL},
    {"rst_prog_422.jpg", kFast, 1, 37, 29, 1, 0x10b1e9afa76f860eULL},
    {"rst_prog_422.jpg", kFast, 3, 37, 29, 3, 0x0d16fba4d27bd794ULL},
};

const char* const kFixtures[] = {"base_411.jpg", "base_420.jpg", "base_422.jpg", "base_440.jpg", "base_444.jpg",
                                 "cmyk.jpg",     "gray.jpg",     "gray_prog.jpg", "prog_420.jpg", "rgb.jpg",
                                 "rst_420.jpg",  "rst_prog_422.jpg"};

// Inserts an Exif APP1 segment holding only an orientation tag right after SOI.
Bytes with_orientation(const Bytes& jpeg, int orientation, bool little_endian) {
  Bytes tiff = little_endian ? Bytes{'I', 'I', 42, 0, 8, 0, 0, 0} : Bytes{'M', 'M', 0, 42, 0, 0, 0, 8};
  auto u16 = [&](unsigned v) {
    if (little_endian) {
      tiff.push_back(static_cast<std::uint8_t>(v & 255));
      tiff.push_back(static_cast<std::uint8_t>(v >> 8));
    } else {
      tiff.push_back(static_cast<std::uint8_t>(v >> 8));
      tiff.push_back(static_cast<std::uint8_t>(v & 255));
    }
  };
  u16(1);       // one IFD entry
  u16(0x0112);  // Orientation
  u16(3);       // SHORT
  u16(little_endian ? 1 : 0);  // count = 1 (u32)
  u16(little_endian ? 0 : 1);
  u16(static_cast<unsigned>(orientation));
  u16(0);
  tiff.insert(tiff.end(), {0, 0, 0, 0});  // next IFD
  Bytes app{0xFF, 0xE1, 0, 0, 'E', 'x', 'i', 'f', 0, 0};
  app.insert(app.end(), tiff.begin(), tiff.end());
  const std::size_t len = app.size() - 2;
  app[2] = static_cast<std::uint8_t>(len >> 8);
  app[3] = static_cast<std::uint8_t>(len & 255);
  Bytes out(jpeg.begin(), jpeg.begin() + 2);
  out.insert(out.end(), app.begin(), app.end());
  out.insert(out.end(), jpeg.begin() + 2, jpeg.end());
  return out;
}

// Replaces the first SOFn marker code.
Bytes patch_sof(Bytes jpeg, std::uint8_t code) {
  for (std::size_t i = 2; i + 1 < jpeg.size(); ++i)
    if (jpeg[i] == 0xFF && jpeg[i + 1] >= 0xC0 && jpeg[i + 1] <= 0xC2) {
      jpeg[i + 1] = code;
      return jpeg;
    }
  throw std::runtime_error("no SOF marker");
}

}  // namespace

TEST(NnJpegDecode, MatchesTensorFlowBitExactly) {
  for (const auto& c : kTfCases) {
    SCOPED_TRACE(std::string(c.file) + (c.dct == kFast ? " fast" : " accurate") + " channels=" + std::to_string(c.channels));
    img::JpegDecodeOptions options;
    options.dct_method = c.dct;
    options.channels   = c.channels;
    const img::Image image = img::decode_jpeg(load(c.file), options);
    ASSERT_EQ(image.width(), c.width);
    ASSERT_EQ(image.height(), c.height);
    ASSERT_EQ(image.channels(), c.out_channels);
    EXPECT_EQ(fnv1a(image.data().data(), image.data().size()), c.hash);
  }
}

TEST(NnJpegDecode, DefaultsAreTensorFlowDefaults) {
  // DecodeJpeg(channels=0, dct_method="") == decode_jpeg(bytes) with default options.
  const img::Image image = img::decode_jpeg(load("base_420.jpg"));
  EXPECT_EQ(image.mode(), img::ImageMode::RGB);
  EXPECT_EQ(fnv1a(image.data().data(), image.data().size()), 0x8b8eb2fd0e34bc6aULL);
  const img::Image px = img::read_jpeg((data_dir() / "base_420.jpg").string());
  EXPECT_EQ(px.get_pixel(0, 0), (img::Pixel{30, 71, 27, 255}));
  EXPECT_EQ(px.get_pixel(1, 0), (img::Pixel{28, 64, 26, 255}));
  EXPECT_EQ(img::read_jpeg((data_dir() / "gray.jpg").string()).mode(), img::ImageMode::Grayscale);
}

TEST(NnJpegDecode, ProgressiveAndRestartMatchBaseline) {
  // Same quantization tables -> same coefficients -> identical pixels.
  for (auto dct : {kFast, kAccurate}) {
    img::JpegDecodeOptions o;
    o.dct_method = dct;
    EXPECT_EQ(img::decode_jpeg(load("prog_420.jpg"), o).data(), img::decode_jpeg(load("base_420.jpg"), o).data());
    EXPECT_EQ(img::decode_jpeg(load("rst_420.jpg"), o).data(), img::decode_jpeg(load("base_420.jpg"), o).data());
    EXPECT_EQ(img::decode_jpeg(load("gray_prog.jpg"), o).data(), img::decode_jpeg(load("gray.jpg"), o).data());
    EXPECT_EQ(img::decode_jpeg(load("rst_prog_422.jpg"), o).data(), img::decode_jpeg(load("base_422.jpg"), o).data());
  }
}

TEST(NnJpegDecode, SubsamplingVariantsStayClose) {
  // Every chroma layout reconstructs the same picture to within JPEG error (cmyk.jpg went
  // through an ICC conversion, so its colors legitimately differ).
  const img::Image ref = img::decode_jpeg(load("base_444.jpg"));
  for (const char* name : {"base_420.jpg", "base_422.jpg", "base_440.jpg", "base_411.jpg", "rgb.jpg"}) {
    SCOPED_TRACE(name);
    const img::Image other = img::decode_jpeg(load(name));
    ASSERT_EQ(other.data().size(), ref.data().size());
    double total = 0;
    for (std::size_t i = 0; i < ref.data().size(); ++i) total += std::abs(int(other.data()[i]) - int(ref.data()[i]));
    EXPECT_LT(total / static_cast<double>(ref.data().size()), 12.0);
  }
}

TEST(NnJpegDecode, PlainUpsamplingReplicatesChroma) {
  img::JpegDecodeOptions o;
  o.fancy_upsampling = false;
  const img::Image plain = img::decode_jpeg(load("base_444.jpg"), o);
  EXPECT_EQ(plain.data(), img::decode_jpeg(load("base_444.jpg")).data());  // nothing to upsample
  const img::Image sub = img::decode_jpeg(load("base_420.jpg"), o);
  EXPECT_NE(sub.data(), img::decode_jpeg(load("base_420.jpg")).data());
}

TEST(NnJpegDecode, HeaderInfo) {
  const Bytes base = load("base_420.jpg");
  img::JpegInfo info = img::jpeg_info(base.data(), base.size());
  EXPECT_EQ(info.width, 37);
  EXPECT_EQ(info.height, 29);
  EXPECT_EQ(info.components, 3);
  EXPECT_FALSE(info.progressive);
  EXPECT_TRUE(info.jfif);
  EXPECT_EQ(info.color_space, img::JpegColorSpace::YCbCr);
  EXPECT_EQ(info.h_sampling, (std::vector<int>{2, 1, 1}));
  EXPECT_EQ(info.v_sampling, (std::vector<int>{2, 1, 1}));
  EXPECT_EQ(info.orientation, 1);
  EXPECT_TRUE(img::read_jpeg_info((data_dir() / "prog_420.jpg").string()).progressive);
  EXPECT_EQ(img::read_jpeg_info((data_dir() / "rst_420.jpg").string()).restart_interval, 1);
  EXPECT_EQ(img::read_jpeg_info((data_dir() / "gray.jpg").string()).color_space, img::JpegColorSpace::Grayscale);
  info = img::read_jpeg_info((data_dir() / "rgb.jpg").string());
  EXPECT_EQ(info.color_space, img::JpegColorSpace::RGB);
  EXPECT_TRUE(info.adobe);
  EXPECT_EQ(info.adobe_transform, 0);
  info = img::read_jpeg_info((data_dir() / "cmyk.jpg").string());
  EXPECT_EQ(info.components, 4);
  EXPECT_TRUE(info.adobe);
  EXPECT_TRUE(info.color_space == img::JpegColorSpace::CMYK || info.color_space == img::JpegColorSpace::YCCK);
  // decode_jpeg fills the same info.
  img::JpegInfo decoded;
  (void)img::decode_jpeg(load("base_422.jpg"), {}, &decoded);
  EXPECT_EQ(decoded.h_sampling, (std::vector<int>{2, 1, 1}));
  EXPECT_EQ(decoded.v_sampling, (std::vector<int>{1, 1, 1}));
}

TEST(NnJpegDecode, ExifOrientationIsReportedNotApplied) {
  const Bytes base = load("base_420.jpg");
  for (bool le : {true, false}) {
    const Bytes tagged = with_orientation(base, 6, le);
    img::JpegInfo info;
    const img::Image image = img::decode_jpeg(tagged, {}, &info);
    EXPECT_EQ(info.orientation, 6);
    EXPECT_EQ(img::jpeg_info(tagged.data(), tagged.size()).orientation, 6);
    EXPECT_EQ(image.data(), img::decode_jpeg(base).data());  // pixels untouched, like TF
  }
}

TEST(NnJpegDecode, ApplyExifOrientation) {
  img::Image a(3, 2, img::ImageMode::Grayscale);  // 0 1 2 / 3 4 5
  for (int i = 0; i < 6; ++i) a.data()[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
  auto flat = [](const img::Image& m) { return std::vector<int>(m.data().begin(), m.data().end()); };
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 1)), (std::vector<int>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 2)), (std::vector<int>{2, 1, 0, 5, 4, 3}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 3)), (std::vector<int>{5, 4, 3, 2, 1, 0}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 4)), (std::vector<int>{3, 4, 5, 0, 1, 2}));
  const img::Image r6 = img::apply_exif_orientation(a, 6);  // rotate 90 clockwise
  EXPECT_EQ(r6.width(), 2);
  EXPECT_EQ(r6.height(), 3);
  EXPECT_EQ(flat(r6), (std::vector<int>{3, 0, 4, 1, 5, 2}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 8)), (std::vector<int>{2, 5, 1, 4, 0, 3}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 5)), (std::vector<int>{0, 3, 1, 4, 2, 5}));
  EXPECT_EQ(flat(img::apply_exif_orientation(a, 7)), (std::vector<int>{5, 2, 4, 1, 3, 0}));
  EXPECT_THROW((void)img::apply_exif_orientation(a, 9), std::invalid_argument);
}

TEST(NnJpegErrors, TruncatedInputThrows) {
  for (const char* name : kFixtures) {
    SCOPED_TRACE(name);
    const Bytes full = load(name);
    for (std::size_t cut : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{20}, full.size() / 4,
                            full.size() / 2, full.size() * 3 / 4, full.size() - 3, full.size() - 2, full.size() - 1}) {
      const Bytes part(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(cut));
      EXPECT_THROW((void)img::decode_jpeg(part), std::runtime_error) << "cut at " << cut;
    }
  }
}

TEST(NnJpegErrors, CorruptInputThrowsOrDecodesButNeverCrashes) {
  std::uint64_t state = 12345;
  auto next = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::size_t>(state >> 33);
  };
  int thrown = 0, decoded = 0;
  for (const char* name : kFixtures) {
    const Bytes base = load(name);
    for (int trial = 0; trial < 60; ++trial) {
      Bytes b = base;
      const std::size_t flips = 1 + next() % 6;
      for (std::size_t f = 0; f < flips; ++f) b[next() % b.size()] = static_cast<std::uint8_t>(next());
      try {
        const img::Image image = img::decode_jpeg(b);
        EXPECT_EQ(image.data().size(), static_cast<std::size_t>(image.width()) * image.height() * image.channels());
        ++decoded;
      } catch (const std::runtime_error&) {
        ++thrown;
      }
    }
  }
  EXPECT_GT(thrown, 0);
  EXPECT_GT(decoded, 0);
}

TEST(NnJpegErrors, UnsupportedAndInvalid) {
  const Bytes base = load("base_420.jpg");
  EXPECT_FALSE(img::is_jpeg(reinterpret_cast<const std::uint8_t*>("\x89PNG"), 4));
  EXPECT_TRUE(img::is_jpeg(base.data(), base.size()));
  EXPECT_THROW((void)img::decode_jpeg(Bytes{0x89, 'P', 'N', 'G', 13, 10, 26, 10}), std::runtime_error);
  EXPECT_THROW((void)img::decode_jpeg(patch_sof(base, 0xC3)), std::runtime_error);  // lossless
  EXPECT_THROW((void)img::decode_jpeg(patch_sof(base, 0xC9)), std::runtime_error);  // arithmetic
  // Header-only stream (no scan) and a stream cut right before EOI.
  Bytes header_only(base.begin(), base.begin() + 2);
  header_only.insert(header_only.end(), {0xFF, 0xD9});
  EXPECT_THROW((void)img::decode_jpeg(header_only), std::runtime_error);
  img::JpegDecodeOptions two;
  two.channels = 2;
  EXPECT_THROW((void)img::decode_jpeg(base, two), std::invalid_argument);
  img::JpegDecodeOptions gray;
  gray.channels = 1;
  EXPECT_THROW((void)img::decode_jpeg(load("cmyk.jpg"), gray), std::runtime_error);  // TF fails too
  EXPECT_THROW((void)img::read_jpeg((data_dir() / "missing.jpg").string()), std::runtime_error);
  // A header claiming a huge image backed by a few bytes is rejected before allocating.
  Bytes huge = base;
  for (std::size_t i = 2; i + 8 < huge.size(); ++i)
    if (huge[i] == 0xFF && huge[i + 1] == 0xC0) {
      huge[i + 5] = huge[i + 6] = huge[i + 7] = huge[i + 8] = 0xFF;  // 65535 x 65535
      break;
    }
  EXPECT_THROW((void)img::decode_jpeg(huge), std::runtime_error);
}

TEST(NnJpegReadImage, DispatchesOnSignature) {
  const fs::path dir = temp_dir("read_image");
  img::Image rgb(5, 3, img::ImageMode::RGB, img::Pixel{10, 200, 30, 255});
  rgb.set_pixel(1, 2, img::Pixel{1, 2, 3, 255});
  img::write_png(rgb, (dir / "a.png").string());
  img::write_bmp(rgb, (dir / "a.bmp").string());
  img::write_ppm(rgb, (dir / "a.ppm").string());
  img::write_pgm(rgb, (dir / "a.pgm").string());
  EXPECT_EQ(img::read_image((dir / "a.png").string()).data(), rgb.data());
  EXPECT_EQ(img::read_image((dir / "a.bmp").string()).data(), rgb.data());
  EXPECT_EQ(img::read_image((dir / "a.ppm").string()).data(), rgb.data());
  EXPECT_EQ(img::read_image((dir / "a.pgm").string()).data(), img::read_pgm((dir / "a.pgm").string()).data());
  // The extension is irrelevant: a JPEG named .png decodes as JPEG.
  write_bytes(dir / "photo.png", load("base_420.jpg"));
  EXPECT_EQ(img::read_image((dir / "photo.png").string()).data(), img::decode_jpeg(load("base_420.jpg")).data());
  write_bytes(dir / "junk.bin", Bytes{1, 2, 3, 4, 5, 6, 7, 8, 9});
  EXPECT_THROW((void)img::read_image((dir / "junk.bin").string()), std::runtime_error);
  fs::remove_all(dir);
}

TEST(NnImageFolder, ScanOrderLabelsAndErrors) {
  const fs::path root = temp_dir("folder");
  for (const char* d : {"zebra", "ant", "mole"}) fs::create_directories(root / d);
  write_bytes(root / "zebra" / "b.jpg", load("base_420.jpg"));
  write_bytes(root / "zebra" / "a.JPEG", load("gray.jpg"));
  img::write_png(img::Image(2, 2, img::ImageMode::RGBA, img::Pixel{9, 8, 7, 1}), (root / "ant" / "x.png").string());
  write_bytes(root / "ant" / "notes.txt", Bytes{'h', 'i'});
  const nn::ImageFolder folder = nn::scan_image_folder(root.string());
  EXPECT_EQ(folder.class_names, (std::vector<std::string>{"ant", "mole", "zebra"}));
  ASSERT_EQ(folder.images.size(), 3u);
  EXPECT_EQ(fs::path(folder.images[0].path).filename(), "x.png");
  EXPECT_EQ(folder.images[0].label, 0);
  EXPECT_EQ(fs::path(folder.images[1].path).filename(), "a.JPEG");
  EXPECT_EQ(folder.images[1].label, 2);
  EXPECT_EQ(folder.images[2].label, 2);
  // Training class names keep labels stable when a split lacks a class.
  const nn::ImageFolder same = nn::scan_image_folder(root.string(), {"ant", "mole", "zebra", "yak"});
  EXPECT_EQ(same.images[1].label, 2);
  EXPECT_THROW(nn::scan_image_folder(root.string(), {"ant", "zebra"}), std::invalid_argument);  // unknown "mole"
  EXPECT_THROW(nn::scan_image_folder(root.string(), {"ant", "ant", "mole", "zebra"}), std::invalid_argument);
  EXPECT_THROW(nn::scan_image_folder((root / "missing").string()), std::invalid_argument);
  const fs::path empty = temp_dir("folder_empty");
  fs::create_directories(empty / "cls");
  EXPECT_THROW(nn::scan_image_folder(empty.string()), std::invalid_argument);
  fs::remove_all(empty);

  const nn::ImageDataset all = nn::load_image_folder(root.string(), 6, 4);
  EXPECT_EQ(all.images.shape(), (nn::Shape{3, 6, 4, 3}));
  EXPECT_EQ(all.labels.dtype(), nn::DType::Int64);
  const std::int64_t* lab = all.labels.data_ptr<std::int64_t>();
  EXPECT_EQ(lab[0], 0);
  EXPECT_EQ(lab[2], 2);
  const float* px = all.images.data_ptr<float>();
  EXPECT_FLOAT_EQ(px[0], 9.f / 255.f);  // RGBA PNG: alpha dropped
  EXPECT_FLOAT_EQ(px[2], 7.f / 255.f);
  const float* g = px + 6 * 4 * 3;  // grayscale JPEG expands to equal RGB
  EXPECT_EQ(g[0], g[1]);
  EXPECT_EQ(g[1], g[2]);

  const Bytes jpeg = load("base_420.jpg");
  write_bytes(root / "zebra" / "c.jpg", Bytes(jpeg.begin(), jpeg.begin() + 300));
  const nn::ImageFolder broken = nn::scan_image_folder(root.string());
  EXPECT_THROW(nn::load_image_batch(broken.images, {0, 1, 2, 3}, 4, 4), std::runtime_error);
  EXPECT_THROW(nn::load_image_rgb((root / "ant" / "notes.txt").string(), 4, 4), std::invalid_argument);
  EXPECT_THROW(nn::load_image_batch(broken.images, {9}, 4, 4), std::out_of_range);
  fs::remove_all(root);
}

TEST(NnImageFolder, ResizeMatchesTensorFlow) {
  // TF: DecodeJpeg(channels=3) -> Cast float -> ResizeBilinear(half_pixel_centers) -> / 255
  // (a grappler-optimized session, which multiplies by 1/255), 8x6 output: FNV-1a of the float bits.
  const fs::path dir = temp_dir("resize");
  write_bytes(dir / "base_420.jpg", load("base_420.jpg"));
  const nn::Tensor t = nn::load_image_rgb((dir / "base_420.jpg").string(), 8, 6);
  ASSERT_EQ(t.shape(), (nn::Shape{8, 6, 3}));
  const float* p = t.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 0.06278594583272934f);
  EXPECT_FLOAT_EQ(p[1], 0.05077614262700081f);
  EXPECT_FLOAT_EQ(p[2], 0.011008987203240395f);
  EXPECT_EQ(fnv1a(reinterpret_cast<const std::uint8_t*>(p), 8 * 6 * 3 * sizeof(float)), 0xc4094e38f776787eULL);
  fs::remove_all(dir);
}
