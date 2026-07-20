#include <gtest/gtest.h>

#include <datamunge/image/bmp.hpp>
#include <datamunge/image/netpbm.hpp>
#include <datamunge/image/png.hpp>

#include <cstdio>
#include <filesystem>
#include <stdexcept>

using namespace datamunge::image;

namespace {

class ImageIOTest : public ::testing::Test {
  protected:
    void SetUp() override { dir_ = std::filesystem::temp_directory_path() / "datamunge_image_io_tests"; std::filesystem::create_directories(dir_); }
    void TearDown() override { std::filesystem::remove_all(dir_); }

    [[nodiscard]] std::string path(const std::string& name) const { return (dir_ / name).string(); }

    [[nodiscard]] static Image make_gradient(int width, int height, ImageMode mode = ImageMode::RGB) {
        Image img(width, height, mode);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const auto r = static_cast<std::uint8_t>((x * 37) % 256);
                const auto g = static_cast<std::uint8_t>((y * 53) % 256);
                const auto b = static_cast<std::uint8_t>((x + y) % 256);
                const auto a = static_cast<std::uint8_t>(255 - (x % 256));
                img.set_pixel(x, y, Pixel{r, g, b, a});
            }
        }
        return img;
    }

    std::filesystem::path dir_;
};

} // namespace

// ---- PPM ----

TEST_F(ImageIOTest, PpmRoundTripsAnRgbGradientExactly) {
    const Image img = make_gradient(9, 7, ImageMode::RGB);
    const std::string p = path("gradient.ppm");
    write_ppm(img, p);
    const Image back = read_ppm(p);
    ASSERT_EQ(back.width(), img.width());
    ASSERT_EQ(back.height(), img.height());
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, PpmWriteConvertsNonRgbSourcesAutomatically) {
    const Image gray(4, 4, ImageMode::Grayscale, Pixel{100, 0, 0, 255});
    const std::string p = path("from_gray.ppm");
    write_ppm(gray, p);
    const Image back = read_ppm(p);
    EXPECT_EQ(back.mode(), ImageMode::RGB);
    EXPECT_EQ(back.get_pixel(0, 0), (Pixel{100, 100, 100, 255}));
}

TEST_F(ImageIOTest, ReadPpmOnMissingFileThrows) { EXPECT_THROW(read_ppm(path("does_not_exist.ppm")), std::runtime_error); }

// ---- PGM ----

TEST_F(ImageIOTest, PgmRoundTripsAGrayscaleImageExactly) {
    Image img(10, 6, ImageMode::Grayscale);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 10; ++x) img.set_pixel(x, y, Pixel{static_cast<std::uint8_t>((x * 20 + y) % 256), 0, 0, 255});
    const std::string p = path("gray.pgm");
    write_pgm(img, p);
    const Image back = read_pgm(p);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 10; ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, ReadPgmOnMissingFileThrows) { EXPECT_THROW(read_pgm(path("does_not_exist.pgm")), std::runtime_error); }

// ---- PBM ----

TEST_F(ImageIOTest, PbmRoundTripsABlackAndWhitePatternWithNonByteAlignedWidth) {
    // Width 13 forces a padded final byte in every row -- exercises the padding logic.
    Image img(13, 5, ImageMode::RGB, Pixel{255, 255, 255, 255});
    for (int i = 0; i < 5; ++i) img.set_pixel(i, i, Pixel{0, 0, 0, 255});
    const std::string p = path("diag.pbm");
    write_pbm(img, p);
    const Image back = read_pbm(p);
    ASSERT_EQ(back.width(), 13);
    ASSERT_EQ(back.height(), 5);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(back.get_pixel(i, i).r, 0);
    EXPECT_EQ(back.get_pixel(12, 4).r, 255);
}

TEST_F(ImageIOTest, ReadPbmOnMissingFileThrows) { EXPECT_THROW(read_pbm(path("does_not_exist.pbm")), std::runtime_error); }

// ---- BMP ----

TEST_F(ImageIOTest, BmpRoundTripsAnRgbGradientExactly) {
    const Image img = make_gradient(11, 8, ImageMode::RGB);
    const std::string p = path("gradient.bmp");
    write_bmp(img, p);
    const Image back = read_bmp(p);
    ASSERT_EQ(back.width(), img.width());
    ASSERT_EQ(back.height(), img.height());
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, BmpHandlesOddWidthRowPadding) {
    // Width 7 at 3 bytes/pixel = 21 bytes/row, which needs 3 bytes of padding to reach a
    // multiple of 4 -- a width evenly divisible by 4 pixels would never exercise this path.
    const Image img = make_gradient(7, 3, ImageMode::RGB);
    const std::string p = path("odd_width.bmp");
    write_bmp(img, p);
    const Image back = read_bmp(p);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 7; ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, ReadBmpOnMissingFileThrows) { EXPECT_THROW(read_bmp(path("does_not_exist.bmp")), std::runtime_error); }

TEST_F(ImageIOTest, ReadBmpOnBadMagicThrows) {
    const std::string p = path("not_a_bmp.bmp");
    FILE* f = std::fopen(p.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("not a bitmap file", f);
    std::fclose(f);
    EXPECT_THROW(read_bmp(p), std::runtime_error);
}

// ---- PNG ----

TEST_F(ImageIOTest, PngRoundTripsAnRgbaGradientExactly) {
    const Image img = make_gradient(15, 11, ImageMode::RGBA);
    const std::string p = path("gradient.png");
    write_png(img, p);
    const Image back = read_png(p);
    ASSERT_EQ(back.mode(), ImageMode::RGBA);
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, PngPreservesEachOfTheFourSupportedModes) {
    for (const ImageMode mode : {ImageMode::Grayscale, ImageMode::GrayscaleAlpha, ImageMode::RGB, ImageMode::RGBA}) {
        const Image img = make_gradient(6, 5, mode);
        const std::string p = path("mode_test.png");
        write_png(img, p);
        const Image back = read_png(p);
        EXPECT_EQ(back.mode(), mode);
        for (int y = 0; y < img.height(); ++y)
            for (int x = 0; x < img.width(); ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
    }
}

TEST_F(ImageIOTest, PngRoundTripsASolidColorImage) {
    // Exercises filter reconstruction on maximally-flat (all-identical-byte) scanlines.
    const Image img(30, 30, ImageMode::RGB, Pixel{77, 88, 99, 255});
    const std::string p = path("solid.png");
    write_png(img, p);
    const Image back = read_png(p);
    for (int y = 0; y < 30; ++y)
        for (int x = 0; x < 30; ++x) EXPECT_EQ(back.get_pixel(x, y), img.get_pixel(x, y));
}

TEST_F(ImageIOTest, ReadPngOnMissingFileThrows) { EXPECT_THROW(read_png(path("does_not_exist.png")), std::runtime_error); }

TEST_F(ImageIOTest, ReadPngOnBadSignatureThrows) {
    const std::string p = path("not_a_png.png");
    FILE* f = std::fopen(p.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("not a png file at all", f);
    std::fclose(f);
    EXPECT_THROW(read_png(p), std::runtime_error);
}

TEST_F(ImageIOTest, ReadPngOnCorruptedChunkThrows) {
    const Image img(10, 10, ImageMode::RGB, Pixel{1, 2, 3, 255});
    const std::string p = path("corrupt.png");
    write_png(img, p);

    // Flip a byte inside the IDAT payload (well past the fixed-size signature+IHDR region) to
    // trigger a CRC mismatch.
    FILE* f = std::fopen(p.c_str(), "r+b");
    ASSERT_NE(f, nullptr);
    std::fseek(f, 40, SEEK_SET);
    const int original = std::fgetc(f);
    std::fseek(f, 40, SEEK_SET);
    std::fputc(original ^ 0xFF, f);
    std::fclose(f);

    EXPECT_THROW(read_png(p), std::runtime_error);
}

TEST_F(ImageIOTest, WritePngRejectsEmptyImage) {
    const Image empty;
    EXPECT_THROW(write_png(empty, path("empty.png")), std::invalid_argument);
}
