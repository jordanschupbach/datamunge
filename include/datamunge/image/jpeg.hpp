#pragma once

/// @file jpeg.hpp
/// @brief Self-contained JPEG (JFIF/Exif/Adobe) decoder.
///
/// Supports baseline and extended sequential DCT (SOF0/SOF1) and progressive DCT (SOF2) with
/// Huffman coding and 8-bit samples; grayscale, YCbCr, RGB (Adobe transform 0), CMYK and
/// YCCK; any integral chroma subsampling (4:4:4, 4:2:2, 4:4:0, 4:2:0, 4:1:1, ...); restart
/// markers. Arithmetic coding, lossless, hierarchical and 12-bit JPEGs throw.
///
/// Pixels match libjpeg-turbo bit for bit: the integer IDCTs are libjpeg's jidctfst.c
/// (IntegerFast) and jidctint.c (IntegerAccurate), chroma upsampling is libjpeg's "fancy"
/// triangle filter (h2v1/h1v2/h2v2; plain replication for other ratios) and the color
/// conversion uses libjpeg's fixed-point tables. The defaults reproduce
/// tf.io.decode_jpeg / tensorflow::ops::DecodeJpeg, whose default dct_method "" means
/// INTEGER_FAST.
///
/// Corrupt or truncated input (including a missing EOI marker, which TensorFlow also rejects)
/// throws std::runtime_error and never reads out of bounds. Like libjpeg, entropy-coded data
/// that stops early at a marker is zero-filled rather than rejected.

#include <datamunge/image/image.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::image {

/// @brief Inverse DCT variant (libjpeg's JDCT_IFAST / JDCT_ISLOW).
enum class JpegDctMethod { IntegerFast, IntegerAccurate };

/// @brief Color space of the encoded components (libjpeg's jpeg_color_space guess).
enum class JpegColorSpace { Grayscale, YCbCr, RGB, CMYK, YCCK };

/// @brief Decoding options; the defaults match TensorFlow's DecodeJpeg(channels=0).
struct JpegDecodeOptions {
  /// Output channels: 0 = the file's natural count (1 for grayscale, otherwise 3),
  /// 1 = grayscale (luma for YCbCr), 3 = RGB.
  int channels = 0;
  /// IDCT; TensorFlow's default is IntegerFast, djpeg's default is IntegerAccurate.
  JpegDctMethod dct_method = JpegDctMethod::IntegerFast;
  /// libjpeg's do_fancy_upsampling (triangle-filter chroma); false replicates samples.
  bool fancy_upsampling = true;
};

/// @brief Header-level facts about a JPEG stream.
struct JpegInfo {
  int width = 0;
  int height = 0;
  int components = 0;                       ///< 1, 3 or 4
  bool progressive = false;                 ///< SOF2
  JpegColorSpace color_space = JpegColorSpace::YCbCr;
  std::vector<int> h_sampling;              ///< per-component horizontal sampling factors
  std::vector<int> v_sampling;              ///< per-component vertical sampling factors
  int restart_interval = 0;                 ///< MCUs between restart markers (last DRI seen)
  bool jfif = false;                        ///< APP0 JFIF marker present
  bool adobe = false;                       ///< APP14 Adobe marker present
  int adobe_transform = -1;                 ///< Adobe transform flag, -1 when absent
  int orientation = 1;                      ///< EXIF orientation 1..8 (1 when absent); not applied
};

/// @brief True when @p data starts with the JPEG SOI marker (FF D8 FF).
[[nodiscard]] bool is_jpeg(const std::uint8_t* data, std::size_t size);

/// @brief Parses the markers up to the first scan and returns the stream's header facts.
///        Throws std::runtime_error when the header is malformed.
[[nodiscard]] JpegInfo jpeg_info(const std::uint8_t* data, std::size_t size);

/// @brief jpeg_info() of the file at @p path.
[[nodiscard]] JpegInfo read_jpeg_info(const std::string& path);

/// @brief Decodes a JPEG held in memory to a Grayscale or RGB Image (see JpegDecodeOptions).
///        Fills @p info when non-null. Throws std::runtime_error on corrupt, truncated or
///        unsupported input.
[[nodiscard]] Image decode_jpeg(const std::uint8_t* data, std::size_t size, const JpegDecodeOptions& options = {},
                                JpegInfo* info = nullptr);

/// @brief decode_jpeg() of a byte vector.
[[nodiscard]] Image decode_jpeg(const std::vector<std::uint8_t>& bytes, const JpegDecodeOptions& options = {},
                                JpegInfo* info = nullptr);

/// @brief Reads and decodes the JPEG file at @p path (see decode_jpeg()).
[[nodiscard]] Image read_jpeg(const std::string& path, const JpegDecodeOptions& options = {}, JpegInfo* info = nullptr);

/// @brief Returns @p img transformed so that it displays upright for EXIF orientation
///        @p orientation (1 = unchanged, 2 = mirror, 3 = rotate 180, 4 = flip vertically,
///        5 = transpose, 6 = rotate 90 clockwise, 7 = transverse, 8 = rotate 90
///        counter-clockwise). Values outside 1..8 throw std::invalid_argument.
[[nodiscard]] Image apply_exif_orientation(const Image& img, int orientation);

}  // namespace datamunge::image
