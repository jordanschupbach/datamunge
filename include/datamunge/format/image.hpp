#pragma once

/// @file image.hpp
/// @brief std::format / std::print support for datamunge::image.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Images print a one-line header (`Image 640x480 RGB`) and never dump pixel data.
///
/// @par Example
/// @code
/// const dm::image::Image img(640, 480, dm::image::ImageMode::RGB);
/// std::print("{}\n", img);   // Image 640x480 RGB
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/image/imaging.hpp>

#include <format>
#include <string>

/// @brief Prints an ImageMode as its enumerator name (Grayscale, GrayscaleAlpha, RGB, RGBA).
template <>
struct std::formatter<datamunge::image::ImageMode> : datamunge::format_detail::text_formatter<datamunge::image::ImageMode> {
  /// @brief The enumerator name, or "ImageMode(<int>)" for an unknown value.
  /// @param mode The mode to print.
  /// @return The enumerator name.
  static std::string text(datamunge::image::ImageMode mode) {
    using datamunge::image::ImageMode;
    switch (mode) {
      case ImageMode::Grayscale: return "Grayscale";
      case ImageMode::GrayscaleAlpha: return "GrayscaleAlpha";
      case ImageMode::RGB: return "RGB";
      case ImageMode::RGBA: return "RGBA";
    }
    return std::format("ImageMode({})", static_cast<int>(mode));
  }
};

/// @brief Prints a JpegDctMethod as its enumerator name (IntegerFast, IntegerAccurate).
template <>
struct std::formatter<datamunge::image::JpegDctMethod>
    : datamunge::format_detail::text_formatter<datamunge::image::JpegDctMethod> {
  /// @brief The enumerator name, or "JpegDctMethod(<int>)" for an unknown value.
  /// @param method The method to print.
  /// @return The enumerator name.
  static std::string text(datamunge::image::JpegDctMethod method) {
    using datamunge::image::JpegDctMethod;
    switch (method) {
      case JpegDctMethod::IntegerFast: return "IntegerFast";
      case JpegDctMethod::IntegerAccurate: return "IntegerAccurate";
    }
    return std::format("JpegDctMethod({})", static_cast<int>(method));
  }
};

/// @brief Prints a JpegColorSpace as its enumerator name (Grayscale, YCbCr, RGB, CMYK, YCCK).
template <>
struct std::formatter<datamunge::image::JpegColorSpace>
    : datamunge::format_detail::text_formatter<datamunge::image::JpegColorSpace> {
  /// @brief The enumerator name, or "JpegColorSpace(<int>)" for an unknown value.
  /// @param space The color space to print.
  /// @return The enumerator name.
  static std::string text(datamunge::image::JpegColorSpace space) {
    using datamunge::image::JpegColorSpace;
    switch (space) {
      case JpegColorSpace::Grayscale: return "Grayscale";
      case JpegColorSpace::YCbCr: return "YCbCr";
      case JpegColorSpace::RGB: return "RGB";
      case JpegColorSpace::CMYK: return "CMYK";
      case JpegColorSpace::YCCK: return "YCCK";
    }
    return std::format("JpegColorSpace({})", static_cast<int>(space));
  }
};

/// @brief Prints a ResampleFilter as its enumerator name (Nearest, Bilinear).
template <>
struct std::formatter<datamunge::image::ResampleFilter>
    : datamunge::format_detail::text_formatter<datamunge::image::ResampleFilter> {
  /// @brief The enumerator name, or "ResampleFilter(<int>)" for an unknown value.
  /// @param filter The filter to print.
  /// @return The enumerator name.
  static std::string text(datamunge::image::ResampleFilter filter) {
    using datamunge::image::ResampleFilter;
    switch (filter) {
      case ResampleFilter::Nearest: return "Nearest";
      case ResampleFilter::Bilinear: return "Bilinear";
    }
    return std::format("ResampleFilter({})", static_cast<int>(filter));
  }
};

/// @brief Prints a Pixel as "Pixel(r, g, b, a)".
template <>
struct std::formatter<datamunge::image::Pixel> : datamunge::format_detail::text_formatter<datamunge::image::Pixel> {
  /// @brief The four channel values as integers.
  /// @param p The pixel to print.
  /// @return "Pixel(r, g, b, a)".
  static std::string text(const datamunge::image::Pixel& p) {
    return std::format("Pixel({}, {}, {}, {})", int{p.r}, int{p.g}, int{p.b}, int{p.a});
  }
};

/// @brief Prints an HSV color as "HSV(h, s, v)".
template <>
struct std::formatter<datamunge::image::HSV> : datamunge::format_detail::text_formatter<datamunge::image::HSV> {
  /// @brief Hue (degrees), saturation and value.
  /// @param c The color to print.
  /// @return "HSV(h, s, v)".
  static std::string text(const datamunge::image::HSV& c) { return std::format("HSV({}, {}, {})", c.h, c.s, c.v); }
};

/// @brief Prints an Image as its header line, e.g. "Image 640x480 RGB" (pixels are never dumped).
template <>
struct std::formatter<datamunge::image::Image> : datamunge::format_detail::text_formatter<datamunge::image::Image> {
  /// @brief Width x height and mode.
  /// @param img The image to print.
  /// @return "Image WxH Mode".
  static std::string text(const datamunge::image::Image& img) {
    return std::format("Image {}x{} {}", img.width(), img.height(), img.mode());
  }
};

/// @brief Prints JpegDecodeOptions as "JpegDecodeOptions(channels=.., dct_method=.., fancy_upsampling=..)".
template <>
struct std::formatter<datamunge::image::JpegDecodeOptions>
    : datamunge::format_detail::text_formatter<datamunge::image::JpegDecodeOptions> {
  /// @brief The configured decode options.
  /// @param o The options to print.
  /// @return The one-line description.
  static std::string text(const datamunge::image::JpegDecodeOptions& o) {
    return std::format("JpegDecodeOptions(channels={}, dct_method={}, fancy_upsampling={})", o.channels, o.dct_method,
                       o.fancy_upsampling);
  }
};

/// @brief Prints JpegInfo as its header facts (size, components, color space, sampling, markers).
template <>
struct std::formatter<datamunge::image::JpegInfo> : datamunge::format_detail::text_formatter<datamunge::image::JpegInfo> {
  /// @brief The stream's header facts on one line.
  /// @param i The info to print.
  /// @return The one-line description.
  static std::string text(const datamunge::image::JpegInfo& i) {
    using datamunge::format_detail::preview;
    return std::format(
        "JpegInfo({}x{}, components={}, color_space={}, progressive={}, h_sampling={}, v_sampling={}, "
        "restart_interval={}, jfif={}, adobe={}, adobe_transform={}, orientation={})",
        i.width, i.height, i.components, i.color_space, i.progressive, preview(i.h_sampling), preview(i.v_sampling),
        i.restart_interval, i.jfif, i.adobe, i.adobe_transform, i.orientation);
  }
};
