#pragma once

/// @file units.hpp
/// @brief Physical lengths (inches, centimetres, points, ...) for plot sizes, converted to
///        pixels at the CSS/SVG reference resolution of 96 pixels per inch.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

/// @brief Length units accepted by the plots' width() and height() setters.
///
/// @par Example
/// @code
/// auto fig = dm::graphics::Splom::create(dm::datasets::iris())
///              .width(dm::units::inch{8})
///              .height(dm::units::cm{10});
/// std::print("{}x{}\n", fig.width(), fig.height());   // 768x378
/// @endcode
namespace datamunge::units {

/// @brief Pixels per inch used for every conversion: the CSS and SVG reference pixel
///        (1in = 96px), so an SVG drawn at a size in inches measures that many inches.
inline constexpr double pixels_per_inch = 96.0;

/// @brief A length in pixels (the plots' native unit).
struct px {
  /// @brief Number of pixels.
  double value;

  /// @brief Wraps @p pixels.
  /// @param pixels  Length in pixels.
  explicit constexpr px(double pixels) noexcept : value(pixels) {}
};

/// @brief A length in inches (96 px each), like R's `units = "in"` in ggsave() and pdf().
struct inch {
  /// @brief Number of inches.
  double value;

  /// @brief Wraps @p inches.
  /// @param inches  Length in inches.
  explicit constexpr inch(double inches) noexcept : value(inches) {}
};

/// @brief A length in centimetres (2.54 cm per inch).
struct cm {
  /// @brief Number of centimetres.
  double value;

  /// @brief Wraps @p centimetres.
  /// @param centimetres  Length in centimetres.
  explicit constexpr cm(double centimetres) noexcept : value(centimetres) {}
};

/// @brief A length in millimetres (25.4 mm per inch).
struct mm {
  /// @brief Number of millimetres.
  double value;

  /// @brief Wraps @p millimetres.
  /// @param millimetres  Length in millimetres.
  explicit constexpr mm(double millimetres) noexcept : value(millimetres) {}
};

/// @brief A length in typographic points (72 pt per inch).
struct pt {
  /// @brief Number of points.
  double value;

  /// @brief Wraps @p points.
  /// @param points  Length in points.
  explicit constexpr pt(double points) noexcept : value(points) {}
};

/// @brief A length in any of the units above, as a plot's width() or height() takes it.
///
/// Converts implicitly from px, inch, cm, mm and pt; the conversion to pixels uses
/// pixels_per_inch (96), as CSS and SVG do.
///
/// @par Example
/// @code
/// std::print("{} {} {}\n", dm::units::Length(dm::units::inch{1}).pixels(),
///            dm::units::Length(dm::units::pt{72}).pixels(),
///            dm::units::Length(dm::units::cm{2.54}).to_pixels());   // 96 96 96
/// @endcode
///
/// @par References
/// - W3C CSS Values and Units Level 3, absolute lengths (1in = 2.54cm = 96px = 72pt):
///   https://www.w3.org/TR/css-values-3/#absolute-lengths
///
/// @see pixels_per_inch
class Length {
 public:
  /// @brief A length of @p length pixels.
  /// @param length  Length in pixels.
  constexpr Length(px length) noexcept : pixels_(length.value) {}  // NOLINT(google-explicit-constructor)
  /// @brief A length of @p length inches.
  /// @param length  Length in inches.
  constexpr Length(inch length) noexcept : pixels_(length.value * pixels_per_inch) {}  // NOLINT(google-explicit-constructor)
  /// @brief A length of @p length centimetres.
  /// @param length  Length in centimetres.
  constexpr Length(cm length) noexcept : pixels_(length.value / 2.54 * pixels_per_inch) {}  // NOLINT(google-explicit-constructor)
  /// @brief A length of @p length millimetres.
  /// @param length  Length in millimetres.
  constexpr Length(mm length) noexcept : pixels_(length.value / 25.4 * pixels_per_inch) {}  // NOLINT(google-explicit-constructor)
  /// @brief A length of @p length points.
  /// @param length  Length in points.
  constexpr Length(pt length) noexcept : pixels_(length.value / 72.0 * pixels_per_inch) {}  // NOLINT(google-explicit-constructor)

  /// @brief The exact length in pixels.
  /// @return Pixels, possibly fractional.
  [[nodiscard]] constexpr double pixels() const noexcept { return pixels_; }

  /// @brief The length rounded to a whole number of pixels, for a plot's size().
  /// @return Pixels, rounded to nearest.
  /// @throws std::invalid_argument  If the length is not finite or rounds to less than one
  ///         pixel.
  [[nodiscard]] std::size_t to_pixels() const {
    if (!std::isfinite(pixels_) || std::lround(pixels_) < 1) {
      throw std::invalid_argument("units::Length: a plot dimension must be at least one pixel, got " +
                                  std::to_string(pixels_) + "px");
    }
    return static_cast<std::size_t>(std::lround(pixels_));
  }

 private:
  double pixels_;  ///< The length in pixels.
};

}  // namespace datamunge::units
