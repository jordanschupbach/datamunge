#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace datamunge::image {

/// @brief The number of stored bytes per pixel doubles as the mode's own value (1/2/3/4), so
///        `channels()` never needs a separate lookup table.
enum class ImageMode : int { Grayscale = 1, GrayscaleAlpha = 2, RGB = 3, RGBA = 4 };

/// @brief A pixel value in fully-expanded RGBA form, regardless of the owning Image's actual
///        mode -- always 4 plain uint8 fields (rather than a variable-length per-mode vector)
///        so this stays trivially bindable across languages, matching this codebase's
///        established preference for fixed-shape structs. get_pixel()/set_pixel() handle the
///        expansion/truncation to and from the Image's real stored channel count.
struct Pixel {
    std::uint8_t r{0};
    std::uint8_t g{0};
    std::uint8_t b{0};
    std::uint8_t a{255};
};

[[nodiscard]] inline bool operator==(const Pixel& lhs, const Pixel& rhs) {
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
}

/// @brief A 2D raster image: row-major, top-left origin, interleaved channel bytes (matching
///        the layout every common image codec uses on disk, so I/O needs no reshuffling).
///        Pixel data is always stored as the mode's exact channel count (1/2/3/4 bytes per
///        pixel) -- get_pixel()/set_pixel() are the only place mode-specific expansion happens.
class Image {
  public:
    Image() = default;

    Image(int width, int height, ImageMode mode = ImageMode::RGB) : Image(width, height, mode, Pixel{}) {}

    Image(int width, int height, ImageMode mode, const Pixel& fill_color) : width_(width), height_(height), mode_(mode) {
        if (width < 0 || height < 0) {
            throw std::invalid_argument("Image: width and height must be non-negative");
        }
        data_.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channel_count(mode)));
        fill(fill_color);
    }

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] ImageMode mode() const { return mode_; }
    [[nodiscard]] int channels() const { return channel_count(mode_); }

    [[nodiscard]] const std::vector<std::uint8_t>& data() const { return data_; }
    [[nodiscard]] std::vector<std::uint8_t>& data() { return data_; }

    [[nodiscard]] Pixel get_pixel(int x, int y) const {
        const std::size_t base = pixel_offset(x, y);
        switch (mode_) {
        case ImageMode::Grayscale: {
            const std::uint8_t v = data_[base];
            return Pixel{v, v, v, 255};
        }
        case ImageMode::GrayscaleAlpha: {
            const std::uint8_t v = data_[base];
            return Pixel{v, v, v, data_[base + 1]};
        }
        case ImageMode::RGB:
            return Pixel{data_[base], data_[base + 1], data_[base + 2], 255};
        case ImageMode::RGBA:
            return Pixel{data_[base], data_[base + 1], data_[base + 2], data_[base + 3]};
        }
        throw std::logic_error("Image::get_pixel: unreachable mode");
    }

    void set_pixel(int x, int y, const Pixel& p) {
        const std::size_t base = pixel_offset(x, y);
        switch (mode_) {
        case ImageMode::Grayscale:
            data_[base] = p.r;
            return;
        case ImageMode::GrayscaleAlpha:
            data_[base] = p.r;
            data_[base + 1] = p.a;
            return;
        case ImageMode::RGB:
            data_[base] = p.r;
            data_[base + 1] = p.g;
            data_[base + 2] = p.b;
            return;
        case ImageMode::RGBA:
            data_[base] = p.r;
            data_[base + 1] = p.g;
            data_[base + 2] = p.b;
            data_[base + 3] = p.a;
            return;
        }
        throw std::logic_error("Image::set_pixel: unreachable mode");
    }

    void fill(const Pixel& p) {
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                set_pixel(x, y, p);
            }
        }
    }

    [[nodiscard]] static int channel_count(ImageMode mode) { return static_cast<int>(mode); }

  private:
    [[nodiscard]] std::size_t pixel_offset(int x, int y) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) {
            throw std::out_of_range("Image: pixel coordinates out of range");
        }
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels());
    }

    int width_{0};
    int height_{0};
    ImageMode mode_{ImageMode::RGB};
    std::vector<std::uint8_t> data_;
};

} // namespace datamunge::image
