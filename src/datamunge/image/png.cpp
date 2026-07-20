#include <datamunge/image/png.hpp>

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace datamunge::image {

namespace {

constexpr unsigned char kPngSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};

void write_u32be(std::ostream& out, std::uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>((v >> 24) & 0xFF), static_cast<unsigned char>((v >> 16) & 0xFF),
                                 static_cast<unsigned char>((v >> 8) & 0xFF), static_cast<unsigned char>(v & 0xFF)};
    out.write(reinterpret_cast<const char*>(b), 4);
}

std::uint32_t read_u32be(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) {
        throw std::runtime_error("read_png: unexpected end of file");
    }
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) | (static_cast<std::uint32_t>(b[2]) << 8) |
           static_cast<std::uint32_t>(b[3]);
}

std::uint32_t png_crc32(const char type[4], const std::vector<unsigned char>& data) {
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(type), 4);
    if (!data.empty()) {
        crc = crc32(crc, data.data(), static_cast<uInt>(data.size()));
    }
    return static_cast<std::uint32_t>(crc);
}

void write_chunk(std::ostream& out, const char type[4], const std::vector<unsigned char>& data) {
    write_u32be(out, static_cast<std::uint32_t>(data.size()));
    out.write(type, 4);
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    write_u32be(out, png_crc32(type, data));
}

struct Chunk {
    char type[4] = {0, 0, 0, 0};
    std::vector<unsigned char> data;
};

Chunk read_chunk(std::istream& in) {
    Chunk chunk;
    const std::uint32_t length = read_u32be(in);
    in.read(chunk.type, 4);
    if (!in) {
        throw std::runtime_error("read_png: truncated chunk header");
    }
    chunk.data.resize(length);
    if (length > 0) {
        in.read(reinterpret_cast<char*>(chunk.data.data()), static_cast<std::streamsize>(length));
        if (!in) {
            throw std::runtime_error("read_png: truncated chunk data");
        }
    }
    const std::uint32_t expected_crc = read_u32be(in);
    const std::uint32_t actual_crc = png_crc32(chunk.type, chunk.data);
    if (expected_crc != actual_crc) {
        throw std::runtime_error("read_png: chunk CRC mismatch (corrupt file)");
    }
    return chunk;
}

std::uint8_t paeth_predictor(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return static_cast<std::uint8_t>(a);
    if (pb <= pc) return static_cast<std::uint8_t>(b);
    return static_cast<std::uint8_t>(c);
}

ImageMode color_type_to_mode(std::uint8_t color_type) {
    switch (color_type) {
    case 0:
        return ImageMode::Grayscale;
    case 2:
        return ImageMode::RGB;
    case 4:
        return ImageMode::GrayscaleAlpha;
    case 6:
        return ImageMode::RGBA;
    default:
        throw std::runtime_error("read_png: unsupported PNG color type (" + std::to_string(color_type) +
                                  "); only grayscale/RGB/RGBA/grayscale+alpha (color types 0/2/4/6) are supported, not indexed-color");
    }
}

std::uint8_t mode_to_color_type(ImageMode mode) {
    switch (mode) {
    case ImageMode::Grayscale:
        return 0;
    case ImageMode::RGB:
        return 2;
    case ImageMode::GrayscaleAlpha:
        return 4;
    case ImageMode::RGBA:
        return 6;
    }
    throw std::logic_error("write_png: unreachable mode");
}

} // namespace

Image read_png(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("read_png: cannot open '" + path + "' for reading");
    }

    unsigned char signature[8];
    in.read(reinterpret_cast<char*>(signature), 8);
    if (!in || std::memcmp(signature, kPngSignature, 8) != 0) {
        throw std::runtime_error("read_png: not a PNG file (bad signature) in '" + path + "'");
    }

    const Chunk ihdr = read_chunk(in);
    if (std::string(ihdr.type, 4) != "IHDR" || ihdr.data.size() != 13) {
        throw std::runtime_error("read_png: missing or malformed IHDR chunk in '" + path + "'");
    }
    const std::uint32_t width = (static_cast<std::uint32_t>(ihdr.data[0]) << 24) | (static_cast<std::uint32_t>(ihdr.data[1]) << 16) |
                                 (static_cast<std::uint32_t>(ihdr.data[2]) << 8) | static_cast<std::uint32_t>(ihdr.data[3]);
    const std::uint32_t height = (static_cast<std::uint32_t>(ihdr.data[4]) << 24) | (static_cast<std::uint32_t>(ihdr.data[5]) << 16) |
                                  (static_cast<std::uint32_t>(ihdr.data[6]) << 8) | static_cast<std::uint32_t>(ihdr.data[7]);
    const std::uint8_t bit_depth = ihdr.data[8];
    const std::uint8_t color_type = ihdr.data[9];
    const std::uint8_t compression_method = ihdr.data[10];
    const std::uint8_t filter_method = ihdr.data[11];
    const std::uint8_t interlace_method = ihdr.data[12];

    if (width == 0 || height == 0) {
        throw std::runtime_error("read_png: invalid dimensions in '" + path + "'");
    }
    if (bit_depth != 8) {
        throw std::runtime_error("read_png: only 8-bit depth PNGs are supported (got bit depth " + std::to_string(bit_depth) + ")");
    }
    if (compression_method != 0 || filter_method != 0) {
        throw std::runtime_error("read_png: unsupported PNG compression/filter method");
    }
    if (interlace_method != 0) {
        throw std::runtime_error("read_png: Adam7-interlaced PNGs are not supported");
    }
    const ImageMode mode = color_type_to_mode(color_type);
    const std::size_t bpp = static_cast<std::size_t>(Image::channel_count(mode));

    std::vector<unsigned char> compressed;
    bool seen_iend = false;
    while (!seen_iend) {
        Chunk chunk = read_chunk(in);
        const std::string type(chunk.type, 4);
        if (type == "IDAT") {
            compressed.insert(compressed.end(), chunk.data.begin(), chunk.data.end());
        } else if (type == "IEND") {
            seen_iend = true;
        }
        // Ancillary/unused chunks (PLTE for non-palette files, tEXt, gAMA, etc.) are ignored.
    }
    if (compressed.empty()) {
        throw std::runtime_error("read_png: no IDAT data in '" + path + "'");
    }

    const std::size_t stride = static_cast<std::size_t>(width) * bpp;
    const std::size_t raw_size = (stride + 1) * static_cast<std::size_t>(height);
    std::vector<unsigned char> raw(raw_size);
    uLongf dest_len = static_cast<uLongf>(raw_size);
    const int rc = uncompress(raw.data(), &dest_len, compressed.data(), static_cast<uLong>(compressed.size()));
    if (rc != Z_OK || dest_len != raw_size) {
        throw std::runtime_error("read_png: failed to decompress IDAT stream in '" + path + "' (corrupt or truncated file)");
    }

    Image img(static_cast<int>(width), static_cast<int>(height), mode);
    auto& out_data = img.data();
    std::vector<unsigned char> prior_row(stride, 0);
    std::vector<unsigned char> current_row(stride);

    std::size_t raw_pos = 0;
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t filter_type = raw[raw_pos++];
        for (std::size_t x = 0; x < stride; ++x) {
            const std::uint8_t filtered = raw[raw_pos + x];
            const int a = (x >= bpp) ? current_row[x - bpp] : 0; // left
            const int b = prior_row[x];                          // up
            const int c = (x >= bpp) ? prior_row[x - bpp] : 0;    // upper-left

            std::uint8_t value = 0;
            switch (filter_type) {
            case 0:
                value = filtered;
                break;
            case 1:
                value = static_cast<std::uint8_t>(filtered + a);
                break;
            case 2:
                value = static_cast<std::uint8_t>(filtered + b);
                break;
            case 3:
                value = static_cast<std::uint8_t>(filtered + ((a + b) / 2));
                break;
            case 4:
                value = static_cast<std::uint8_t>(filtered + paeth_predictor(a, b, c));
                break;
            default:
                throw std::runtime_error("read_png: invalid scanline filter type (" + std::to_string(filter_type) + ")");
            }
            current_row[x] = value;
        }
        raw_pos += stride;
        std::copy(current_row.begin(), current_row.end(),
                  out_data.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y) * stride));
        std::swap(prior_row, current_row);
    }

    return img;
}

void write_png(const Image& img, const std::string& path) {
    const int width = img.width();
    const int height = img.height();
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("write_png: image must be non-empty");
    }
    const std::size_t stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(img.channels());

    std::vector<unsigned char> raw((stride + 1) * static_cast<std::size_t>(height));
    const auto& src_data = img.data();
    for (int y = 0; y < height; ++y) {
        const std::size_t raw_row_start = static_cast<std::size_t>(y) * (stride + 1);
        raw[raw_row_start] = 0; // filter type: None
        std::copy_n(src_data.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y) * stride), stride,
                    raw.begin() + static_cast<std::ptrdiff_t>(raw_row_start) + 1);
    }

    uLongf bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> compressed(bound);
    const int rc = compress2(compressed.data(), &bound, raw.data(), static_cast<uLong>(raw.size()), Z_DEFAULT_COMPRESSION);
    if (rc != Z_OK) {
        throw std::runtime_error("write_png: zlib compression failed while writing '" + path + "'");
    }
    compressed.resize(bound);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_png: cannot open '" + path + "' for writing");
    }
    out.write(reinterpret_cast<const char*>(kPngSignature), 8);

    std::vector<unsigned char> ihdr_data(13);
    const auto put_u32 = [&](std::size_t offset, std::uint32_t v) {
        ihdr_data[offset] = static_cast<unsigned char>((v >> 24) & 0xFF);
        ihdr_data[offset + 1] = static_cast<unsigned char>((v >> 16) & 0xFF);
        ihdr_data[offset + 2] = static_cast<unsigned char>((v >> 8) & 0xFF);
        ihdr_data[offset + 3] = static_cast<unsigned char>(v & 0xFF);
    };
    put_u32(0, static_cast<std::uint32_t>(width));
    put_u32(4, static_cast<std::uint32_t>(height));
    ihdr_data[8] = 8; // bit depth
    ihdr_data[9] = mode_to_color_type(img.mode());
    ihdr_data[10] = 0; // compression method
    ihdr_data[11] = 0; // filter method
    ihdr_data[12] = 0; // interlace method

    write_chunk(out, "IHDR", ihdr_data);
    write_chunk(out, "IDAT", compressed);
    write_chunk(out, "IEND", {});

    if (!out) {
        throw std::runtime_error("write_png: I/O error while writing '" + path + "'");
    }
}

} // namespace datamunge::image
