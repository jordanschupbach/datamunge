#include <datamunge/image/bmp.hpp>

#include <datamunge/image/color.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace datamunge::image {

namespace {

std::uint16_t read_u16le(std::istream& in) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}

std::uint32_t read_u32le(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) | (static_cast<std::uint32_t>(b[2]) << 16) |
           (static_cast<std::uint32_t>(b[3]) << 24);
}

std::int32_t read_i32le(std::istream& in) { return static_cast<std::int32_t>(read_u32le(in)); }

void write_u16le(std::ostream& out, std::uint16_t v) {
    const unsigned char b[2] = {static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF)};
    out.write(reinterpret_cast<const char*>(b), 2);
}

void write_u32le(std::ostream& out, std::uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
                                 static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) & 0xFF)};
    out.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32le(std::ostream& out, std::int32_t v) { write_u32le(out, static_cast<std::uint32_t>(v)); }

std::size_t row_stride(int width, int bytes_per_pixel) { return (static_cast<std::size_t>(width) * bytes_per_pixel + 3) & ~std::size_t{3}; }

} // namespace

Image read_bmp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("read_bmp: cannot open '" + path + "' for reading");
    }

    char magic[2];
    in.read(magic, 2);
    if (magic[0] != 'B' || magic[1] != 'M') {
        throw std::runtime_error("read_bmp: not a BMP file (bad magic) in '" + path + "'");
    }
    read_u32le(in); // file size, unused
    read_u16le(in); // reserved1
    read_u16le(in); // reserved2
    const std::uint32_t pixel_offset = read_u32le(in);

    const std::uint32_t dib_header_size = read_u32le(in);
    if (dib_header_size < 40) {
        throw std::runtime_error("read_bmp: unsupported (pre-BITMAPINFOHEADER) DIB header in '" + path + "'");
    }
    const std::int32_t raw_width = read_i32le(in);
    const std::int32_t raw_height = read_i32le(in);
    read_u16le(in); // planes
    const std::uint16_t bpp = read_u16le(in);
    const std::uint32_t compression = read_u32le(in);
    read_u32le(in); // image size (recomputed below rather than trusted)
    read_i32le(in); // x pixels per meter
    read_i32le(in); // y pixels per meter
    std::uint32_t colors_used = read_u32le(in);
    read_u32le(in); // colors important
    // Skip any additional V4/V5 header fields beyond the 40-byte BITMAPINFOHEADER core.
    if (dib_header_size > 40) {
        in.seekg(static_cast<std::streamoff>(dib_header_size - 40), std::ios::cur);
    }

    if (compression != 0) {
        throw std::runtime_error("read_bmp: only uncompressed (BI_RGB) BMP files are supported");
    }
    if (raw_width <= 0) {
        throw std::runtime_error("read_bmp: invalid width in '" + path + "'");
    }
    const int width = raw_width;
    const int height = std::abs(raw_height);
    const bool top_down = raw_height < 0; // negative height means rows are stored top-to-bottom

    std::vector<Pixel> palette;
    if (bpp == 8) {
        if (colors_used == 0) colors_used = 256;
        palette.resize(colors_used);
        for (std::uint32_t i = 0; i < colors_used; ++i) {
            unsigned char bgra[4];
            in.read(reinterpret_cast<char*>(bgra), 4);
            palette[i] = Pixel{bgra[2], bgra[1], bgra[0], 255};
        }
    } else if (bpp != 24 && bpp != 32) {
        throw std::runtime_error("read_bmp: unsupported bit depth (" + std::to_string(bpp) + ") in '" + path + "'");
    }

    in.seekg(static_cast<std::streamoff>(pixel_offset), std::ios::beg);
    const int bytes_per_pixel = (bpp == 8) ? 1 : (bpp / 8);
    const std::size_t stride = row_stride(width, bytes_per_pixel);
    std::vector<unsigned char> row(stride);

    Image img(width, height, ImageMode::RGB);
    for (int file_row = 0; file_row < height; ++file_row) {
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(stride));
        if (!in) {
            throw std::runtime_error("read_bmp: truncated pixel data in '" + path + "'");
        }
        const int y = top_down ? file_row : (height - 1 - file_row);
        for (int x = 0; x < width; ++x) {
            Pixel p;
            if (bpp == 8) {
                p = palette[row[static_cast<std::size_t>(x)]];
            } else {
                const std::size_t base = static_cast<std::size_t>(x) * bytes_per_pixel;
                p = Pixel{row[base + 2], row[base + 1], row[base], 255}; // BGR(x) -> RGB, alpha ignored
            }
            img.set_pixel(x, y, p);
        }
    }
    return img;
}

void write_bmp(const Image& img, const std::string& path) {
    Image converted;
    const Image* rgb = &img;
    if (img.mode() != ImageMode::RGB) {
        converted = to_rgb(img);
        rgb = &converted;
    }

    const int width = rgb->width();
    const int height = rgb->height();
    const std::size_t stride = row_stride(width, 3);
    const std::uint32_t pixel_data_size = static_cast<std::uint32_t>(stride * static_cast<std::size_t>(height));
    const std::uint32_t pixel_offset = 14 + 40;
    const std::uint32_t file_size = pixel_offset + pixel_data_size;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_bmp: cannot open '" + path + "' for writing");
    }

    out.write("BM", 2);
    write_u32le(out, file_size);
    write_u16le(out, 0);
    write_u16le(out, 0);
    write_u32le(out, pixel_offset);

    write_u32le(out, 40); // BITMAPINFOHEADER size
    write_i32le(out, width);
    write_i32le(out, height); // positive: bottom-up row order
    write_u16le(out, 1);      // planes
    write_u16le(out, 24);     // bpp
    write_u32le(out, 0);      // BI_RGB
    write_u32le(out, pixel_data_size);
    write_i32le(out, 2835); // ~72 DPI
    write_i32le(out, 2835);
    write_u32le(out, 0); // colors used
    write_u32le(out, 0); // colors important

    std::vector<unsigned char> row(stride, 0);
    for (int file_row = 0; file_row < height; ++file_row) {
        const int y = height - 1 - file_row; // BMP rows are bottom-up
        for (int x = 0; x < width; ++x) {
            const Pixel p = rgb->get_pixel(x, y);
            row[static_cast<std::size_t>(x) * 3 + 0] = p.b;
            row[static_cast<std::size_t>(x) * 3 + 1] = p.g;
            row[static_cast<std::size_t>(x) * 3 + 2] = p.r;
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(stride));
    }
    if (!out) {
        throw std::runtime_error("write_bmp: I/O error while writing '" + path + "'");
    }
}

} // namespace datamunge::image
