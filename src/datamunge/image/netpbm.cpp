#include <datamunge/image/netpbm.hpp>

#include <datamunge/image/color.hpp>

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace datamunge::image {

namespace {

void skip_whitespace_and_comments(std::istream& in) {
    int c;
    while ((c = in.peek()) != EOF) {
        if (std::isspace(c)) {
            in.get();
            continue;
        }
        if (c == '#') {
            while ((c = in.get()) != EOF && c != '\n') {
            }
            continue;
        }
        break;
    }
}

std::string read_token(std::istream& in) {
    skip_whitespace_and_comments(in);
    std::string token;
    int c;
    while ((c = in.peek()) != EOF && !std::isspace(c)) {
        token.push_back(static_cast<char>(in.get()));
    }
    if (token.empty()) {
        throw std::runtime_error("netpbm: unexpected end of file while reading header");
    }
    return token;
}

int read_int_token(std::istream& in) {
    const std::string token = read_token(in);
    try {
        return std::stoi(token);
    } catch (const std::exception&) {
        throw std::runtime_error("netpbm: expected an integer in the header, got '" + token + "'");
    }
}

void check_magic(std::istream& in, const std::string& expected) {
    const std::string magic = read_token(in);
    if (magic != expected) {
        throw std::runtime_error("netpbm: expected magic number '" + expected + "', got '" + magic + "'");
    }
}

std::ifstream open_binary_read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("netpbm: cannot open '" + path + "' for reading");
    }
    return in;
}

std::ofstream open_binary_write(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("netpbm: cannot open '" + path + "' for writing");
    }
    return out;
}

std::pair<int, int> read_dimensions(std::istream& in) {
    const int width = read_int_token(in);
    const int height = read_int_token(in);
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("netpbm: width and height must be positive");
    }
    return {width, height};
}

} // namespace

Image read_ppm(const std::string& path) {
    std::ifstream in = open_binary_read(path);
    check_magic(in, "P6");
    const auto [width, height] = read_dimensions(in);
    const int maxval = read_int_token(in);
    if (maxval <= 0 || maxval > 255) {
        throw std::runtime_error("read_ppm: only 8-bit (maxval <= 255) files are supported");
    }
    in.get(); // the single whitespace byte the spec requires right before the raster

    Image img(width, height, ImageMode::RGB);
    in.read(reinterpret_cast<char*>(img.data().data()), static_cast<std::streamsize>(img.data().size()));
    if (!in) {
        throw std::runtime_error("read_ppm: truncated pixel data in '" + path + "'");
    }
    return img;
}

void write_ppm(const Image& img, const std::string& path) {
    Image converted;
    const Image* rgb = &img;
    if (img.mode() != ImageMode::RGB) {
        converted = to_rgb(img);
        rgb = &converted;
    }

    std::ofstream out = open_binary_write(path);
    out << "P6\n" << rgb->width() << ' ' << rgb->height() << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb->data().data()), static_cast<std::streamsize>(rgb->data().size()));
    if (!out) {
        throw std::runtime_error("write_ppm: I/O error while writing '" + path + "'");
    }
}

Image read_pgm(const std::string& path) {
    std::ifstream in = open_binary_read(path);
    check_magic(in, "P5");
    const auto [width, height] = read_dimensions(in);
    const int maxval = read_int_token(in);
    if (maxval <= 0 || maxval > 255) {
        throw std::runtime_error("read_pgm: only 8-bit (maxval <= 255) files are supported");
    }
    in.get();

    Image img(width, height, ImageMode::Grayscale);
    in.read(reinterpret_cast<char*>(img.data().data()), static_cast<std::streamsize>(img.data().size()));
    if (!in) {
        throw std::runtime_error("read_pgm: truncated pixel data in '" + path + "'");
    }
    return img;
}

void write_pgm(const Image& img, const std::string& path) {
    Image converted;
    const Image* gray = &img;
    if (img.mode() != ImageMode::Grayscale) {
        converted = to_grayscale(img);
        gray = &converted;
    }

    std::ofstream out = open_binary_write(path);
    out << "P5\n" << gray->width() << ' ' << gray->height() << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray->data().data()), static_cast<std::streamsize>(gray->data().size()));
    if (!out) {
        throw std::runtime_error("write_pgm: I/O error while writing '" + path + "'");
    }
}

Image read_pbm(const std::string& path) {
    std::ifstream in = open_binary_read(path);
    check_magic(in, "P4");
    const auto [width, height] = read_dimensions(in);
    in.get();

    const std::size_t bytes_per_row = (static_cast<std::size_t>(width) + 7) / 8;
    std::vector<unsigned char> raster(bytes_per_row * static_cast<std::size_t>(height));
    in.read(reinterpret_cast<char*>(raster.data()), static_cast<std::streamsize>(raster.size()));
    if (!in) {
        throw std::runtime_error("read_pbm: truncated pixel data in '" + path + "'");
    }

    Image img(width, height, ImageMode::Grayscale);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned char byte = raster[static_cast<std::size_t>(y) * bytes_per_row + static_cast<std::size_t>(x) / 8];
            const bool bit_set = (byte >> (7 - (x % 8))) & 1; // PBM: a set bit is BLACK
            const std::uint8_t value = bit_set ? 0 : 255;
            img.set_pixel(x, y, Pixel{value, value, value, 255});
        }
    }
    return img;
}

void write_pbm(const Image& img, const std::string& path, std::uint8_t threshold) {
    const Image gray = to_grayscale(img);
    const std::size_t bytes_per_row = (static_cast<std::size_t>(gray.width()) + 7) / 8;
    std::vector<unsigned char> raster(bytes_per_row * static_cast<std::size_t>(gray.height()), 0);

    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            if (gray.get_pixel(x, y).r < threshold) {
                raster[static_cast<std::size_t>(y) * bytes_per_row + static_cast<std::size_t>(x) / 8] |=
                    static_cast<unsigned char>(1u << (7 - (x % 8)));
            }
        }
    }

    std::ofstream out = open_binary_write(path);
    out << "P4\n" << gray.width() << ' ' << gray.height() << '\n';
    out.write(reinterpret_cast<const char*>(raster.data()), static_cast<std::streamsize>(raster.size()));
    if (!out) {
        throw std::runtime_error("write_pbm: I/O error while writing '" + path + "'");
    }
}

} // namespace datamunge::image
