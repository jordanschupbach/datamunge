#include <datamunge/image/read_image.hpp>

#include <datamunge/image/bmp.hpp>
#include <datamunge/image/jpeg.hpp>
#include <datamunge/image/netpbm.hpp>
#include <datamunge/image/png.hpp>

#include <fstream>
#include <stdexcept>

namespace datamunge::image {

Image read_image(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("read_image: cannot open " + path);
  unsigned char sig[8] = {};
  in.read(reinterpret_cast<char*>(sig), sizeof sig);
  const std::streamsize got = in.gcount();
  in.close();
  if (got >= 8 && sig[0] == 0x89 && sig[1] == 'P' && sig[2] == 'N' && sig[3] == 'G') return read_png(path);
  if (got >= 3 && is_jpeg(sig, static_cast<std::size_t>(got))) return read_jpeg(path);
  if (got >= 2 && sig[0] == 'B' && sig[1] == 'M') return read_bmp(path);
  if (got >= 2 && sig[0] == 'P') {
    if (sig[1] == '6') return read_ppm(path);
    if (sig[1] == '5') return read_pgm(path);
    if (sig[1] == '4') return read_pbm(path);
  }
  throw std::runtime_error("read_image: unrecognised image format in " + path);
}

}  // namespace datamunge::image
