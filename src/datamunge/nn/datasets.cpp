// Dataset loaders: IDX (MNIST), CIFAR binary, CSV, image folders (PNG/JPEG).

#include "datamunge/nn/datasets.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>

#include "datamunge/image/detail/crc32.hpp"
#include "datamunge/image/detail/zlib_codec.hpp"
#include "datamunge/image/jpeg.hpp"
#include "datamunge/image/png.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/parallel.hpp"

namespace datamunge::nn {

namespace {

using Bytes = std::vector<unsigned char>;

bool file_exists(const std::string& path) { return std::ifstream(path, std::ios::binary).good(); }

Bytes read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "' for reading");
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  Bytes bytes(size);
  if (size && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)))
    throw std::runtime_error("failed reading '" + path + "'");
  return bytes;
}

/// Decompresses a single-member gzip file (RFC 1952).
Bytes gunzip(const Bytes& b, const std::string& path) {
  const auto fail = [&](const std::string& why) { return std::runtime_error("gzip: " + why + " in '" + path + "'"); };
  if (b.size() < 18 || b[0] != 0x1f || b[1] != 0x8b) throw fail("bad magic");
  if (b[2] != 8) throw fail("unsupported compression method");
  const unsigned flags = b[3];
  std::size_t    at    = 10;
  if (flags & 0x04u) {  // FEXTRA
    if (at + 2 > b.size()) throw fail("truncated header");
    at += 2 + (b[at] | (static_cast<std::size_t>(b[at + 1]) << 8));
  }
  for (unsigned bit : {0x08u, 0x10u}) {  // FNAME, FCOMMENT: zero-terminated
    if (!(flags & bit)) continue;
    while (at < b.size() && b[at] != 0) ++at;
    ++at;
  }
  if (flags & 0x02u) at += 2;  // FHCRC
  if (at + 8 > b.size()) throw fail("truncated file");
  const std::size_t   trailer = b.size() - 8;
  const std::uint32_t crc = b[trailer] | (b[trailer + 1] << 8) | (b[trailer + 2] << 16) |
                            (static_cast<std::uint32_t>(b[trailer + 3]) << 24);
  const std::uint32_t isize = b[trailer + 4] | (b[trailer + 5] << 8) | (b[trailer + 6] << 16) |
                              (static_cast<std::uint32_t>(b[trailer + 7]) << 24);
  Bytes out;
  try {
    out = image::detail::deflate_decompress(b.data() + at, trailer - at, isize);
  } catch (const std::runtime_error& e) {
    throw fail(e.what());
  }
  if (image::detail::crc32_update(0, out.data(), out.size()) != crc) throw fail("CRC-32 mismatch");
  return out;
}

Bytes read_maybe_gzip(const std::string& path) {
  Bytes b = read_file(path);
  if (b.size() >= 2 && b[0] == 0x1f && b[1] == 0x8b) return gunzip(b, path);
  return b;
}

std::uint32_t big32(const Bytes& b, std::size_t at) {
  return (static_cast<std::uint32_t>(b[at]) << 24) | (static_cast<std::uint32_t>(b[at + 1]) << 16) |
         (static_cast<std::uint32_t>(b[at + 2]) << 8) | b[at + 3];
}

/// Reverses a big-endian element into host order.
template <typename T>
T read_big(const unsigned char* p) {
  unsigned char buf[sizeof(T)];
  for (std::size_t k = 0; k < sizeof(T); ++k) buf[k] = p[sizeof(T) - 1 - k];
  T v;
  std::memcpy(&v, buf, sizeof(T));
  return v;
}

}  // namespace

Tensor read_idx(const std::string& path) {
  const Bytes b    = read_maybe_gzip(path);
  const auto  fail = [&](const std::string& why) { return std::runtime_error("IDX: " + why + " in '" + path + "'"); };
  if (b.size() < 4 || b[0] != 0 || b[1] != 0) throw fail("bad magic");
  const unsigned type = b[2], rank = b[3];
  std::size_t    item = 0;
  switch (type) {
    case 0x08:
    case 0x09: item = 1; break;
    case 0x0B: item = 2; break;
    case 0x0C:
    case 0x0D: item = 4; break;
    case 0x0E: item = 8; break;
    default: throw fail("unknown data type 0x" + std::to_string(type));
  }
  if (rank == 0) throw fail("rank 0");
  const std::size_t header = 4 + 4 * static_cast<std::size_t>(rank);
  if (b.size() < header) throw fail("truncated header");
  Shape         shape;
  std::uint64_t count = 1;
  for (unsigned i = 0; i < rank; ++i) {
    const std::uint32_t d = big32(b, 4 + 4 * i);
    shape.push_back(static_cast<std::int64_t>(d));
    // Reject products that could wrap around to a size matching the payload.
    if (d != 0 && count > (std::numeric_limits<std::uint64_t>::max() / item) / d) throw fail("dimension overflow");
    count *= d;
  }
  if (b.size() - header != count * item)
    throw fail("data size " + std::to_string(b.size() - header) + " does not match shape " + shape_str(shape));
  const unsigned char* p = b.data() + header;
  switch (type) {
    case 0x08:
    case 0x09:
    case 0x0B:
    case 0x0C: {
      Tensor        out = empty(shape, DType::Int32);
      std::int32_t* dst = out.data_ptr<std::int32_t>();
      for (std::uint64_t i = 0; i < count; ++i) {
        if (type == 0x08)
          dst[i] = p[i];
        else if (type == 0x09)
          dst[i] = static_cast<std::int8_t>(p[i]);
        else if (type == 0x0B)
          dst[i] = read_big<std::int16_t>(p + 2 * i);
        else
          dst[i] = read_big<std::int32_t>(p + 4 * i);
      }
      return out;
    }
    case 0x0D: {
      Tensor out = empty(shape, DType::Float32);
      float* dst = out.data_ptr<float>();
      for (std::uint64_t i = 0; i < count; ++i) dst[i] = read_big<float>(p + 4 * i);
      return out;
    }
    default: {
      Tensor  out = empty(shape, DType::Float64);
      double* dst = out.data_ptr<double>();
      for (std::uint64_t i = 0; i < count; ++i) dst[i] = read_big<double>(p + 8 * i);
      return out;
    }
  }
}

ImageDataset load_mnist(const std::string& directory, const std::string& split) {
  std::string prefix;
  if (split == "train")
    prefix = "train";
  else if (split == "test")
    prefix = "t10k";
  else
    throw std::invalid_argument("load_mnist: split must be \"train\" or \"test\", got \"" + split + "\"");
  const auto locate = [&](const std::string& base) {
    const std::string plain = directory + "/" + base;
    if (file_exists(plain)) return plain;
    if (file_exists(plain + ".gz")) return plain + ".gz";
    throw std::runtime_error("load_mnist: neither '" + plain + "' nor '" + plain + ".gz' exists");
  };
  const Tensor raw_images = read_idx(locate(prefix + "-images-idx3-ubyte"));
  const Tensor raw_labels = read_idx(locate(prefix + "-labels-idx1-ubyte"));
  if (raw_images.dim() != 3 || raw_labels.dim() != 1 || raw_images.size(0) != raw_labels.size(0))
    throw std::runtime_error("load_mnist: images " + shape_str(raw_images.shape()) + " and labels " +
                             shape_str(raw_labels.shape()) + " do not match");
  ImageDataset out;
  out.images = reshape(cast(raw_images, DType::Float32) / 255.0,
                       {raw_images.size(0), raw_images.size(1), raw_images.size(2), 1});
  out.labels = cast(raw_labels, DType::Int64);
  return out;
}

namespace {

/// Appends the CIFAR records of one file (label bytes, then 3x32x32 planar RGB).
void read_cifar_file(const std::string& path, std::size_t label_bytes, std::size_t label_index,
                     std::vector<float>& images, std::vector<std::int64_t>& labels) {
  const Bytes       b      = read_file(path);
  const std::size_t pixels = 32 * 32 * 3;
  const std::size_t record = label_bytes + pixels;
  if (b.empty() || b.size() % record != 0)
    throw std::runtime_error("CIFAR: '" + path + "' has " + std::to_string(b.size()) +
                             " bytes, not a positive multiple of the " + std::to_string(record) + "-byte record");
  const std::size_t n = b.size() / record;
  for (std::size_t r = 0; r < n; ++r) {
    const unsigned char* rec = b.data() + r * record;
    labels.push_back(rec[label_index]);
    const unsigned char* px = rec + label_bytes;
    // Planar CHW -> interleaved HWC.
    for (std::size_t y = 0; y < 32; ++y)
      for (std::size_t x = 0; x < 32; ++x)
        for (std::size_t c = 0; c < 3; ++c) images.push_back(px[c * 1024 + y * 32 + x] / 255.0f);
  }
}

ImageDataset cifar_dataset(std::vector<float>& images, std::vector<std::int64_t>& labels) {
  const auto   n = static_cast<std::int64_t>(labels.size());
  ImageDataset out;
  out.images = from_vector(images, {n, 32, 32, 3});
  out.labels = from_vector(labels);
  return out;
}

}  // namespace

ImageDataset load_cifar10(const std::string& directory, const std::string& split) {
  std::vector<std::string> files;
  if (split == "train")
    for (int i = 1; i <= 5; ++i) files.push_back(directory + "/data_batch_" + std::to_string(i) + ".bin");
  else if (split == "test")
    files.push_back(directory + "/test_batch.bin");
  else
    throw std::invalid_argument("load_cifar10: split must be \"train\" or \"test\", got \"" + split + "\"");
  std::vector<float>        images;
  std::vector<std::int64_t> labels;
  for (const auto& f : files) read_cifar_file(f, 1, 0, images, labels);
  return cifar_dataset(images, labels);
}

ImageDataset load_cifar100(const std::string& directory, const std::string& split, bool fine) {
  if (split != "train" && split != "test")
    throw std::invalid_argument("load_cifar100: split must be \"train\" or \"test\", got \"" + split + "\"");
  std::vector<float>        images;
  std::vector<std::int64_t> labels;
  read_cifar_file(directory + "/" + split + ".bin", 2, fine ? 1 : 0, images, labels);
  return cifar_dataset(images, labels);
}

// ---- CSV ------------------------------------------------------------------------------------

namespace {

/// Splits CSV text into rows of fields (RFC 4180 quoting; CRLF tolerated).
std::vector<std::vector<std::string>> parse_csv(const std::string& text, char delimiter, const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  std::vector<std::string>              row;
  std::string                           field;
  bool                                  quoted = false, field_started = false;
  const auto end_row = [&] {
    if (field_started || !field.empty() || !row.empty()) {
      row.push_back(field);
      rows.push_back(row);
    }
    row.clear();
    field.clear();
    field_started = false;
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char ch = text[i];
    if (quoted) {
      if (ch == '"') {
        if (i + 1 < text.size() && text[i + 1] == '"') {
          field += '"';
          ++i;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
    } else if (ch == '"') {
      quoted        = true;
      field_started = true;
    } else if (ch == delimiter) {
      row.push_back(field);
      field.clear();
      field_started = true;
    } else if (ch == '\n' || ch == '\r') {
      if (ch == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
      end_row();
    } else {
      field += ch;
    }
  }
  if (quoted) throw std::runtime_error("read_csv: unterminated quoted field in '" + path + "'");
  end_row();
  return rows;
}

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

Tensor read_csv(const std::string& path, std::vector<std::string>* column_names, bool header, char delimiter) {
  const Bytes       raw = read_file(path);
  const std::string text(raw.begin(), raw.end());
  auto              rows = parse_csv(text, delimiter, path);
  std::size_t       first = 0;
  if (header) {
    if (rows.empty()) throw std::runtime_error("read_csv: '" + path + "' has no header row");
    if (column_names) {
      column_names->clear();
      for (const auto& f : rows[0]) column_names->push_back(trim(f));
    }
    first = 1;
  } else if (column_names) {
    column_names->clear();
  }
  const std::size_t columns = header ? rows[0].size() : (rows.empty() ? 0 : rows[0].size());
  std::vector<double> values;
  for (std::size_t r = first; r < rows.size(); ++r) {
    if (rows[r].size() != columns)
      throw std::runtime_error("read_csv: row " + std::to_string(r + 1) + " of '" + path + "' has " +
                               std::to_string(rows[r].size()) + " fields, expected " + std::to_string(columns));
    for (std::size_t c = 0; c < columns; ++c) {
      const std::string cell = trim(rows[r][c]);
      if (cell.empty() || cell == "NA" || cell == "nan" || cell == "NaN" || cell == "NAN") {
        values.push_back(std::numeric_limits<double>::quiet_NaN());
        continue;
      }
      char*  end = nullptr;
      errno      = 0;
      double v   = std::strtod(cell.c_str(), &end);
      if (end == cell.c_str() || *end != '\0')
        throw std::runtime_error("read_csv: non-numeric value \"" + cell + "\" at row " + std::to_string(r + 1) +
                                 ", column " + std::to_string(c + 1) + " of '" + path + "'");
      values.push_back(v);
    }
  }
  const auto n = static_cast<std::int64_t>(rows.size() - first);
  return from_vector(values, {n, static_cast<std::int64_t>(columns)});
}

// ---------------------------------------------------------------------------
// Image folders (tflow common/data/image_folder.h semantics).

namespace {

namespace fs = std::filesystem;

std::string lower_extension(const fs::path& path) {
  std::string ext = path.extension().string();
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

bool is_image_extension(const std::string& ext) { return ext == ".png" || ext == ".jpg" || ext == ".jpeg"; }

}  // namespace

ImageFolder scan_image_folder(const std::string& root, std::vector<std::string> class_names) {
  const fs::path base(root);
  if (!fs::is_directory(base)) throw std::invalid_argument("scan_image_folder: '" + root + "' is not a directory");
  std::vector<std::string> discovered;
  for (const auto& entry : fs::directory_iterator(base))
    if (entry.is_directory()) discovered.push_back(entry.path().filename().string());
  std::sort(discovered.begin(), discovered.end());
  if (class_names.empty()) class_names = discovered;
  std::set<std::string> unique;
  for (const auto& name : class_names) {
    const fs::path part(name);
    if (name.empty() || name == "." || name == ".." || part.filename() != part || !unique.insert(name).second)
      throw std::invalid_argument("scan_image_folder: invalid or duplicate class name '" + name + "'");
  }
  for (const auto& name : discovered)
    if (!unique.count(name)) throw std::invalid_argument("scan_image_folder: unknown image class '" + name + "'");
  ImageFolder result{std::move(class_names), {}};
  for (std::size_t label = 0; label < result.class_names.size(); ++label) {
    const fs::path directory = base / result.class_names[label];
    if (!fs::exists(directory)) continue;  // a class absent from this split keeps its label
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.is_regular_file() && is_image_extension(lower_extension(entry.path()))) paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) result.images.push_back({path.string(), static_cast<std::int64_t>(label)});
  }
  if (result.images.empty()) throw std::invalid_argument("scan_image_folder: no PNG/JPEG images found in '" + root + "'");
  return result;
}

Tensor load_image_rgb(const std::string& path, std::int64_t height, std::int64_t width) {
  if (height <= 0 || width <= 0) throw std::invalid_argument("load_image_rgb: resize dimensions must be positive");
  const std::string ext = lower_extension(fs::path(path));
  if (!is_image_extension(ext)) throw std::invalid_argument("load_image_rgb: unsupported image extension in '" + path + "'");
  image::Image img;
  try {
    if (ext == ".png") {
      img = image::read_png(path);
    } else {
      image::JpegDecodeOptions options;
      options.channels = 3;
      img = image::read_jpeg(path, options);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("image decode failed for " + path + ": " + e.what());
  }
  // To RGB like DecodePng(channels=3): gray expands, alpha is dropped.
  const int ch = img.channels();
  const std::int64_t h = img.height(), w = img.width();
  Tensor rgb = empty({1, h, w, 3}, DType::Float32);
  float* dst = rgb.data_ptr<float>();
  const std::uint8_t* src = img.data().data();
  const std::int64_t n = h * w;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::uint8_t* p = src + i * ch;
    for (int c = 0; c < 3; ++c) dst[i * 3 + c] = static_cast<float>(ch >= 3 ? p[c] : p[0]);
  }
  NoGradGuard guard;
  Tensor resized = resize_bilinear(rgb, ImageSize{height, width}, false, true).contiguous();
  Tensor out = empty({height, width, 3}, DType::Float32);
  const float* r = resized.data_ptr<float>();
  float* o = out.data_ptr<float>();
  for (std::int64_t i = 0; i < height * width * 3; ++i) o[i] = r[i] * (1.f / 255.f);
  return out;
}

ImageDataset load_image_batch(const std::vector<ImageFile>& images, const std::vector<std::size_t>& rows,
                              std::int64_t height, std::int64_t width) {
  if (height <= 0 || width <= 0) throw std::invalid_argument("load_image_batch: resize dimensions must be positive");
  const auto count = static_cast<std::int64_t>(rows.size());
  for (std::size_t row : rows)
    if (row >= images.size()) throw std::out_of_range("load_image_batch: row index out of range");
  Tensor batch = empty({count, height, width, 3}, DType::Float32);
  Tensor labels = empty({count}, DType::Int64);
  const std::int64_t stride = height * width * 3;
  float* dst = batch.data_ptr<float>();
  std::int64_t* lab = labels.data_ptr<std::int64_t>();
  parallel_for(0, count, 1, [&](std::int64_t lo, std::int64_t hi) {
    for (std::int64_t i = lo; i < hi; ++i) {
      const ImageFile& file = images[rows[static_cast<std::size_t>(i)]];
      const Tensor one = load_image_rgb(file.path, height, width);
      std::copy_n(one.data_ptr<float>(), stride, dst + i * stride);
      lab[i] = file.label;
    }
  });
  return {batch, labels};
}

ImageDataset load_image_folder(const std::string& root, std::int64_t height, std::int64_t width,
                               std::vector<std::string> class_names, ImageFolder* folder) {
  ImageFolder scan = scan_image_folder(root, std::move(class_names));
  std::vector<std::size_t> rows(scan.images.size());
  for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = i;
  ImageDataset data = load_image_batch(scan.images, rows, height, width);
  if (folder) *folder = std::move(scan);
  return data;
}

}  // namespace datamunge::nn
