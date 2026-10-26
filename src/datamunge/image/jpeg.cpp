#include <datamunge/image/jpeg.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// JPEG decoder following ITU T.81 and, wherever the standard leaves freedom (IDCT arithmetic,
// upsampling, color conversion, error tolerance), libjpeg-turbo's behaviour so that the output is
// bit-identical to libjpeg-turbo / TensorFlow.

namespace datamunge::image {

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("decode_jpeg: " + message); }

// Zigzag position -> natural (row-major) index, padded with 16 extra entries like libjpeg's
// jpeg_natural_order so a corrupt run length lands on coefficient 63 instead of out of bounds.
constexpr int kNaturalOrder[64 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
    6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
    39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

// libjpeg's AAN scale factors (jddctmgr.c), scaled by 2^14, for the fast IDCT's quant tables.
constexpr int kAanScales[64] = {
    16384, 22725, 21407, 19266, 16384, 12873, 8867,  4520,  22725, 31521, 29692, 26722, 22725, 17855, 12299, 6270,
    21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906,  19266, 26722, 25172, 22654, 19266, 15137, 10426, 5315,
    16384, 22725, 21407, 19266, 16384, 12873, 8867,  4520,  12873, 17855, 16819, 15137, 12873, 10114, 6967,  3552,
    8867,  12299, 11585, 10426, 8867,  6967,  4799,  2446,  4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247};

std::uint8_t clamp255(int v) { return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// libjpeg's post-IDCT range limit: a 10-bit wrapped value centred on 128, then clamped.
struct IdctLimit {
  std::uint8_t table[1024];
  IdctLimit() {
    for (int i = 0; i < 1024; ++i) table[i] = clamp255((i < 512 ? i : i - 1024) + 128);
  }
};
const IdctLimit& idct_limit() {
  static const IdctLimit limit;
  return limit;
}

// ---------------------------------------------------------------------------------------------
// Inverse DCTs (libjpeg jidctint.c "islow" and jidctfst.c "ifast", 8-bit samples).

void idct_islow(const std::int16_t* in, const int* quant, std::uint8_t* out, int stride) {
  constexpr int kConstBits = 13, kPass1Bits = 2;
  auto descale = [](long long x, int n) { return static_cast<int>((x + (1LL << (n - 1))) >> n); };
  constexpr long long F0298 = 2446, F0390 = 3196, F0541 = 4433, F0765 = 6270, F0899 = 7373, F1175 = 9633,
                      F1501 = 12299, F1847 = 15137, F1961 = 16069, F2053 = 16819, F2562 = 20995, F3072 = 25172;
  int ws[64];
  for (int c = 0; c < 8; ++c) {
    const std::int16_t* ip = in + c;
    const int* qp = quant + c;
    auto deq = [&](int row) { return static_cast<long long>(ip[row * 8]) * qp[row * 8]; };
    long long z2 = deq(2), z3 = deq(6);
    long long z1 = (z2 + z3) * F0541;
    long long tmp2 = z1 + z3 * -F1847, tmp3 = z1 + z2 * F0765;
    z2 = deq(0);
    z3 = deq(4);
    long long tmp0 = (z2 + z3) * (1LL << kConstBits), tmp1 = (z2 - z3) * (1LL << kConstBits);
    const long long tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    tmp0 = deq(7);
    tmp1 = deq(5);
    tmp2 = deq(3);
    tmp3 = deq(1);
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    long long z4 = tmp1 + tmp3;
    const long long z5 = (z3 + z4) * F1175;
    tmp0 *= F0298;
    tmp1 *= F2053;
    tmp2 *= F3072;
    tmp3 *= F1501;
    z1 *= -F0899;
    z2 *= -F2562;
    z3 *= -F1961;
    z4 *= -F0390;
    z3 += z5;
    z4 += z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    constexpr int n = kConstBits - kPass1Bits;
    ws[c + 0] = descale(tmp10 + tmp3, n);
    ws[c + 56] = descale(tmp10 - tmp3, n);
    ws[c + 8] = descale(tmp11 + tmp2, n);
    ws[c + 48] = descale(tmp11 - tmp2, n);
    ws[c + 16] = descale(tmp12 + tmp1, n);
    ws[c + 40] = descale(tmp12 - tmp1, n);
    ws[c + 24] = descale(tmp13 + tmp0, n);
    ws[c + 32] = descale(tmp13 - tmp0, n);
  }
  const std::uint8_t* limit = idct_limit().table;
  for (int r = 0; r < 8; ++r) {
    const int* w = ws + r * 8;
    std::uint8_t* op = out + static_cast<std::ptrdiff_t>(r) * stride;
    long long z2 = w[2], z3 = w[6];
    long long z1 = (z2 + z3) * F0541;
    long long tmp2 = z1 + z3 * -F1847, tmp3 = z1 + z2 * F0765;
    long long tmp0 = (static_cast<long long>(w[0]) + w[4]) * (1LL << kConstBits);
    long long tmp1 = (static_cast<long long>(w[0]) - w[4]) * (1LL << kConstBits);
    const long long tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    tmp0 = w[7];
    tmp1 = w[5];
    tmp2 = w[3];
    tmp3 = w[1];
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    long long z4 = tmp1 + tmp3;
    const long long z5 = (z3 + z4) * F1175;
    tmp0 *= F0298;
    tmp1 *= F2053;
    tmp2 *= F3072;
    tmp3 *= F1501;
    z1 *= -F0899;
    z2 *= -F2562;
    z3 *= -F1961;
    z4 *= -F0390;
    z3 += z5;
    z4 += z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    constexpr int n = kConstBits + kPass1Bits + 3;
    op[0] = limit[descale(tmp10 + tmp3, n) & 1023];
    op[7] = limit[descale(tmp10 - tmp3, n) & 1023];
    op[1] = limit[descale(tmp11 + tmp2, n) & 1023];
    op[6] = limit[descale(tmp11 - tmp2, n) & 1023];
    op[2] = limit[descale(tmp12 + tmp1, n) & 1023];
    op[5] = limit[descale(tmp12 - tmp1, n) & 1023];
    op[3] = limit[descale(tmp13 + tmp0, n) & 1023];
    op[4] = limit[descale(tmp13 - tmp0, n) & 1023];
  }
}

void idct_ifast(const std::int16_t* in, const int* quant, std::uint8_t* out, int stride) {
  // CONST_BITS 8; DESCALE is a plain (truncating) shift in jidctfst.c.
  auto mul = [](long long v, long long c) { return (v * c) >> 8; };
  constexpr long long F1082 = 277, F1414 = 362, F1847 = 473, F2613 = 669;
  long long ws[64];
  for (int c = 0; c < 8; ++c) {
    const std::int16_t* ip = in + c;
    const int* qp = quant + c;
    auto deq = [&](int row) { return static_cast<long long>(ip[row * 8]) * qp[row * 8]; };
    long long tmp0 = deq(0), tmp1 = deq(2), tmp2 = deq(4), tmp3 = deq(6);
    long long tmp10 = tmp0 + tmp2, tmp11 = tmp0 - tmp2;
    long long tmp13 = tmp1 + tmp3, tmp12 = mul(tmp1 - tmp3, F1414) - tmp13;
    tmp0 = tmp10 + tmp13;
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;
    long long tmp4 = deq(1), tmp5 = deq(3), tmp6 = deq(5), tmp7 = deq(7);
    const long long z13 = tmp6 + tmp5, z10 = tmp6 - tmp5, z11 = tmp4 + tmp7, z12 = tmp4 - tmp7;
    tmp7 = z11 + z13;
    tmp11 = mul(z11 - z13, F1414);
    const long long z5 = mul(z10 + z12, F1847);
    tmp10 = mul(z12, F1082) - z5;
    tmp12 = mul(z10, -F2613) + z5;
    tmp6 = tmp12 - tmp7;
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 + tmp5;
    ws[c + 0] = tmp0 + tmp7;
    ws[c + 56] = tmp0 - tmp7;
    ws[c + 8] = tmp1 + tmp6;
    ws[c + 48] = tmp1 - tmp6;
    ws[c + 16] = tmp2 + tmp5;
    ws[c + 40] = tmp2 - tmp5;
    ws[c + 32] = tmp3 + tmp4;
    ws[c + 24] = tmp3 - tmp4;
  }
  const std::uint8_t* limit = idct_limit().table;
  for (int r = 0; r < 8; ++r) {
    const long long* w = ws + r * 8;
    std::uint8_t* op = out + static_cast<std::ptrdiff_t>(r) * stride;
    long long tmp10 = w[0] + w[4], tmp11 = w[0] - w[4];
    long long tmp13 = w[2] + w[6], tmp12 = mul(w[2] - w[6], F1414) - tmp13;
    long long tmp0 = tmp10 + tmp13, tmp3 = tmp10 - tmp13, tmp1 = tmp11 + tmp12, tmp2 = tmp11 - tmp12;
    const long long z13 = w[5] + w[3], z10 = w[5] - w[3], z11 = w[1] + w[7], z12 = w[1] - w[7];
    long long tmp7 = z11 + z13;
    tmp11 = mul(z11 - z13, F1414);
    const long long z5 = mul(z10 + z12, F1847);
    tmp10 = mul(z12, F1082) - z5;
    tmp12 = mul(z10, -F2613) + z5;
    const long long tmp6 = tmp12 - tmp7, tmp5 = tmp11 - tmp6, tmp4 = tmp10 + tmp5;
    // IDESCALE(x, PASS1_BITS + 3) is a plain arithmetic shift by 5.
    op[0] = limit[((tmp0 + tmp7) >> 5) & 1023];
    op[7] = limit[((tmp0 - tmp7) >> 5) & 1023];
    op[1] = limit[((tmp1 + tmp6) >> 5) & 1023];
    op[6] = limit[((tmp1 - tmp6) >> 5) & 1023];
    op[2] = limit[((tmp2 + tmp5) >> 5) & 1023];
    op[5] = limit[((tmp2 - tmp5) >> 5) & 1023];
    op[4] = limit[((tmp3 + tmp4) >> 5) & 1023];
    op[3] = limit[((tmp3 - tmp4) >> 5) & 1023];
  }
}

// ---------------------------------------------------------------------------------------------
// Huffman tables and the entropy-coded-segment bit reader.

struct HuffTable {
  bool present = false;
  std::uint8_t values[256] = {};
  int maxcode[18] = {};     // largest code of each length, -1 if none (maxcode[17] sentinel)
  int valoffset[18] = {};   // values index = code + valoffset[length]
  std::uint16_t fast[512] = {};  // 9-bit lookahead: (length << 8) | symbol, 0 when longer

  void build(const std::uint8_t counts[16], const std::uint8_t* vals, int total, bool dc) {
    std::copy(vals, vals + total, values);
    int code = 0, k = 0;
    std::fill(std::begin(fast), std::end(fast), std::uint16_t{0});
    for (int len = 1; len <= 16; ++len) {
      const int n = counts[len - 1];
      valoffset[len] = k - code;
      if (n > 0) {
        for (int i = 0; i < n; ++i, ++k, ++code) {
          if (code >= (1 << len)) fail("bad Huffman table");
          if (len <= 9) {
            const int shift = 9 - len;
            for (int j = 0; j < (1 << shift); ++j) fast[(code << shift) | j] = static_cast<std::uint16_t>((len << 8) | values[k]);
          }
        }
        maxcode[len] = code - 1;
      } else {
        maxcode[len] = -1;
      }
      code <<= 1;
    }
    maxcode[17] = 0x7FFFFFFF;
    if (dc) {
      for (int i = 0; i < total; ++i)
        if (values[i] > 15) fail("bad DC Huffman table");
    }
    present = true;
  }
};

class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t size, std::size_t pos) : d_(data), n_(size), pos_(pos) {}

  std::size_t position() const { return pos_; }

  // Drops buffered bits and any marker state (used at restart markers).
  void reset(std::size_t pos) {
    pos_ = pos;
    buf_ = 0;
    cnt_ = 0;
    fake_ = 0;
    marker_ = false;
  }

  int get_bits(int n) {
    if (n == 0) return 0;
    if (cnt_ < n) fill();
    cnt_ -= n;
    return static_cast<int>((buf_ >> cnt_) & ((1u << n) - 1));
  }

  int get_bit() { return get_bits(1); }

  int decode(const HuffTable& t) {
    if (cnt_ < 16) fill();
    const int look = static_cast<int>((buf_ >> (cnt_ - 9)) & 511);
    const int entry = t.fast[look];
    if (entry) {
      cnt_ -= entry >> 8;
      return entry & 255;
    }
    int len = 10;
    int code = static_cast<int>((buf_ >> (cnt_ - 10)) & 1023);
    while (code > t.maxcode[len]) {
      if (++len > 16) fail("corrupt data (bad Huffman code)");
      code = static_cast<int>((buf_ >> (cnt_ - len)) & ((1u << len) - 1));
    }
    cnt_ -= len;
    return t.values[(code + t.valoffset[len]) & 255];
  }

  // Signed value of an s-bit magnitude category (HUFF_EXTEND).
  int receive_extend(int s) {
    if (s == 0) return 0;
    if (s > 16) fail("corrupt data (bad coefficient size)");
    const int v = get_bits(s);
    return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v;
  }

  // Throws when decoding consumed bits invented past the physical end of the file.
  void check_truncation() const {
    if (cnt_ < fake_) fail("unexpected end of data (truncated file)");
  }

 private:
  void fill() {
    while (cnt_ <= 56) {
      unsigned byte = 0;
      if (marker_) {
        byte = 0;  // libjpeg zero-fills after a marker
      } else if (pos_ >= n_) {
        fake_ += 8;
      } else {
        byte = d_[pos_];
        if (byte == 0xFF) {
          std::size_t p = pos_ + 1;
          while (p < n_ && d_[p] == 0xFF) ++p;
          if (p >= n_) {
            pos_ = n_;
            fake_ += 8;
            byte = 0;
          } else if (d_[p] == 0x00) {
            pos_ = p + 1;  // stuffed 0xFF
          } else {
            pos_ = p - 1;  // leave the marker for the caller
            marker_ = true;
            byte = 0;
          }
        } else {
          ++pos_;
        }
      }
      buf_ = (buf_ << 8) | byte;
      cnt_ += 8;
      if (fake_ > 64) fake_ = 64;  // keep the counter bounded; any value > cnt_ means truncated
    }
  }

  const std::uint8_t* d_;
  std::size_t n_;
  std::size_t pos_;
  std::uint64_t buf_ = 0;
  int cnt_ = 0;
  int fake_ = 0;
  bool marker_ = false;
};

// ---------------------------------------------------------------------------------------------
// libjpeg color conversion tables (jdcolor.c, SCALEBITS 16).

struct ColorTables {
  int cr_r[256], cb_b[256];
  long long cr_g[256], cb_g[256];
  ColorTables() {
    constexpr int kScale = 16;
    constexpr long long kHalf = 1LL << (kScale - 1);
    auto fix = [](double x) { return static_cast<long long>(x * (1LL << 16) + 0.5); };
    for (int i = 0; i < 256; ++i) {
      const long long x = i - 128;
      cr_r[i] = static_cast<int>((fix(1.40200) * x + kHalf) >> kScale);
      cb_b[i] = static_cast<int>((fix(1.77200) * x + kHalf) >> kScale);
      cr_g[i] = -fix(0.71414) * x;
      cb_g[i] = -fix(0.34414) * x + kHalf;
    }
  }
};
const ColorTables& color_tables() {
  static const ColorTables tables;
  return tables;
}

// ---------------------------------------------------------------------------------------------
// The decoder proper.

struct Component {
  int id = 0, h = 1, v = 1, tq = 0;
  int bw = 0, bh = 0;          // allocated (MCU-padded) blocks per row / column
  int dw = 0, dh = 0;          // downsampled size in samples
  int dc_table = 0, ac_table = 0;
  int dc_pred = 0;
  bool dc_seen = false;        // a DC scan (first or sequential) has covered this component
  std::vector<std::int16_t> coef;  // bw * bh blocks of 64 coefficients, natural order
};

class Decoder {
 public:
  Decoder(const std::uint8_t* data, std::size_t size) : d_(data), n_(size) {}

  // Walks the marker stream; stops at the first SOS when header_only.
  void run(bool header_only) {
    if (n_ < 3 || d_[0] != 0xFF || d_[1] != 0xD8) fail("not a JPEG file (missing SOI marker)");
    pos_ = 2;
    for (;;) {
      const int m = next_marker();
      // A missing EOI means a truncated file (TensorFlow rejects it too, even after the last scan).
      if (m < 0) fail("unexpected end of data (truncated file)");
      if (m == 0xD9) {  // EOI
        if (!frame_) fail("no frame before EOI");
        if (header_only) return;
        break;
      }
      if (m >= 0xD0 && m <= 0xD7) continue;  // stray RST (libjpeg: warning)
      if (m == 0x01) continue;                // TEM
      switch (m) {
        case 0xC0:
        case 0xC1:
        case 0xC2:
          read_frame(m);
          break;
        case 0xC4:
          read_dht();
          break;
        case 0xCC:
          fail("arithmetic-coded JPEG is not supported");
        case 0xDB:
          read_dqt();
          break;
        case 0xDD: {
          const std::size_t end = segment();
          if (end - pos_ < 2) fail("bad DRI segment");
          info_.restart_interval = (d_[pos_] << 8) | d_[pos_ + 1];
          pos_ = end;
          break;
        }
        case 0xDA:
          if (!frame_) fail("SOS before SOF");
          if (header_only) return;
          read_scan();
          break;
        case 0xE0:
        case 0xE1:
        case 0xEE: {
          const std::size_t end = segment();
          read_app(m, pos_, end);
          pos_ = end;
          break;
        }
        case 0xDC:
          fail("DNL marker (height defined after the first scan) is not supported");
        default:
          if (m >= 0xC3 && m <= 0xCF) fail("unsupported JPEG process (lossless, hierarchical or arithmetic)");
          pos_ = segment();  // APPn, COM and other segments with a length
      }
    }
    if (scans_ == 0) fail("no image data");
  }

  JpegInfo info() const {
    JpegInfo out = info_;
    out.color_space = color_space();
    return out;
  }

  Image render(const JpegDecodeOptions& options) {
    const JpegColorSpace cs = color_space();
    int channels = options.channels;
    if (channels != 0 && channels != 1 && channels != 3) throw std::invalid_argument("decode_jpeg: channels must be 0, 1 or 3");
    if (channels == 0) channels = cs == JpegColorSpace::Grayscale ? 1 : 3;
    if (channels == 1 && (cs == JpegColorSpace::CMYK || cs == JpegColorSpace::YCCK))
      fail("grayscale output from CMYK/YCCK is not supported");

    // libjpeg only processes the luma component when converting YCbCr/gray to grayscale.
    const bool luma_only = channels == 1 && (cs == JpegColorSpace::Grayscale || cs == JpegColorSpace::YCbCr);
    const int used = luma_only ? 1 : static_cast<int>(comps_.size());
    std::vector<std::vector<std::uint8_t>> planes(static_cast<std::size_t>(used));
    for (int c = 0; c < used; ++c) planes[static_cast<std::size_t>(c)] = component_plane(comps_[static_cast<std::size_t>(c)], options);

    const int w = info_.width, h = info_.height;
    const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    Image img(w, h, channels == 1 ? ImageMode::Grayscale : ImageMode::RGB);
    std::uint8_t* out = img.data().data();
    if (channels == 1) {
      if (luma_only) {
        std::copy(planes[0].begin(), planes[0].end(), out);
      } else {  // RGB -> gray, libjpeg rgb_gray_convert
        for (std::size_t i = 0; i < npix; ++i)
          out[i] = static_cast<std::uint8_t>((planes[0][i] * 19595LL + planes[1][i] * 38470LL + planes[2][i] * 7471LL + 32768) >> 16);
      }
      return img;
    }
    const ColorTables& t = color_tables();
    switch (cs) {
      case JpegColorSpace::Grayscale:
        for (std::size_t i = 0; i < npix; ++i) out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = planes[0][i];
        break;
      case JpegColorSpace::RGB:
        for (std::size_t i = 0; i < npix; ++i)
          for (int c = 0; c < 3; ++c) out[3 * i + static_cast<std::size_t>(c)] = planes[static_cast<std::size_t>(c)][i];
        break;
      case JpegColorSpace::YCbCr:
        for (std::size_t i = 0; i < npix; ++i) {
          const int y = planes[0][i], cb = planes[1][i], cr = planes[2][i];
          out[3 * i] = clamp255(y + t.cr_r[cr]);
          out[3 * i + 1] = clamp255(y + static_cast<int>((t.cb_g[cb] + t.cr_g[cr]) >> 16));
          out[3 * i + 2] = clamp255(y + t.cb_b[cb]);
        }
        break;
      case JpegColorSpace::CMYK:
      case JpegColorSpace::YCCK:
        for (std::size_t i = 0; i < npix; ++i) {
          int c = planes[0][i], m = planes[1][i], ye = planes[2][i];
          const int k = planes[3][i];
          if (cs == JpegColorSpace::YCCK) {  // libjpeg ycck_cmyk_convert
            const int y = c, cb = m, cr = ye;
            c = clamp255(255 - (y + t.cr_r[cr]));
            m = clamp255(255 - (y + static_cast<int>((t.cb_g[cb] + t.cr_g[cr]) >> 16)));
            ye = clamp255(255 - (y + t.cb_b[cb]));
          }
          // TensorFlow's CMYK -> RGB (Adobe files store inverted CMYK).
          if (info_.adobe) {
            out[3 * i] = static_cast<std::uint8_t>(k * c / 255);
            out[3 * i + 1] = static_cast<std::uint8_t>(k * m / 255);
            out[3 * i + 2] = static_cast<std::uint8_t>(k * ye / 255);
          } else {
            out[3 * i] = static_cast<std::uint8_t>((255 - k) * (255 - c) / 255);
            out[3 * i + 1] = static_cast<std::uint8_t>((255 - k) * (255 - m) / 255);
            out[3 * i + 2] = static_cast<std::uint8_t>((255 - k) * (255 - ye) / 255);
          }
        }
        break;
    }
    return img;
  }

 private:
  // ----- marker-level parsing -----

  int next_marker() {
    for (;;) {
      while (pos_ < n_ && d_[pos_] != 0xFF) ++pos_;
      while (pos_ < n_ && d_[pos_] == 0xFF) ++pos_;
      if (pos_ >= n_) return -1;
      const int m = d_[pos_++];
      if (m != 0x00) return m;
    }
  }

  // Reads a segment length at pos_, leaves pos_ at the payload and returns the payload end.
  std::size_t segment() {
    if (n_ - pos_ < 2) fail("unexpected end of data (truncated segment)");
    const std::size_t len = (static_cast<std::size_t>(d_[pos_]) << 8) | d_[pos_ + 1];
    if (len < 2) fail("bad segment length");
    if (len > n_ - pos_) fail("unexpected end of data (truncated segment)");
    const std::size_t end = pos_ + len;
    pos_ += 2;
    return end;
  }

  void read_app(int m, std::size_t p, std::size_t end) {
    const std::size_t len = end - p;
    const std::uint8_t* s = d_ + p;
    if (m == 0xE0 && len >= 14 && std::memcmp(s, "JFIF\0", 5) == 0) info_.jfif = true;
    if (m == 0xEE && len >= 12 && std::memcmp(s, "Adobe", 5) == 0) {
      info_.adobe = true;
      info_.adobe_transform = s[11];
    }
    if (m == 0xE1 && !exif_seen_ && len >= 14 && std::memcmp(s, "Exif\0\0", 6) == 0) {
      exif_seen_ = true;
      read_exif_orientation(s + 6, len - 6);
    }
  }

  void read_exif_orientation(const std::uint8_t* t, std::size_t len) {
    if (len < 8) return;
    const bool le = t[0] == 'I' && t[1] == 'I';
    if (!le && !(t[0] == 'M' && t[1] == 'M')) return;
    auto u16 = [&](std::size_t o) -> unsigned { return le ? t[o] | (t[o + 1] << 8) : (t[o] << 8) | t[o + 1]; };
    auto u32 = [&](std::size_t o) -> std::size_t {
      return le ? (static_cast<std::size_t>(u16(o + 2)) << 16) | u16(o) : (static_cast<std::size_t>(u16(o)) << 16) | u16(o + 2);
    };
    if (u16(2) != 42) return;
    const std::size_t ifd = u32(4);
    if (ifd > len || len - ifd < 2) return;
    const unsigned count = u16(ifd);
    for (unsigned i = 0; i < count; ++i) {
      const std::size_t e = ifd + 2 + 12 * static_cast<std::size_t>(i);
      if (e + 12 > len) return;
      if (u16(e) == 0x0112 && u16(e + 2) == 3) {
        const unsigned v = u16(e + 8);
        if (v >= 1 && v <= 8) info_.orientation = static_cast<int>(v);
        return;
      }
    }
  }

  void read_dqt() {
    const std::size_t end = segment();
    while (pos_ < end) {
      const int pq = d_[pos_] >> 4, tq = d_[pos_] & 15;
      ++pos_;
      if (tq > 3 || pq > 1) fail("bad DQT segment");
      const std::size_t need = pq ? 128 : 64;
      if (end - pos_ < need) fail("bad DQT segment");
      for (int i = 0; i < 64; ++i) {
        const int v = pq ? (d_[pos_ + 2 * i] << 8) | d_[pos_ + 2 * i + 1] : d_[pos_ + i];
        quant_[tq][kNaturalOrder[i]] = v;
      }
      quant_present_[tq] = true;
      pos_ += need;
    }
    pos_ = end;
  }

  void read_dht() {
    const std::size_t end = segment();
    while (pos_ < end) {
      if (end - pos_ < 17) fail("bad DHT segment");
      const int tc = d_[pos_] >> 4, th = d_[pos_] & 15;
      if (tc > 1 || th > 3) fail("bad DHT segment");
      const std::uint8_t* counts = d_ + pos_ + 1;
      int total = 0;
      for (int i = 0; i < 16; ++i) total += counts[i];
      if (total > 256 || end - pos_ - 17 < static_cast<std::size_t>(total)) fail("bad DHT segment");
      (tc ? ac_[th] : dc_[th]).build(counts, d_ + pos_ + 17, total, tc == 0);
      pos_ += 17 + static_cast<std::size_t>(total);
    }
    pos_ = end;
  }

  void read_frame(int m) {
    const std::size_t end = segment();
    if (frame_) fail("multiple frames");
    if (end - pos_ < 6) fail("bad SOF segment");
    const int precision = d_[pos_];
    const int height = (d_[pos_ + 1] << 8) | d_[pos_ + 2];
    const int width = (d_[pos_ + 3] << 8) | d_[pos_ + 4];
    const int nc = d_[pos_ + 5];
    if (precision != 8) fail("only 8-bit JPEGs are supported (precision " + std::to_string(precision) + ")");
    if (height == 0) fail("DNL-defined image height is not supported");
    if (width == 0) fail("bad image width");
    if (nc != 1 && nc != 3 && nc != 4) fail("unsupported component count " + std::to_string(nc));
    if (end - pos_ != 6 + 3 * static_cast<std::size_t>(nc)) fail("bad SOF segment length");
    comps_.resize(static_cast<std::size_t>(nc));
    for (int i = 0; i < nc; ++i) {
      Component& c = comps_[static_cast<std::size_t>(i)];
      const std::uint8_t* p = d_ + pos_ + 6 + 3 * i;
      c.id = p[0];
      c.h = p[1] >> 4;
      c.v = p[1] & 15;
      c.tq = p[2];
      if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) fail("bad component parameters");
      for (int j = 0; j < i; ++j)
        if (comps_[static_cast<std::size_t>(j)].id == c.id) fail("duplicate component id");
    }
    pos_ = end;
    frame_ = true;
    info_.width = width;
    info_.height = height;
    info_.components = nc;
    info_.progressive = m == 0xC2;
    for (const auto& c : comps_) {
      hmax_ = std::max(hmax_, c.h);
      vmax_ = std::max(vmax_, c.v);
      info_.h_sampling.push_back(c.h);
      info_.v_sampling.push_back(c.v);
    }
    mcux_ = (width + 8 * hmax_ - 1) / (8 * hmax_);
    mcuy_ = (height + 8 * vmax_ - 1) / (8 * vmax_);
    for (auto& c : comps_) {
      c.dw = static_cast<int>((static_cast<long long>(width) * c.h + hmax_ - 1) / hmax_);
      c.dh = static_cast<int>((static_cast<long long>(height) * c.v + vmax_ - 1) / vmax_);
      c.bw = mcux_ * c.h;
      c.bh = mcuy_ * c.v;
    }
  }

  JpegColorSpace color_space() const {
    if (comps_.size() == 1) return JpegColorSpace::Grayscale;
    if (comps_.size() == 3) {
      if (info_.jfif) return JpegColorSpace::YCbCr;
      if (info_.adobe) return info_.adobe_transform == 0 ? JpegColorSpace::RGB : JpegColorSpace::YCbCr;
      if (comps_[0].id == 'R' && comps_[1].id == 'G' && comps_[2].id == 'B') return JpegColorSpace::RGB;
      return JpegColorSpace::YCbCr;
    }
    if (info_.adobe) return info_.adobe_transform == 0 ? JpegColorSpace::CMYK : JpegColorSpace::YCCK;
    return JpegColorSpace::CMYK;
  }

  // ----- scan decoding -----

  void ensure_allocated(Component& c) {
    if (!c.coef.empty()) return;
    // Every coded block costs at least one bit, so a component cannot have more blocks than
    // 8x the remaining bytes; this stops tiny corrupt headers from requesting gigabytes.
    const long long coded = static_cast<long long>((c.dw + 7) / 8) * ((c.dh + 7) / 8);
    if (coded > 8LL * static_cast<long long>(n_ - pos_) + 4096) fail("image dimensions exceed the available data (truncated or corrupt file)");
    c.coef.assign(static_cast<std::size_t>(c.bw) * static_cast<std::size_t>(c.bh) * 64, 0);
  }

  void read_scan() {
    const std::size_t end = segment();
    if (end - pos_ < 1) fail("bad SOS segment");
    const int ns = d_[pos_];
    if (ns < 1 || ns > 4 || end - pos_ != 4 + 2 * static_cast<std::size_t>(ns)) fail("bad SOS segment");
    std::vector<int> sc;
    int blocks_per_mcu = 0;
    for (int i = 0; i < ns; ++i) {
      const int id = d_[pos_ + 1 + 2 * i], tables = d_[pos_ + 2 + 2 * i];
      int index = -1;
      for (std::size_t j = 0; j < comps_.size(); ++j)
        if (comps_[j].id == id) index = static_cast<int>(j);
      if (index < 0 || std::find(sc.begin(), sc.end(), index) != sc.end()) fail("bad component in SOS");
      Component& c = comps_[static_cast<std::size_t>(index)];
      c.dc_table = tables >> 4;
      c.ac_table = tables & 15;
      if (c.dc_table > 3 || c.ac_table > 3) fail("bad Huffman table index in SOS");
      blocks_per_mcu += c.h * c.v;
      sc.push_back(index);
    }
    const std::uint8_t* p = d_ + pos_ + 1 + 2 * ns;
    ss_ = p[0];
    se_ = p[1];
    ah_ = p[2] >> 4;
    al_ = p[2] & 15;
    pos_ = end;
    if (ns > 1 && blocks_per_mcu > 10) fail("too many blocks per MCU");
    if (info_.progressive) {
      if (ss_ > se_ || se_ > 63 || ah_ > 13 || al_ > 13) fail("bad progressive scan parameters");
      if (ss_ == 0 && se_ != 0) fail("bad progressive scan parameters (DC scan with AC coefficients)");
      if (ss_ > 0 && ns != 1) fail("bad progressive scan parameters (interleaved AC scan)");
    } else {
      ss_ = 0;
      se_ = 63;
      ah_ = al_ = 0;
    }
    const bool dc_scan = ss_ == 0;
    for (int index : sc) {
      Component& c = comps_[static_cast<std::size_t>(index)];
      if (dc_scan && ah_ == 0 && !dc_[c.dc_table].present) fail("missing DC Huffman table");
      if ((!info_.progressive || !dc_scan) && !ac_[c.ac_table].present) fail("missing AC Huffman table");
      if (!dc_scan && !c.dc_seen) fail("AC scan before the DC scan");
      if (!quant_present_[c.tq]) fail("missing quantization table");
      ensure_allocated(c);
      if (dc_scan) c.dc_seen = true;
      c.dc_pred = 0;
    }
    eobrun_ = 0;
    BitReader br(d_, n_, pos_);

    // MCU geometry: a single-component scan walks that component's own blocks.
    const bool interleaved = ns > 1;
    const Component& c0 = comps_[static_cast<std::size_t>(sc[0])];
    const int cols = interleaved ? mcux_ : (c0.dw + 7) / 8;
    const int rows = interleaved ? mcuy_ : (c0.dh + 7) / 8;
    const long long total = static_cast<long long>(cols) * rows;
    const int ri = info_.restart_interval;
    int to_go = ri;
    for (long long mcu = 0; mcu < total; ++mcu) {
      if (ri > 0) {
        if (to_go == 0) {
          restart(br);
          for (int index : sc) comps_[static_cast<std::size_t>(index)].dc_pred = 0;
          eobrun_ = 0;
          to_go = ri;
        }
        --to_go;
      }
      const int my = static_cast<int>(mcu / cols), mx = static_cast<int>(mcu % cols);
      if (!interleaved) {
        Component& c = comps_[static_cast<std::size_t>(sc[0])];
        decode_block(br, c, c.coef.data() + (static_cast<std::size_t>(my) * c.bw + mx) * 64);
      } else {
        for (int index : sc) {
          Component& c = comps_[static_cast<std::size_t>(index)];
          for (int v = 0; v < c.v; ++v)
            for (int h = 0; h < c.h; ++h) {
              const std::size_t by = static_cast<std::size_t>(my) * c.v + v, bx = static_cast<std::size_t>(mx) * c.h + h;
              decode_block(br, c, c.coef.data() + (by * c.bw + bx) * 64);
            }
        }
      }
      br.check_truncation();
    }
    pos_ = br.position();
    ++scans_;
  }

  // Consumes the next RSTn marker (libjpeg tolerates other markers by zero-filling).
  void restart(BitReader& br) {
    std::size_t p = br.position();
    for (;;) {
      while (p < n_ && d_[p] != 0xFF) ++p;
      std::size_t q = p;
      while (q < n_ && d_[q] == 0xFF) ++q;
      if (q >= n_) fail("unexpected end of data (truncated file)");
      if (d_[q] == 0x00) {
        p = q + 1;
        continue;
      }
      // Not a restart marker: leave it in place; the rest of the scan reads as zeros (libjpeg).
      br.reset((d_[q] >= 0xD0 && d_[q] <= 0xD7) ? q + 1 : q - 1);
      return;
    }
  }

  static void add_dc(Component& c, int diff) {
    const long long next = static_cast<long long>(c.dc_pred) + diff;
    if (next > 2147483647LL || next < -2147483647LL) fail("corrupt data (DC coefficient overflow)");
    c.dc_pred = static_cast<int>(next);
  }

  void decode_block(BitReader& br, Component& c, std::int16_t* blk) {
    if (!info_.progressive) {
      add_dc(c, br.receive_extend(br.decode(dc_[c.dc_table])));
      blk[0] = static_cast<std::int16_t>(c.dc_pred);
      const HuffTable& ac = ac_[c.ac_table];
      for (int k = 1; k < 64; ++k) {
        const int rs = br.decode(ac);
        const int r = rs >> 4, s2 = rs & 15;
        if (s2) {
          k += r;
          blk[kNaturalOrder[k]] = static_cast<std::int16_t>(br.receive_extend(s2));
        } else {
          if (r != 15) break;
          k += 15;
        }
      }
      return;
    }
    if (ss_ == 0) {
      if (ah_ == 0) {  // DC first
        add_dc(c, br.receive_extend(br.decode(dc_[c.dc_table])));
        blk[0] = static_cast<std::int16_t>(static_cast<unsigned>(c.dc_pred) << al_);
      } else if (br.get_bit()) {  // DC refine
        blk[0] = static_cast<std::int16_t>(blk[0] | (1 << al_));
      }
      return;
    }
    const HuffTable& ac = ac_[c.ac_table];
    if (ah_ == 0) {  // AC first
      if (eobrun_ > 0) {
        --eobrun_;
        return;
      }
      for (int k = ss_; k <= se_; ++k) {
        const int rs = br.decode(ac);
        const int r = rs >> 4, s = rs & 15;
        if (s) {
          k += r;
          blk[kNaturalOrder[k]] = static_cast<std::int16_t>(static_cast<unsigned>(br.receive_extend(s)) << al_);
        } else if (r == 15) {
          k += 15;
        } else {
          eobrun_ = 1 << r;
          if (r) eobrun_ += br.get_bits(r);
          --eobrun_;
          break;
        }
      }
      return;
    }
    // AC refine (jdphuff.c decode_mcu_AC_refine).
    const int p1 = 1 << al_, m1 = -1 * (1 << al_);
    int k = ss_;
    auto refine = [&](std::int16_t& coef) {
      if (br.get_bit() && (coef & p1) == 0) coef = static_cast<std::int16_t>(coef >= 0 ? coef + p1 : coef + m1);
    };
    if (eobrun_ == 0) {
      for (; k <= se_; ++k) {
        const int rs = br.decode(ac);
        int r = rs >> 4, s = rs & 15;
        if (s) {
          s = br.get_bit() ? p1 : m1;  // s != 1 is a libjpeg warning, not an error
        } else if (r != 15) {
          eobrun_ = 1 << r;
          if (r) eobrun_ += br.get_bits(r);
          break;
        }
        do {
          std::int16_t& coef = blk[kNaturalOrder[k]];
          if (coef != 0) {
            refine(coef);
          } else {
            if (--r < 0) break;
          }
          ++k;
        } while (k <= se_);
        if (s) blk[kNaturalOrder[k]] = static_cast<std::int16_t>(s);
      }
    }
    if (eobrun_ > 0) {
      for (; k <= se_; ++k) {
        std::int16_t& coef = blk[kNaturalOrder[k]];
        if (coef != 0) refine(coef);
      }
      --eobrun_;
    }
  }

  // ----- reconstruction -----

  // IDCT of one component, then upsampling to the full image size.
  std::vector<std::uint8_t> component_plane(Component& c, const JpegDecodeOptions& options) {
    if (hmax_ % c.h != 0 || vmax_ % c.v != 0) fail("non-integral chroma sampling ratios are not supported");
    if (c.coef.empty()) c.coef.assign(static_cast<std::size_t>(c.bw) * static_cast<std::size_t>(c.bh) * 64, 0);
    const int cols = (c.dw + 7) / 8, rows = (c.dh + 7) / 8;
    const int stride = cols * 8;
    std::vector<std::uint8_t> plane(static_cast<std::size_t>(stride) * static_cast<std::size_t>(rows) * 8);
    const int* q = quant_[c.tq];
    int scaled[64];
    const bool fast = options.dct_method == JpegDctMethod::IntegerFast;
    if (fast)
      for (int i = 0; i < 64; ++i) scaled[i] = static_cast<std::int16_t>((static_cast<long long>(q[i]) * kAanScales[i] + (1 << 11)) >> 12);
    for (int by = 0; by < rows; ++by)
      for (int bx = 0; bx < cols; ++bx) {
        const std::int16_t* blk = c.coef.data() + (static_cast<std::size_t>(by) * c.bw + bx) * 64;
        std::uint8_t* out = plane.data() + static_cast<std::size_t>(by) * 8 * stride + static_cast<std::size_t>(bx) * 8;
        if (fast)
          idct_ifast(blk, scaled, out, stride);
        else
          idct_islow(blk, q, out, stride);
      }

    const int w = info_.width, h = info_.height;
    const int hx = hmax_ / c.h, vy = vmax_ / c.v;
    std::vector<std::uint8_t> full(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    auto in = [&](int y, int x) -> int { return plane[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x)]; };
    const bool fancy = options.fancy_upsampling;
    const int dw = c.dw, dh = c.dh;
    std::vector<int> row(static_cast<std::size_t>(2 * dw + 2));
    if (hx == 1 && vy == 1) {
      for (int y = 0; y < h; ++y) std::copy_n(plane.data() + static_cast<std::size_t>(y) * stride, w, full.data() + static_cast<std::size_t>(y) * w);
    } else if (hx == 2 && vy == 1 && fancy && dw > 2) {  // h2v1_fancy_upsample
      for (int y = 0; y < h; ++y) {
        std::uint8_t* o = full.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < dw; ++x) {
          const int cur = in(y, x) * 3;
          const int left = in(y, x > 0 ? x - 1 : 0), right = in(y, x < dw - 1 ? x + 1 : dw - 1);
          row[static_cast<std::size_t>(2 * x)] = (cur + left + 1) >> 2;
          row[static_cast<std::size_t>(2 * x + 1)] = (cur + right + 2) >> 2;
        }
        for (int x = 0; x < w; ++x) o[x] = static_cast<std::uint8_t>(row[static_cast<std::size_t>(x)]);
      }
    } else if (hx == 1 && vy == 2 && fancy) {  // h1v2_fancy_upsample
      for (int y = 0; y < h; ++y) {
        const int iy = y / 2;
        const int ny = (y % 2 == 0) ? std::max(iy - 1, 0) : std::min(iy + 1, dh - 1);
        const int bias = (y % 2 == 0) ? 1 : 2;
        std::uint8_t* o = full.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) o[x] = static_cast<std::uint8_t>((in(iy, x) * 3 + in(ny, x) + bias) >> 2);
      }
    } else if (hx == 2 && vy == 2 && fancy && dw > 2) {  // h2v2_fancy_upsample
      std::vector<int> colsum(static_cast<std::size_t>(dw));
      for (int y = 0; y < h; ++y) {
        const int iy = y / 2;
        const int ny = (y % 2 == 0) ? std::max(iy - 1, 0) : std::min(iy + 1, dh - 1);
        for (int x = 0; x < dw; ++x) colsum[static_cast<std::size_t>(x)] = in(iy, x) * 3 + in(ny, x);
        for (int x = 0; x < dw; ++x) {
          const int cur = colsum[static_cast<std::size_t>(x)] * 3;
          const int left = colsum[static_cast<std::size_t>(x > 0 ? x - 1 : 0)];
          const int right = colsum[static_cast<std::size_t>(x < dw - 1 ? x + 1 : dw - 1)];
          row[static_cast<std::size_t>(2 * x)] = (cur + left + 8) >> 4;
          row[static_cast<std::size_t>(2 * x + 1)] = (cur + right + 7) >> 4;
        }
        std::uint8_t* o = full.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) o[x] = static_cast<std::uint8_t>(row[static_cast<std::size_t>(x)]);
      }
    } else {  // box replication (h2v1/h2v2 plain, int_upsample)
      for (int y = 0; y < h; ++y) {
        std::uint8_t* o = full.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) o[x] = static_cast<std::uint8_t>(in(y / vy, x / hx));
      }
    }
    return full;
  }

  const std::uint8_t* d_;
  std::size_t n_;
  std::size_t pos_ = 0;
  JpegInfo info_;
  bool frame_ = false;
  bool exif_seen_ = false;
  int scans_ = 0;
  std::vector<Component> comps_;
  int hmax_ = 1, vmax_ = 1, mcux_ = 0, mcuy_ = 0;
  int quant_[4][64] = {};
  bool quant_present_[4] = {};
  HuffTable dc_[4], ac_[4];
  int ss_ = 0, se_ = 63, ah_ = 0, al_ = 0;
  int eobrun_ = 0;
};

std::vector<std::uint8_t> read_file(const std::string& path, const char* who) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::string(who) + ": cannot open " + path);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

bool is_jpeg(const std::uint8_t* data, std::size_t size) {
  return size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
}

JpegInfo jpeg_info(const std::uint8_t* data, std::size_t size) {
  Decoder dec(data, size);
  dec.run(true);
  return dec.info();
}

JpegInfo read_jpeg_info(const std::string& path) {
  const auto bytes = read_file(path, "read_jpeg_info");
  return jpeg_info(bytes.data(), bytes.size());
}

Image decode_jpeg(const std::uint8_t* data, std::size_t size, const JpegDecodeOptions& options, JpegInfo* info) {
  Decoder dec(data, size);
  dec.run(false);
  if (info) *info = dec.info();
  return dec.render(options);
}

Image decode_jpeg(const std::vector<std::uint8_t>& bytes, const JpegDecodeOptions& options, JpegInfo* info) {
  return decode_jpeg(bytes.data(), bytes.size(), options, info);
}

Image read_jpeg(const std::string& path, const JpegDecodeOptions& options, JpegInfo* info) {
  const auto bytes = read_file(path, "read_jpeg");
  return decode_jpeg(bytes.data(), bytes.size(), options, info);
}

Image apply_exif_orientation(const Image& img, int orientation) {
  if (orientation < 1 || orientation > 8) throw std::invalid_argument("apply_exif_orientation: orientation must be 1..8");
  const int w = img.width(), h = img.height(), ch = img.channels();
  const bool swap = orientation >= 5;
  Image out(swap ? h : w, swap ? w : h, img.mode());
  const int ow = out.width(), oh = out.height();
  const std::uint8_t* src = img.data().data();
  std::uint8_t* dst = out.data().data();
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x) {
      int sx = x, sy = y;  // source pixel for output (x, y)
      switch (orientation) {
        case 2: sx = w - 1 - x; break;
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;
        case 4: sy = h - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = h - 1 - x; break;
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;
        case 8: sx = w - 1 - y; sy = x; break;
        default: break;
      }
      std::copy_n(src + (static_cast<std::size_t>(sy) * w + sx) * ch, ch, dst + (static_cast<std::size_t>(y) * ow + x) * ch);
    }
  return out;
}

}  // namespace datamunge::image
