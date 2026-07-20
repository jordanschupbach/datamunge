#include <datamunge/image/detail/zlib_codec.hpp>

#include <datamunge/image/detail/adler32.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace datamunge::image::detail {

namespace {

constexpr int kMaxBits = 15;

// ---- bit-level reader over the raw (post zlib-header) DEFLATE stream ----
//
// DEFLATE packs ordinary multi-bit fields LSB-first (the first bit read becomes the field's
// low bit); Huffman codes are matched MSB-first (the first bit read is the code's high bit) --
// see get_bits() vs. get_bit() below, and decode_symbol()'s use of the latter.
class BitReader {
  public:
    BitReader(const unsigned char* data, std::size_t size) : data_(data), size_(size) {}

    [[nodiscard]] std::uint32_t get_bit() {
        if (bit_count_ == 0) {
            if (byte_pos_ >= size_) {
                throw std::runtime_error("zlib_decompress: unexpected end of compressed data");
            }
            bit_buffer_ = data_[byte_pos_++];
            bit_count_ = 8;
        }
        const std::uint32_t bit = bit_buffer_ & 1u;
        bit_buffer_ >>= 1;
        --bit_count_;
        return bit;
    }

    [[nodiscard]] std::uint32_t get_bits(int n) {
        std::uint32_t value = 0;
        for (int i = 0; i < n; ++i) value |= get_bit() << i;
        return value;
    }

    /// @brief Discards any partial byte currently buffered, so the next read starts at a real
    ///        byte boundary -- required before a stored block's LEN/NLEN/data.
    void align_to_byte() {
        bit_buffer_ = 0;
        bit_count_ = 0;
    }

    [[nodiscard]] unsigned char get_byte() {
        if (byte_pos_ >= size_) {
            throw std::runtime_error("zlib_decompress: unexpected end of compressed data");
        }
        return data_[byte_pos_++];
    }

  private:
    const unsigned char* data_;
    std::size_t size_;
    std::size_t byte_pos_{0};
    std::uint32_t bit_buffer_{0};
    int bit_count_{0};
};

// ---- canonical Huffman decode table (RFC 1951 3.2.2) ----
//
// counts[len] = how many symbols have that code length; symbols[] holds every symbol with a
// nonzero length, grouped by length (and, within a length, in ascending symbol-index order) --
// exactly the order DEFLATE's canonical code assignment produces, so decode_symbol() can
// recover the symbol from (code length, code value) alone without ever materializing the codes
// themselves.
struct HuffmanTable {
    std::array<int, kMaxBits + 1> counts{};
    std::vector<int> symbols;
};

[[nodiscard]] HuffmanTable build_huffman_table(const std::vector<int>& lengths) {
    HuffmanTable table;
    for (const int len : lengths) {
        if (len < 0 || len > kMaxBits) throw std::runtime_error("zlib_decompress: invalid Huffman code length");
        if (len > 0) ++table.counts[static_cast<std::size_t>(len)];
    }

    std::array<int, kMaxBits + 2> offsets{};
    for (int len = 1; len <= kMaxBits; ++len) offsets[static_cast<std::size_t>(len + 1)] = offsets[static_cast<std::size_t>(len)] + table.counts[static_cast<std::size_t>(len)];

    table.symbols.resize(static_cast<std::size_t>(offsets[kMaxBits + 1]));
    for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
        const int len = lengths[symbol];
        if (len > 0) table.symbols[static_cast<std::size_t>(offsets[static_cast<std::size_t>(len)]++)] = static_cast<int>(symbol);
    }
    return table;
}

[[nodiscard]] int decode_symbol(BitReader& reader, const HuffmanTable& table) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= kMaxBits; ++len) {
        code |= static_cast<int>(reader.get_bit());
        const int count = table.counts[static_cast<std::size_t>(len)];
        if (code - first < count) {
            return table.symbols[static_cast<std::size_t>(index + (code - first))];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    throw std::runtime_error("zlib_decompress: invalid Huffman code");
}

[[nodiscard]] HuffmanTable fixed_literal_length_table() {
    std::vector<int> lengths(288);
    for (int i = 0; i < 144; ++i) lengths[static_cast<std::size_t>(i)] = 8;
    for (int i = 144; i < 256; ++i) lengths[static_cast<std::size_t>(i)] = 9;
    for (int i = 256; i < 280; ++i) lengths[static_cast<std::size_t>(i)] = 7;
    for (int i = 280; i < 288; ++i) lengths[static_cast<std::size_t>(i)] = 8;
    return build_huffman_table(lengths);
}

[[nodiscard]] HuffmanTable fixed_distance_table() { return build_huffman_table(std::vector<int>(30, 5)); }

constexpr std::array<int, 29> kLengthBase = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                              31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<int, 29> kLengthExtra = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<int, 30> kDistBase = {1,    2,    3,    4,    5,    7,    9,    13,   17,    25,   33,   49,    65,    97, 129,
                                            193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<int, 30> kDistExtra = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

void inflate_block(BitReader& reader, const HuffmanTable& lit_table, const HuffmanTable& dist_table, std::vector<unsigned char>& out) {
    while (true) {
        const int symbol = decode_symbol(reader, lit_table);
        if (symbol < 256) {
            out.push_back(static_cast<unsigned char>(symbol));
        } else if (symbol == 256) {
            return; // end of block
        } else {
            const std::size_t length_index = static_cast<std::size_t>(symbol - 257);
            if (length_index >= kLengthBase.size()) throw std::runtime_error("zlib_decompress: invalid length symbol");
            const int length = kLengthBase[length_index] + static_cast<int>(reader.get_bits(kLengthExtra[length_index]));

            const int dist_symbol = decode_symbol(reader, dist_table);
            if (dist_symbol < 0 || static_cast<std::size_t>(dist_symbol) >= kDistBase.size())
                throw std::runtime_error("zlib_decompress: invalid distance symbol");
            const int distance = kDistBase[static_cast<std::size_t>(dist_symbol)] + static_cast<int>(reader.get_bits(kDistExtra[static_cast<std::size_t>(dist_symbol)]));

            if (static_cast<std::size_t>(distance) > out.size()) {
                throw std::runtime_error("zlib_decompress: back-reference points before the start of the output");
            }
            std::size_t copy_from = out.size() - static_cast<std::size_t>(distance);
            for (int i = 0; i < length; ++i) {
                out.push_back(out[copy_from++]); // byte-by-byte: back-references may overlap the source
            }
        }
    }
}

void inflate_stored_block(BitReader& reader, std::vector<unsigned char>& out) {
    reader.align_to_byte();
    const unsigned int len_lo = reader.get_byte();
    const unsigned int len_hi = reader.get_byte();
    const unsigned int nlen_lo = reader.get_byte();
    const unsigned int nlen_hi = reader.get_byte();
    const unsigned int len = len_lo | (len_hi << 8);
    const unsigned int nlen = nlen_lo | (nlen_hi << 8);
    if ((len ^ 0xFFFFu) != nlen) {
        throw std::runtime_error("zlib_decompress: corrupt stored block (LEN/NLEN mismatch)");
    }
    for (unsigned int i = 0; i < len; ++i) out.push_back(reader.get_byte());
}

constexpr std::array<int, 19> kCodeLengthOrder = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

void inflate_dynamic_block(BitReader& reader, std::vector<unsigned char>& out) {
    const int hlit = static_cast<int>(reader.get_bits(5)) + 257;
    const int hdist = static_cast<int>(reader.get_bits(5)) + 1;
    const int hclen = static_cast<int>(reader.get_bits(4)) + 4;

    std::vector<int> code_length_lengths(19, 0);
    for (int i = 0; i < hclen; ++i) code_length_lengths[static_cast<std::size_t>(kCodeLengthOrder[static_cast<std::size_t>(i)])] =
        static_cast<int>(reader.get_bits(3));
    const HuffmanTable code_length_table = build_huffman_table(code_length_lengths);

    std::vector<int> lengths;
    lengths.reserve(static_cast<std::size_t>(hlit + hdist));
    while (static_cast<int>(lengths.size()) < hlit + hdist) {
        const int symbol = decode_symbol(reader, code_length_table);
        if (symbol < 16) {
            lengths.push_back(symbol);
        } else if (symbol == 16) {
            if (lengths.empty()) throw std::runtime_error("zlib_decompress: repeat code with no previous length");
            const int repeat = 3 + static_cast<int>(reader.get_bits(2));
            const int previous = lengths.back();
            for (int i = 0; i < repeat; ++i) lengths.push_back(previous);
        } else if (symbol == 17) {
            const int repeat = 3 + static_cast<int>(reader.get_bits(3));
            for (int i = 0; i < repeat; ++i) lengths.push_back(0);
        } else if (symbol == 18) {
            const int repeat = 11 + static_cast<int>(reader.get_bits(7));
            for (int i = 0; i < repeat; ++i) lengths.push_back(0);
        } else {
            throw std::runtime_error("zlib_decompress: invalid code-length symbol");
        }
    }
    if (static_cast<int>(lengths.size()) != hlit + hdist) {
        throw std::runtime_error("zlib_decompress: code-length sequence overran the declared table sizes");
    }

    const std::vector<int> lit_lengths(lengths.begin(), lengths.begin() + hlit);
    const std::vector<int> dist_lengths(lengths.begin() + hlit, lengths.end());
    const HuffmanTable lit_table = build_huffman_table(lit_lengths);
    const HuffmanTable dist_table = build_huffman_table(dist_lengths);
    inflate_block(reader, lit_table, dist_table, out);
}

} // namespace

std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, std::size_t expected_size) {
    if (compressed.size() < 6) { // 2-byte header + at least an empty deflate stream + 4-byte trailer
        throw std::runtime_error("zlib_decompress: input too short to be a valid zlib stream");
    }
    const unsigned char cmf = compressed[0];
    const unsigned char flg = compressed[1];
    if ((cmf & 0x0Fu) != 8) {
        throw std::runtime_error("zlib_decompress: unsupported compression method (only DEFLATE/method 8 is supported)");
    }
    if (((static_cast<unsigned int>(cmf) << 8) | flg) % 31 != 0) {
        throw std::runtime_error("zlib_decompress: invalid zlib header checksum");
    }
    if (flg & 0x20u) {
        throw std::runtime_error("zlib_decompress: preset dictionaries are not supported");
    }

    std::vector<unsigned char> out;
    out.reserve(expected_size);
    BitReader reader(compressed.data() + 2, compressed.size() - 2 - 4);

    bool final_block = false;
    while (!final_block) {
        final_block = reader.get_bit() != 0;
        const std::uint32_t block_type = reader.get_bits(2);
        switch (block_type) {
        case 0:
            inflate_stored_block(reader, out);
            break;
        case 1: {
            static const HuffmanTable lit_table = fixed_literal_length_table();
            static const HuffmanTable dist_table = fixed_distance_table();
            inflate_block(reader, lit_table, dist_table, out);
            break;
        }
        case 2:
            inflate_dynamic_block(reader, out);
            break;
        default:
            throw std::runtime_error("zlib_decompress: reserved (invalid) block type");
        }
    }

    if (out.size() != expected_size) {
        throw std::runtime_error("zlib_decompress: decompressed size does not match the expected size");
    }

    const std::size_t trailer_offset = compressed.size() - 4;
    const std::uint32_t expected_adler = (static_cast<std::uint32_t>(compressed[trailer_offset]) << 24) |
                                          (static_cast<std::uint32_t>(compressed[trailer_offset + 1]) << 16) |
                                          (static_cast<std::uint32_t>(compressed[trailer_offset + 2]) << 8) |
                                          static_cast<std::uint32_t>(compressed[trailer_offset + 3]);
    const std::uint32_t actual_adler = out.empty() ? 1u : adler32(out.data(), out.size());
    if (expected_adler != actual_adler) {
        throw std::runtime_error("zlib_decompress: Adler-32 checksum mismatch (corrupt data)");
    }

    return out;
}

std::vector<unsigned char> zlib_compress(const std::vector<unsigned char>& data) {
    std::vector<unsigned char> out;
    out.reserve(data.size() + data.size() / 65535 * 5 + 11);

    out.push_back(0x78); // CMF: CINFO=7 (32K window), method=8 (DEFLATE)
    out.push_back(0x01); // FLG: FCHECK made (CMF*256+FLG) a multiple of 31, FDICT=0, FLEVEL=0

    constexpr std::size_t kMaxStoredBlock = 65535;
    std::size_t offset = 0;
    do {
        const std::size_t remaining = data.size() - offset;
        const std::size_t block_size = remaining < kMaxStoredBlock ? remaining : kMaxStoredBlock;
        const bool is_final = (offset + block_size) >= data.size();

        // 3-bit block header (BFINAL, BTYPE=00), then pad to a byte boundary -- both fit in a
        // single byte since a fresh byte boundary is guaranteed at the start of every stored
        // block (this function only ever emits stored blocks back-to-back).
        out.push_back(is_final ? 0x01 : 0x00);

        const auto len = static_cast<std::uint16_t>(block_size);
        const std::uint16_t nlen = static_cast<std::uint16_t>(~len);
        out.push_back(static_cast<unsigned char>(len & 0xFFu));
        out.push_back(static_cast<unsigned char>((len >> 8) & 0xFFu));
        out.push_back(static_cast<unsigned char>(nlen & 0xFFu));
        out.push_back(static_cast<unsigned char>((nlen >> 8) & 0xFFu));
        out.insert(out.end(), data.begin() + static_cast<std::ptrdiff_t>(offset), data.begin() + static_cast<std::ptrdiff_t>(offset + block_size));

        offset += block_size;
    } while (offset < data.size());

    const std::uint32_t checksum = data.empty() ? 1u : adler32(data.data(), data.size());
    out.push_back(static_cast<unsigned char>((checksum >> 24) & 0xFFu));
    out.push_back(static_cast<unsigned char>((checksum >> 16) & 0xFFu));
    out.push_back(static_cast<unsigned char>((checksum >> 8) & 0xFFu));
    out.push_back(static_cast<unsigned char>(checksum & 0xFFu));

    return out;
}

} // namespace datamunge::image::detail
