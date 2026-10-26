#include <datamunge/image/detail/zlib_codec.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace datamunge::image::detail {

namespace {

constexpr int kMaxBits = 15;

constexpr std::array<int, 29> kLengthBase = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                              31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<int, 29> kLengthExtra = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<int, 30> kDistBase = {1,    2,    3,    4,    5,    7,    9,    13,   17,    25,   33,   49,    65,    97, 129,
                                            193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<int, 30> kDistExtra = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
constexpr std::array<int, 19> kCodeLengthOrder = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

// Adler-32 with deferred modulo (same result as adler32(), several times faster).
[[nodiscard]] std::uint32_t fast_adler32(const unsigned char* data, std::size_t length) {
    constexpr std::uint32_t kMod = 65521;
    constexpr std::size_t kMaxRun = 5552; // largest n keeping b below 2^32 before reducing
    std::uint32_t a = 1, b = 0;
    while (length > 0) {
        const std::size_t run = std::min(length, kMaxRun);
        for (std::size_t i = 0; i < run; ++i) {
            a += data[i];
            b += a;
        }
        a %= kMod;
        b %= kMod;
        data += run;
        length -= run;
    }
    return (b << 16) | a;
}

// =====================================================================================
// Decoder
// =====================================================================================

// Bit reader over a raw DEFLATE stream. Fields are packed LSB-first; Huffman codes are matched
// MSB-first (decode_symbol() reads them one bit at a time).
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

    /// @brief Drops the buffered partial byte (stored blocks start on a byte boundary).
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

// Canonical Huffman decode table (RFC 1951 3.2.2): counts per length + symbols ordered by
// (length, symbol), enough to decode without materializing the codes.
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

[[nodiscard]] std::vector<int> fixed_literal_lengths() {
    std::vector<int> lengths(288);
    for (int i = 0; i < 144; ++i) lengths[static_cast<std::size_t>(i)] = 8;
    for (int i = 144; i < 256; ++i) lengths[static_cast<std::size_t>(i)] = 9;
    for (int i = 256; i < 280; ++i) lengths[static_cast<std::size_t>(i)] = 7;
    for (int i = 280; i < 288; ++i) lengths[static_cast<std::size_t>(i)] = 8;
    return lengths;
}

// Output sink enforcing a size limit; `overflow_message` distinguishes sized/unsized callers.
struct Output {
    std::vector<unsigned char> bytes;
    std::size_t limit;
    const char* overflow_message;

    void reserve_room(std::size_t n) const {
        if (n > limit - bytes.size()) throw std::runtime_error(overflow_message);
    }
};

void inflate_block(BitReader& reader, const HuffmanTable& lit_table, const HuffmanTable& dist_table, Output& out) {
    std::vector<unsigned char>& bytes = out.bytes;
    while (true) {
        const int symbol = decode_symbol(reader, lit_table);
        if (symbol < 256) {
            out.reserve_room(1);
            bytes.push_back(static_cast<unsigned char>(symbol));
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

            if (static_cast<std::size_t>(distance) > bytes.size()) {
                throw std::runtime_error("zlib_decompress: back-reference points before the start of the output");
            }
            out.reserve_room(static_cast<std::size_t>(length));
            std::size_t copy_from = bytes.size() - static_cast<std::size_t>(distance);
            for (int i = 0; i < length; ++i) {
                bytes.push_back(bytes[copy_from++]); // byte-by-byte: the source may overlap the destination
            }
        }
    }
}

void inflate_stored_block(BitReader& reader, Output& out) {
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
    out.reserve_room(len);
    for (unsigned int i = 0; i < len; ++i) out.bytes.push_back(reader.get_byte());
}

void inflate_dynamic_block(BitReader& reader, Output& out) {
    const int hlit = static_cast<int>(reader.get_bits(5)) + 257;
    const int hdist = static_cast<int>(reader.get_bits(5)) + 1;
    const int hclen = static_cast<int>(reader.get_bits(4)) + 4;

    std::vector<int> code_length_lengths(19, 0);
    for (int i = 0; i < hclen; ++i) code_length_lengths[static_cast<std::size_t>(kCodeLengthOrder[static_cast<std::size_t>(i)])] =
        static_cast<int>(reader.get_bits(3));
    const HuffmanTable code_length_table = build_huffman_table(code_length_lengths);

    std::vector<int> lengths;
    lengths.reserve(static_cast<std::size_t>(hlit + hdist) + 138);
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

void inflate(const unsigned char* data, std::size_t size, Output& out) {
    BitReader reader(data, size);
    bool final_block = false;
    while (!final_block) {
        final_block = reader.get_bit() != 0;
        const std::uint32_t block_type = reader.get_bits(2);
        switch (block_type) {
        case 0:
            inflate_stored_block(reader, out);
            break;
        case 1: {
            static const HuffmanTable lit_table = build_huffman_table(fixed_literal_lengths());
            static const HuffmanTable dist_table = build_huffman_table(std::vector<int>(30, 5));
            inflate_block(reader, lit_table, dist_table, out);
            break;
        }
        case 2:
            inflate_dynamic_block(reader, out);
            break;
        default:
            throw std::runtime_error("deflate_decompress: reserved (invalid) block type");
        }
    }
}

constexpr const char* kSizeMismatch = "deflate_decompress: decompressed size does not match the expected size";
constexpr const char* kTooLarge = "deflate_decompress: decompressed output exceeds max_output";

[[nodiscard]] std::vector<unsigned char> zlib_unwrap(const std::vector<unsigned char>& compressed, Output& out) {
    if (compressed.size() < 6) { // 2-byte header + at least an empty deflate stream + 4-byte trailer
        throw std::runtime_error("zlib_decompress: input too short to be a valid zlib stream");
    }
    const unsigned char cmf = compressed[0];
    const unsigned char flg = compressed[1];
    if ((cmf & 0x0Fu) != 8 || (cmf >> 4) > 7) {
        throw std::runtime_error("zlib_decompress: unsupported compression method (only DEFLATE/method 8 is supported)");
    }
    if (((static_cast<unsigned int>(cmf) << 8) | flg) % 31 != 0) {
        throw std::runtime_error("zlib_decompress: invalid zlib header checksum");
    }
    if (flg & 0x20u) {
        throw std::runtime_error("zlib_decompress: preset dictionaries are not supported");
    }

    inflate(compressed.data() + 2, compressed.size() - 2 - 4, out);

    const std::size_t t = compressed.size() - 4;
    const std::uint32_t expected_adler = (static_cast<std::uint32_t>(compressed[t]) << 24) | (static_cast<std::uint32_t>(compressed[t + 1]) << 16) |
                                         (static_cast<std::uint32_t>(compressed[t + 2]) << 8) | static_cast<std::uint32_t>(compressed[t + 3]);
    const std::uint32_t actual_adler = fast_adler32(out.bytes.data(), out.bytes.size());
    if (expected_adler != actual_adler) {
        throw std::runtime_error("zlib_decompress: Adler-32 checksum mismatch (corrupt data)");
    }
    return std::move(out.bytes);
}

// =====================================================================================
// Encoder
// =====================================================================================

// LSB-first bit writer. Huffman codes are stored pre-reversed so they can be written as-is.
class BitWriter {
  public:
    explicit BitWriter(std::vector<unsigned char>& out) : out_(out) {}

    void put(std::uint32_t value, int n) {
        buffer_ |= static_cast<std::uint64_t>(value) << count_;
        count_ += n;
        while (count_ >= 8) {
            out_.push_back(static_cast<unsigned char>(buffer_ & 0xFFu));
            buffer_ >>= 8;
            count_ -= 8;
        }
    }

    void align() {
        if (count_ > 0) put(0, 8 - count_);
    }

    /// Appends raw bytes; only valid on a byte boundary (after align()).
    void put_bytes(const unsigned char* data, std::size_t n) { out_.insert(out_.end(), data, data + n); }

    [[nodiscard]] int pending_bits() const { return count_; }

  private:
    std::vector<unsigned char>& out_;
    std::uint64_t buffer_{0};
    int count_{0};
};

constexpr int kNumLitLen = 286;
constexpr int kNumDist = 30;
constexpr std::size_t kWindowSize = 32768;
constexpr std::size_t kWindowMask = kWindowSize - 1;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr std::size_t kTooFar = 4096; // length-3 matches farther than this aren't worth it
constexpr std::size_t kMaxBlockSymbols = 16383;
constexpr std::size_t kMaxStoredBlock = 65535;

struct CodeTables {
    std::array<std::uint8_t, kMaxMatch + 1> length_code{};
    std::array<std::uint8_t, 512> dist_code{}; // dist-1 < 256 -> [dist-1]; else 256 + ((dist-1) >> 7)
    CodeTables() {
        for (std::size_t c = 0; c < kLengthBase.size(); ++c)
            for (int len = kLengthBase[c]; len < kLengthBase[c] + (1 << kLengthExtra[c]) && len <= kMaxMatch; ++len)
                length_code[static_cast<std::size_t>(len)] = static_cast<std::uint8_t>(c);
        for (std::size_t c = 0; c < kDistBase.size(); ++c) {
            for (int d = kDistBase[c]; d < kDistBase[c] + (1 << kDistExtra[c]); ++d) {
                const int v = d - 1;
                if (v < 256) dist_code[static_cast<std::size_t>(v)] = static_cast<std::uint8_t>(c);
                else dist_code[256 + static_cast<std::size_t>(v >> 7)] = static_cast<std::uint8_t>(c);
            }
        }
    }
    [[nodiscard]] int dist(std::size_t d) const {
        const std::size_t v = d - 1;
        return v < 256 ? dist_code[v] : dist_code[256 + (v >> 7)];
    }
};

const CodeTables& code_tables() {
    static const CodeTables tables;
    return tables;
}

// Length-limited Huffman code lengths via package-merge (optimal for the limit). Symbols with
// zero frequency get length 0; the caller guarantees at least two nonzero frequencies.
[[nodiscard]] std::vector<std::uint8_t> limited_code_lengths(const std::vector<std::uint32_t>& freqs, int max_bits) {
    struct Item {
        std::uint64_t weight;
        int symbol; // >= 0: leaf; -1: package of children (child, child + 1) one level down
        int child;
    };
    std::vector<Item> leaves;
    for (std::size_t s = 0; s < freqs.size(); ++s)
        if (freqs[s] > 0) leaves.push_back({freqs[s], static_cast<int>(s), 0});
    std::stable_sort(leaves.begin(), leaves.end(), [](const Item& a, const Item& b) { return a.weight < b.weight; });

    std::vector<std::uint8_t> lengths(freqs.size(), 0);
    const std::size_t n = leaves.size();
    if (n == 0) return lengths;
    if (n == 1) {
        lengths[static_cast<std::size_t>(leaves[0].symbol)] = 1;
        return lengths;
    }

    std::vector<std::vector<Item>> levels(static_cast<std::size_t>(max_bits));
    levels[0] = leaves;
    for (std::size_t k = 1; k < levels.size(); ++k) {
        const std::vector<Item>& prev = levels[k - 1];
        std::vector<Item>& cur = levels[k];
        cur.reserve(n + prev.size() / 2);
        std::size_t li = 0, pi = 0;
        while (li < n || pi + 1 < prev.size()) {
            const bool take_leaf =
                pi + 1 >= prev.size() || (li < n && leaves[li].weight <= prev[pi].weight + prev[pi + 1].weight);
            if (take_leaf) {
                cur.push_back(leaves[li++]);
            } else {
                cur.push_back({prev[pi].weight + prev[pi + 1].weight, -1, static_cast<int>(pi)});
                pi += 2;
            }
        }
    }

    // Each appearance of a leaf among the first 2n-2 top-level items (recursively) adds one bit.
    std::vector<std::pair<std::size_t, std::size_t>> stack; // (level, count of leading items)
    stack.emplace_back(levels.size() - 1, 2 * n - 2);
    while (!stack.empty()) {
        const auto [k, count] = stack.back();
        stack.pop_back();
        std::size_t packages = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const Item& it = levels[k][i];
            if (it.symbol >= 0) ++lengths[static_cast<std::size_t>(it.symbol)];
            else ++packages;
        }
        // Packages in the first `count` items are exactly the first `packages` packages, which
        // consume the first 2*packages items of the level below.
        if (packages > 0) stack.emplace_back(k - 1, 2 * packages);
    }
    return lengths;
}

// Canonical codes (RFC 1951 3.2.2), returned bit-reversed for LSB-first emission.
[[nodiscard]] std::vector<std::uint16_t> canonical_codes(const std::vector<std::uint8_t>& lengths) {
    std::array<std::uint16_t, kMaxBits + 2> bl_count{}, next_code{};
    for (const auto len : lengths)
        if (len > 0) ++bl_count[len];
    std::uint16_t code = 0;
    for (int bits = 1; bits <= kMaxBits; ++bits) {
        code = static_cast<std::uint16_t>((code + bl_count[static_cast<std::size_t>(bits - 1)]) << 1);
        next_code[static_cast<std::size_t>(bits)] = code;
    }
    std::vector<std::uint16_t> codes(lengths.size(), 0);
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        const int len = lengths[s];
        if (len == 0) continue;
        std::uint16_t c = next_code[static_cast<std::size_t>(len)]++;
        std::uint16_t rev = 0;
        for (int i = 0; i < len; ++i) {
            rev = static_cast<std::uint16_t>((static_cast<unsigned>(rev) << 1) | (c & 1u));
            c = static_cast<std::uint16_t>(c >> 1);
        }
        codes[s] = rev;
    }
    return codes;
}

// Huffman trees need >= 2 used symbols so every decoder (zlib included) sees a complete code.
void ensure_two_symbols(std::vector<std::uint32_t>& freqs) {
    int used = 0;
    for (const auto f : freqs) used += f > 0 ? 1 : 0;
    for (std::size_t s = 0; s < freqs.size() && used < 2; ++s) {
        if (freqs[s] == 0) {
            freqs[s] = 1;
            ++used;
        }
    }
}

struct Symbol {
    std::uint16_t litlen; // literal byte, or match length
    std::uint16_t dist;   // 0 for a literal
};

struct RleItem {
    std::uint8_t symbol;
    std::uint8_t extra;
};

[[nodiscard]] std::vector<RleItem> rle_code_lengths(const std::vector<std::uint8_t>& lengths) {
    std::vector<RleItem> items;
    std::size_t i = 0;
    while (i < lengths.size()) {
        const std::uint8_t v = lengths[i];
        std::size_t run = 1;
        while (i + run < lengths.size() && lengths[i + run] == v) ++run;
        i += run;
        if (v == 0) {
            while (run >= 11) {
                const std::size_t r = std::min<std::size_t>(run, 138);
                items.push_back({18, static_cast<std::uint8_t>(r - 11)});
                run -= r;
            }
            if (run >= 3) {
                items.push_back({17, static_cast<std::uint8_t>(run - 3)});
                run = 0;
            }
        } else {
            items.push_back({v, 0});
            --run;
            while (run >= 3) {
                const std::size_t r = std::min<std::size_t>(run, 6);
                items.push_back({16, static_cast<std::uint8_t>(r - 3)});
                run -= r;
            }
        }
        for (; run > 0; --run) items.push_back({v, 0});
    }
    return items;
}

struct FixedCodes {
    std::vector<std::uint8_t> lit_lengths, dist_lengths;
    std::vector<std::uint16_t> lit_codes, dist_codes;
    FixedCodes() {
        for (const int len : fixed_literal_lengths()) lit_lengths.push_back(static_cast<std::uint8_t>(len));
        dist_lengths.assign(30, 5);
        lit_codes = canonical_codes(lit_lengths);
        dist_codes = canonical_codes(dist_lengths);
    }
};

const FixedCodes& fixed_codes() {
    static const FixedCodes codes;
    return codes;
}

// Buffers LZ77 symbols and emits each block as the cheapest of stored / fixed / dynamic.
class BlockEncoder {
  public:
    BlockEncoder(const unsigned char* data, BitWriter& writer) : data_(data), writer_(writer) { symbols_.reserve(kMaxBlockSymbols); }

    void literal(unsigned char c) {
        symbols_.push_back({c, 0});
        ++raw_end_;
        if (symbols_.size() >= kMaxBlockSymbols) flush(false);
    }

    void match(int length, std::size_t distance) {
        symbols_.push_back({static_cast<std::uint16_t>(length), static_cast<std::uint16_t>(distance)});
        raw_end_ += static_cast<std::size_t>(length);
        if (symbols_.size() >= kMaxBlockSymbols) flush(false);
    }

    void flush(bool final_block) {
        const CodeTables& tables = code_tables();
        std::vector<std::uint32_t> lit_freq(kNumLitLen, 0), dist_freq(kNumDist, 0);
        lit_freq[256] = 1;
        std::uint64_t extra_bits = 0;
        for (const Symbol& s : symbols_) {
            if (s.dist == 0) {
                ++lit_freq[s.litlen];
            } else {
                const int lc = tables.length_code[s.litlen];
                const int dc = tables.dist(s.dist);
                ++lit_freq[static_cast<std::size_t>(257 + lc)];
                ++dist_freq[static_cast<std::size_t>(dc)];
                extra_bits += static_cast<std::uint64_t>(kLengthExtra[static_cast<std::size_t>(lc)] + kDistExtra[static_cast<std::size_t>(dc)]);
            }
        }

        // Fixed-Huffman cost.
        const FixedCodes& fixed = fixed_codes();
        std::uint64_t fixed_cost = 3 + extra_bits;
        for (int s = 0; s < kNumLitLen; ++s) fixed_cost += static_cast<std::uint64_t>(lit_freq[static_cast<std::size_t>(s)]) * fixed.lit_lengths[static_cast<std::size_t>(s)];
        for (int s = 0; s < kNumDist; ++s) fixed_cost += static_cast<std::uint64_t>(dist_freq[static_cast<std::size_t>(s)]) * 5u;

        // Dynamic-Huffman trees, header and cost.
        std::vector<std::uint32_t> lit_tree_freq = lit_freq, dist_tree_freq = dist_freq;
        ensure_two_symbols(lit_tree_freq);
        ensure_two_symbols(dist_tree_freq);
        const std::vector<std::uint8_t> lit_len = limited_code_lengths(lit_tree_freq, 15);
        const std::vector<std::uint8_t> dist_len = limited_code_lengths(dist_tree_freq, 15);
        int hlit = kNumLitLen;
        while (hlit > 257 && lit_len[static_cast<std::size_t>(hlit - 1)] == 0) --hlit;
        int hdist = kNumDist;
        while (hdist > 1 && dist_len[static_cast<std::size_t>(hdist - 1)] == 0) --hdist;
        std::vector<std::uint8_t> all_lengths(lit_len.begin(), lit_len.begin() + hlit);
        all_lengths.insert(all_lengths.end(), dist_len.begin(), dist_len.begin() + hdist);
        const std::vector<RleItem> rle = rle_code_lengths(all_lengths);
        std::vector<std::uint32_t> cl_freq(19, 0);
        for (const RleItem& it : rle) ++cl_freq[it.symbol];
        ensure_two_symbols(cl_freq);
        const std::vector<std::uint8_t> cl_len = limited_code_lengths(cl_freq, 7);
        int hclen = 19;
        while (hclen > 4 && cl_len[static_cast<std::size_t>(kCodeLengthOrder[static_cast<std::size_t>(hclen - 1)])] == 0) --hclen;

        std::uint64_t dynamic_cost = 3 + 14 + 3 * static_cast<std::uint64_t>(hclen) + extra_bits;
        for (const RleItem& it : rle)
            dynamic_cost += cl_len[it.symbol] + (it.symbol == 16 ? 2u : it.symbol == 17 ? 3u : it.symbol == 18 ? 7u : 0u);
        for (int s = 0; s < kNumLitLen; ++s) dynamic_cost += static_cast<std::uint64_t>(lit_freq[static_cast<std::size_t>(s)]) * lit_len[static_cast<std::size_t>(s)];
        for (int s = 0; s < kNumDist; ++s) dynamic_cost += static_cast<std::uint64_t>(dist_freq[static_cast<std::size_t>(s)]) * dist_len[static_cast<std::size_t>(s)];

        // Stored cost: every chunk is header + pad to byte + LEN/NLEN + raw bytes.
        const std::size_t raw_size = raw_end_ - raw_start_;
        const std::size_t chunks = raw_size == 0 ? 1 : (raw_size + kMaxStoredBlock - 1) / kMaxStoredBlock;
        const int first_pad = (8 - (writer_.pending_bits() + 3) % 8) % 8;
        const std::uint64_t stored_cost =
            static_cast<std::uint64_t>(first_pad) + 3 + 8 * (static_cast<std::uint64_t>(chunks) - 1) + 32 * static_cast<std::uint64_t>(chunks) + 8 * static_cast<std::uint64_t>(raw_size);

        if (stored_cost <= fixed_cost && stored_cost <= dynamic_cost) {
            emit_stored(data_ + raw_start_, raw_size, final_block, writer_);
        } else if (fixed_cost <= dynamic_cost) {
            writer_.put(final_block ? 1u : 0u, 1);
            writer_.put(1, 2);
            emit_symbols(fixed.lit_lengths, fixed.lit_codes, fixed.dist_lengths, fixed.dist_codes);
        } else {
            writer_.put(final_block ? 1u : 0u, 1);
            writer_.put(2, 2);
            writer_.put(static_cast<std::uint32_t>(hlit - 257), 5);
            writer_.put(static_cast<std::uint32_t>(hdist - 1), 5);
            writer_.put(static_cast<std::uint32_t>(hclen - 4), 4);
            for (int i = 0; i < hclen; ++i) writer_.put(cl_len[static_cast<std::size_t>(kCodeLengthOrder[static_cast<std::size_t>(i)])], 3);
            const std::vector<std::uint16_t> cl_codes = canonical_codes(cl_len);
            for (const RleItem& it : rle) {
                writer_.put(cl_codes[it.symbol], cl_len[it.symbol]);
                if (it.symbol == 16) writer_.put(it.extra, 2);
                else if (it.symbol == 17) writer_.put(it.extra, 3);
                else if (it.symbol == 18) writer_.put(it.extra, 7);
            }
            emit_symbols(lit_len, canonical_codes(lit_len), dist_len, canonical_codes(dist_len));
        }

        symbols_.clear();
        raw_start_ = raw_end_;
    }

    static void emit_stored(const unsigned char* data, std::size_t size, bool final_block, BitWriter& writer) {
        std::size_t offset = 0;
        do {
            const std::size_t n = std::min(size - offset, kMaxStoredBlock);
            const bool last = offset + n >= size;
            writer.put(final_block && last ? 1u : 0u, 1);
            writer.put(0, 2);
            writer.align();
            writer.put(static_cast<std::uint32_t>(n), 16);
            writer.put(static_cast<std::uint32_t>(~n & 0xFFFFu), 16);
            if (n > 0) writer.put_bytes(data + offset, n);
            offset += n;
        } while (offset < size);
    }

  private:
    void emit_symbols(const std::vector<std::uint8_t>& lit_len, const std::vector<std::uint16_t>& lit_codes,
                      const std::vector<std::uint8_t>& dist_len, const std::vector<std::uint16_t>& dist_codes) {
        const CodeTables& tables = code_tables();
        for (const Symbol& s : symbols_) {
            if (s.dist == 0) {
                writer_.put(lit_codes[s.litlen], lit_len[s.litlen]);
                continue;
            }
            const std::size_t lc = tables.length_code[s.litlen];
            writer_.put(lit_codes[257 + lc], lit_len[257 + lc]);
            if (kLengthExtra[lc] > 0) writer_.put(static_cast<std::uint32_t>(s.litlen - kLengthBase[lc]), kLengthExtra[lc]);
            const auto dc = static_cast<std::size_t>(tables.dist(s.dist));
            writer_.put(dist_codes[dc], dist_len[dc]);
            if (kDistExtra[dc] > 0) writer_.put(static_cast<std::uint32_t>(s.dist - kDistBase[dc]), kDistExtra[dc]);
        }
        writer_.put(lit_codes[256], lit_len[256]);
    }

    const unsigned char* data_;
    BitWriter& writer_;
    std::vector<Symbol> symbols_;
    std::size_t raw_start_{0};
    std::size_t raw_end_{0};
};

// zlib's per-level tuning: good_length, max_lazy, nice_length, max_chain; levels 1-3 are greedy.
struct LevelConfig {
    int good;
    int lazy;
    int nice;
    int chain;
};
constexpr std::array<LevelConfig, 10> kLevels = {{{0, 0, 0, 0},
                                                  {4, 4, 8, 4},
                                                  {4, 5, 16, 8},
                                                  {4, 6, 32, 32},
                                                  {4, 4, 16, 16},
                                                  {8, 16, 32, 32},
                                                  {8, 16, 128, 128},
                                                  {8, 32, 128, 256},
                                                  {32, 128, 258, 1024},
                                                  {32, 258, 258, 4096}}};

// Hash-chain match finder over the whole input. Positions are searched before being inserted,
// so every chain entry precedes the current position and distances stay <= 32 KiB.
class MatchFinder {
  public:
    MatchFinder(const unsigned char* data, std::size_t size)
        : data_(data), size_(size), head_(std::size_t{1} << kHashBits, 0), prev_(kWindowSize, 0) {}

    void insert(std::size_t pos) {
        if (pos + 2 >= size_) return;
        const std::uint32_t h = hash(pos);
        prev_[pos & kWindowMask] = head_[h];
        head_[h] = tag(pos);
    }

    /// Longest match at @p pos strictly longer than @p best_so_far (>= 2); 0 if none.
    int longest(std::size_t pos, int best_so_far, int chain, const LevelConfig& cfg, std::size_t& distance) const {
        if (pos + 2 >= size_) return 0;
        const int max_len = static_cast<int>(std::min<std::size_t>(kMaxMatch, size_ - pos));
        int best = std::max(best_so_far, kMinMatch - 1);
        if (best >= max_len) return 0;
        if (best_so_far >= cfg.good) chain >>= 2;
        const int nice = std::min(cfg.nice, max_len);
        const unsigned char* q = data_ + pos;
        int found = 0;
        std::uint32_t t = head_[hash(pos)];
        while (t != 0 && chain-- > 0) {
            const std::uint32_t dist = tag(pos) - t; // mod 2^32; candidates are byte-verified anyway
            if (dist == 0 || dist > kWindowSize || dist > pos) break;
            const std::size_t cand = pos - dist;
            const unsigned char* p = data_ + cand;
            if (p[best] == q[best] && p[best - 1] == q[best - 1] && p[0] == q[0] && p[1] == q[1]) {
                int len = 2;
                while (len + 8 <= max_len && load64(p + len) == load64(q + len)) len += 8;
                while (len < max_len && p[len] == q[len]) ++len;
                if (len > best) {
                    best = len;
                    found = len;
                    distance = pos - cand;
                    if (len >= nice) break;
                }
            }
            t = prev_[cand & kWindowMask];
        }
        return found;
    }

  private:
    static constexpr int kHashBits = 15;

    // Chain entries are (pos + 1) mod 2^32; 0 means empty.
    [[nodiscard]] static std::uint32_t tag(std::size_t pos) { return static_cast<std::uint32_t>(pos + 1); }

    [[nodiscard]] static std::uint64_t load64(const unsigned char* p) {
        std::uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }

    [[nodiscard]] std::uint32_t hash(std::size_t pos) const {
        const std::uint32_t v = static_cast<std::uint32_t>(data_[pos]) | (static_cast<std::uint32_t>(data_[pos + 1]) << 8) |
                                (static_cast<std::uint32_t>(data_[pos + 2]) << 16);
        return (v * 2654435761u) >> (32 - kHashBits);
    }

    const unsigned char* data_;
    std::size_t size_;
    std::vector<std::uint32_t> head_;
    std::vector<std::uint32_t> prev_;
};

void deflate_into(const std::vector<unsigned char>& data, int level, std::vector<unsigned char>& out) {
    BitWriter writer(out);
    if (level == 0) {
        BlockEncoder::emit_stored(data.data(), data.size(), true, writer);
        writer.align();
        return;
    }

    const LevelConfig& cfg = kLevels[static_cast<std::size_t>(level)];
    const unsigned char* d = data.data();
    const std::size_t n = data.size();
    MatchFinder finder(d, n);
    BlockEncoder encoder(d, writer);
    const auto too_far = [](int len, std::size_t dist) { return len == kMinMatch && dist > kTooFar; };

    if (level <= 3) { // greedy
        std::size_t i = 0;
        while (i < n) {
            std::size_t dist = 0;
            int len = finder.longest(i, 0, cfg.chain, cfg, dist);
            if (too_far(len, dist)) len = 0;
            if (len >= kMinMatch) {
                encoder.match(len, dist);
                if (len <= cfg.lazy) {
                    for (std::size_t j = i; j < i + static_cast<std::size_t>(len); ++j) finder.insert(j);
                } else {
                    finder.insert(i);
                }
                i += static_cast<std::size_t>(len);
            } else {
                finder.insert(i);
                encoder.literal(d[i]);
                ++i;
            }
        }
    } else { // lazy: defer a match by one byte if the next position has a longer one
        std::size_t i = 0;
        int prev_len = 0;
        std::size_t prev_dist = 0;
        bool pending = false; // d[i-1] not yet emitted
        while (i < n) {
            int cur_len = 0;
            std::size_t cur_dist = 0;
            if (prev_len < cfg.lazy) {
                cur_len = finder.longest(i, prev_len, cfg.chain, cfg, cur_dist);
                if (too_far(cur_len, cur_dist)) cur_len = 0;
            }
            finder.insert(i);
            if (prev_len >= kMinMatch && cur_len <= prev_len) {
                encoder.match(prev_len, prev_dist);
                const std::size_t end = i - 1 + static_cast<std::size_t>(prev_len);
                for (std::size_t j = i + 1; j < end; ++j) finder.insert(j);
                i = end;
                prev_len = 0;
                pending = false;
            } else {
                if (pending) encoder.literal(d[i - 1]);
                pending = true;
                prev_len = cur_len;
                prev_dist = cur_dist;
                ++i;
            }
        }
        if (pending) encoder.literal(d[n - 1]);
    }
    encoder.flush(true);
    writer.align();
}

[[nodiscard]] int checked_level(int level) {
    if (level == -1) return 6;
    if (level < 0 || level > 9) throw std::invalid_argument("zlib_compress: level must be in -1..9");
    return level;
}

} // namespace

std::vector<unsigned char> deflate_decompress(const unsigned char* data, std::size_t size, std::size_t expected_size) {
    Output out{{}, expected_size, kSizeMismatch};
    out.bytes.reserve(expected_size);
    inflate(data, size, out);
    if (out.bytes.size() != expected_size) throw std::runtime_error(kSizeMismatch);
    return std::move(out.bytes);
}

std::vector<unsigned char> deflate_decompress(const unsigned char* data, std::size_t size, InflateLimit limit) {
    Output out{{}, limit.max_output, kTooLarge};
    out.bytes.reserve(std::min(limit.max_output, size * 4));
    inflate(data, size, out);
    return std::move(out.bytes);
}

std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, std::size_t expected_size) {
    Output out{{}, expected_size, kSizeMismatch};
    out.bytes.reserve(expected_size);
    std::vector<unsigned char> result = zlib_unwrap(compressed, out);
    if (result.size() != expected_size) throw std::runtime_error(kSizeMismatch);
    return result;
}

std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, InflateLimit limit) {
    Output out{{}, limit.max_output, kTooLarge};
    out.bytes.reserve(std::min(limit.max_output, compressed.size() * 4));
    return zlib_unwrap(compressed, out);
}

std::vector<unsigned char> deflate_compress(const std::vector<unsigned char>& data, int level) {
    std::vector<unsigned char> out;
    out.reserve(data.size() / 2 + 64);
    deflate_into(data, checked_level(level), out);
    return out;
}

std::vector<unsigned char> zlib_compress(const std::vector<unsigned char>& data, int level) {
    level = checked_level(level);
    std::vector<unsigned char> out;
    out.reserve(level == 0 ? data.size() + data.size() / 65535 * 5 + 11 : data.size() / 2 + 64);
    // CMF 0x78 (deflate, 32K window); FLG carries zlib's FLEVEL hint and a valid FCHECK.
    out.push_back(0x78);
    out.push_back(level <= 1 ? 0x01 : level <= 5 ? 0x5E : level == 6 ? 0x9C : 0xDA);
    deflate_into(data, level, out);

    const std::uint32_t checksum = fast_adler32(data.data(), data.size());
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>((checksum >> shift) & 0xFFu));
    return out;
}

} // namespace datamunge::image::detail
