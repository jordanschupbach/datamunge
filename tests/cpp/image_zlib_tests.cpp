// Tests for the DEFLATE encoder (levels 0..9), unsized decompression, and interop with real
// zlib. Fixtures in tests/cpp/data/zlib/ were produced by Python's zlib module:
//   text.txt / scanlines.bin            -- inputs (source text; PNG-like RGB rows, filter 0)
//   text.l{0,1,6,9}.zz, scanlines.l6.zz -- zlib.compress(input, level)
// Encoder output is also checked against Python's zlib.decompress when python3 is on PATH.

#include <gtest/gtest.h>

#include <datamunge/image/detail/zlib_codec.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace datamunge::image::detail;
namespace fs = std::filesystem;

namespace {

using Bytes = std::vector<unsigned char>;

fs::path data_dir() { return fs::path(__FILE__).parent_path() / "data" / "zlib"; }

Bytes read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open fixture " + path.string());
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

Bytes random_bytes(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Bytes out(n);
    for (auto& b : out) b = static_cast<unsigned char>(rng() & 0xFFu);
    return out;
}

Bytes text_like(std::size_t n) {
    static const char* words[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ", "and ", "a ",
                                  "tensor ", "image ", "pixel ", "of ", "in ", "compression ", "\n", ", ", ". "};
    std::mt19937 rng(42);
    Bytes out;
    while (out.size() < n) {
        const std::string w = words[rng() % (sizeof(words) / sizeof(words[0]))];
        out.insert(out.end(), w.begin(), w.end());
    }
    out.resize(n);
    return out;
}

Bytes png_like_scanlines(int width, int height) {
    std::mt19937 rng(9);
    Bytes out;
    for (int y = 0; y < height; ++y) {
        out.push_back(0);
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) out.push_back(static_cast<unsigned char>((x + 2 * y + 50 * c + static_cast<int>(rng() % 5)) & 0xFF));
    }
    return out;
}

// Long repeated segment placed ~32 KiB back, so matches sit at/near the maximum distance.
Bytes long_distance(std::size_t gap) {
    Bytes seg = random_bytes(600, 3);
    Bytes out = seg;
    Bytes filler = random_bytes(gap - seg.size(), 4);
    out.insert(out.end(), filler.begin(), filler.end());
    out.insert(out.end(), seg.begin(), seg.end());
    return out;
}

std::vector<std::pair<std::string, Bytes>> corpus() {
    std::vector<std::pair<std::string, Bytes>> c;
    c.emplace_back("empty", Bytes{});
    c.emplace_back("one_byte", Bytes{0x5A});
    c.emplace_back("two_bytes", Bytes{1, 2});
    c.emplace_back("all_equal", Bytes(100000, 0x41));
    c.emplace_back("random", random_bytes(70000, 1));
    c.emplace_back("text", text_like(50000));
    c.emplace_back("scanlines", png_like_scanlines(200, 120));
    c.emplace_back("big_mixed", [] {
        Bytes b = text_like(90000);
        Bytes r = random_bytes(80000, 5);
        b.insert(b.end(), r.begin(), r.end());
        Bytes s = png_like_scanlines(300, 100);
        b.insert(b.end(), s.begin(), s.end());
        return b;
    }());
    c.emplace_back("dist_32768", long_distance(32768));
    c.emplace_back("dist_32767", long_distance(32767));
    c.emplace_back("dist_32769", long_distance(32769));
    c.emplace_back("max_len_258", [] {
        Bytes b = random_bytes(258, 6);
        Bytes copy = b;
        for (int i = 0; i < 4; ++i) b.insert(b.end(), copy.begin(), copy.end());
        return b;
    }());
    c.emplace_back("period_3", [] {
        Bytes b;
        for (int i = 0; i < 5000; ++i) b.push_back(static_cast<unsigned char>("abc"[i % 3]));
        return b;
    }());
    return c;
}

bool python_available() {
    static const bool ok = std::system("python3 -c \"import zlib\" > /dev/null 2>&1") == 0;
    return ok;
}

// True if Python's zlib.decompress(compressed) == original (wbits selects zlib vs raw).
bool python_decompresses_to(const Bytes& compressed, const Bytes& original, int wbits) {
    const fs::path dir = fs::temp_directory_path();
    const fs::path cpath = dir / ("dm_zlib_c_" + std::to_string(std::rand()) + ".bin");
    const fs::path opath = dir / ("dm_zlib_o_" + std::to_string(std::rand()) + ".bin");
    write_file(cpath, compressed);
    write_file(opath, original);
    const std::string cmd = "python3 -c \"import zlib,sys; c=open(sys.argv[1],'rb').read(); o=open(sys.argv[2],'rb').read(); "
                            "sys.exit(0 if zlib.decompress(c," +
                            std::to_string(wbits) + ")==o else 1)\" '" + cpath.string() + "' '" + opath.string() + "' > /dev/null 2>&1";
    const bool ok = std::system(cmd.c_str()) == 0;
    fs::remove(cpath);
    fs::remove(opath);
    return ok;
}

} // namespace

TEST(ZlibCompress, RoundTripsTheCorpusAtEveryLevel) {
    for (const auto& [name, data] : corpus()) {
        for (int level = 0; level <= 9; ++level) {
            SCOPED_TRACE(name + " level " + std::to_string(level));
            const Bytes z = zlib_compress(data, level);
            EXPECT_EQ(zlib_decompress(z, data.size()), data);
            EXPECT_EQ(zlib_decompress(z), data);
            const Bytes raw = deflate_compress(data, level);
            EXPECT_EQ(deflate_decompress(raw.data(), raw.size()), data);
            EXPECT_EQ(deflate_decompress(raw.data(), raw.size(), data.size()), data);
        }
    }
}

TEST(ZlibCompress, IsDeterministicAndDefaultsToLevel6) {
    const Bytes data = text_like(20000);
    EXPECT_EQ(zlib_compress(data), zlib_compress(data, 6));
    EXPECT_EQ(zlib_compress(data, -1), zlib_compress(data, 6));
    EXPECT_EQ(zlib_compress(data, 9), zlib_compress(data, 9));
    EXPECT_THROW((void)zlib_compress(data, 10), std::invalid_argument);
    EXPECT_THROW((void)deflate_compress(data, -2), std::invalid_argument);
}

TEST(ZlibCompress, Level0IsStoredOnly) {
    const Bytes data = text_like(70000); // two stored blocks
    const Bytes z = zlib_compress(data, 0);
    EXPECT_EQ(z.size(), 2 + data.size() + 2 * 5 + 4);
    EXPECT_EQ(z[0], 0x78);
    EXPECT_EQ(z[1], 0x01);
    EXPECT_EQ(z[2], 0x00); // non-final stored block header
    EXPECT_EQ(zlib_decompress(z), data);
}

TEST(ZlibCompress, EmitsTheValidHeaderForEachLevel) {
    for (int level = 0; level <= 9; ++level) {
        const Bytes z = zlib_compress(Bytes{1, 2, 3}, level);
        EXPECT_EQ(z[0], 0x78);
        EXPECT_EQ(((z[0] << 8) | z[1]) % 31, 0);
    }
    EXPECT_EQ(zlib_compress(Bytes{}, 6), (Bytes{0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01})); // same as zlib
}

TEST(ZlibCompress, FindsMatchesAtTheWindowEdgeAndOfMaximumLength) {
    // A 600-byte random segment repeated exactly 32768 bytes later is matched (saving ~600
    // bytes); one byte farther is outside the window and stays literal.
    for (const int level : {1, 6, 9}) {
        const std::size_t in_window = zlib_compress(long_distance(32768), level).size();
        const std::size_t outside = zlib_compress(long_distance(32769), level).size();
        EXPECT_LT(in_window + 500, outside) << "level " << level;
    }
    // 258 random bytes repeated 4 more times: ~4 max-length matches, so far under 2x258 bytes.
    Bytes b = random_bytes(258, 6);
    const Bytes seg = b;
    for (int i = 0; i < 4; ++i) b.insert(b.end(), seg.begin(), seg.end());
    EXPECT_LT(zlib_compress(b, 6).size(), 258u + 60u);
}

TEST(ZlibCompress, CompressesRedundantDataWell) {
    const Bytes text = text_like(100000);
    EXPECT_LT(zlib_compress(text, 6).size(), text.size() * 40 / 100);
    EXPECT_LT(zlib_compress(Bytes(100000, 7), 6).size(), 300u);
    const Bytes rnd = random_bytes(100000, 8); // incompressible: falls back to stored blocks
    EXPECT_LE(zlib_compress(rnd, 6).size(), rnd.size() * 1001 / 1000 + 16); // zlib itself: 100041
    // Higher levels never do worse than level 1 on these inputs.
    EXPECT_LE(zlib_compress(text, 9).size(), zlib_compress(text, 1).size());
}

TEST(ZlibCompress, SizeIsCloseToRealZlibLevel6) {
    for (const auto& name : {"text", "scanlines"}) {
        SCOPED_TRACE(name);
        const std::string in_name = std::string(name) == "text" ? "text.txt" : "scanlines.bin";
        const Bytes input = read_file(data_dir() / in_name);
        const Bytes reference = read_file(data_dir() / (std::string(name) + ".l6.zz"));
        const Bytes ours = zlib_compress(input, 6);
        EXPECT_LE(static_cast<double>(ours.size()), 1.10 * static_cast<double>(reference.size()))
            << "ours " << ours.size() << " vs zlib " << reference.size();
    }
}

TEST(ZlibDecompress, ReadsPythonZlibFixturesWithoutKnowingTheSize) {
    const Bytes text = read_file(data_dir() / "text.txt");
    for (const char* level : {"0", "1", "6", "9"}) {
        SCOPED_TRACE(level);
        const Bytes z = read_file(data_dir() / (std::string("text.l") + level + ".zz"));
        EXPECT_EQ(zlib_decompress(z), text);
        EXPECT_EQ(zlib_decompress(z, text.size()), text);
    }
    const Bytes scan = read_file(data_dir() / "scanlines.bin");
    EXPECT_EQ(zlib_decompress(read_file(data_dir() / "scanlines.l6.zz")), scan);
}

TEST(ZlibCompress, OutputDecompressesWithPythonZlib) {
    if (!python_available()) GTEST_SKIP() << "python3 with zlib not available";
    for (const auto& [name, data] : corpus()) {
        for (const int level : {0, 1, 4, 6, 9}) {
            SCOPED_TRACE(name + " level " + std::to_string(level));
            EXPECT_TRUE(python_decompresses_to(zlib_compress(data, level), data, 15));
            if (level == 6) EXPECT_TRUE(python_decompresses_to(deflate_compress(data, level), data, -15));
        }
    }
}

TEST(ZlibDecompress, EnforcesMaxOutput) {
    const Bytes bomb = zlib_compress(Bytes(1 << 20, 0), 9); // ~1 KiB -> 1 MiB
    EXPECT_LT(bomb.size(), 4096u);
    EXPECT_THROW((void)zlib_decompress(bomb, InflateLimit{1000}), std::runtime_error);
    EXPECT_EQ(zlib_decompress(bomb, InflateLimit{1 << 20}).size(), std::size_t{1} << 20);
    const Bytes raw = deflate_compress(Bytes(5000, 1), 6);
    EXPECT_THROW((void)deflate_decompress(raw.data(), raw.size(), InflateLimit{4999}), std::runtime_error);
    EXPECT_EQ(deflate_decompress(raw.data(), raw.size(), InflateLimit{5000}).size(), 5000u);
    // Sized overloads keep their exact-size check in both directions.
    EXPECT_THROW((void)deflate_decompress(raw.data(), raw.size(), std::size_t{4999}), std::runtime_error);
    EXPECT_THROW((void)deflate_decompress(raw.data(), raw.size(), std::size_t{5001}), std::runtime_error);
}

TEST(ZlibDecompress, TruncatedStreamsThrow) {
    for (const int level : {0, 1, 6, 9}) {
        const Bytes data = text_like(3000);
        const Bytes z = zlib_compress(data, level);
        for (std::size_t len = 0; len < z.size(); ++len) {
            const Bytes cut(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(len));
            EXPECT_THROW((void)zlib_decompress(cut), std::runtime_error) << "level " << level << " len " << len;
        }
        const Bytes raw = deflate_compress(data, level);
        for (std::size_t len = 0; len < raw.size(); len += 7) {
            EXPECT_THROW((void)deflate_decompress(raw.data(), len), std::runtime_error) << "level " << level << " len " << len;
        }
    }
}

TEST(ZlibDecompress, BitFlipsThrowOrReproduceTheInputAndNeverCrash) {
    std::mt19937 rng(77);
    const Bytes data = text_like(4000);
    for (const int level : {1, 6, 9}) {
        const Bytes z = zlib_compress(data, level);
        int thrown = 0;
        const int trials = 600;
        for (int t = 0; t < trials; ++t) {
            Bytes bad = z;
            const std::size_t bit = rng() % (bad.size() * 8);
            bad[bit / 8] = static_cast<unsigned char>(bad[bit / 8] ^ (1u << (bit % 8)));
            try {
                // A flip in padding bits is harmless; anything else must be detected.
                EXPECT_EQ(zlib_decompress(bad, InflateLimit{1 << 20}), data);
            } catch (const std::runtime_error&) {
                ++thrown;
            }
            // Raw DEFLATE has no checksum: only require "no crash, runtime_error at worst".
            try {
                (void)deflate_decompress(bad.data() + 2, bad.size() - 6, InflateLimit{1 << 20});
            } catch (const std::runtime_error&) {
            }
        }
        EXPECT_GT(thrown, trials * 9 / 10);
    }
}

TEST(ZlibDecompress, RandomGarbageNeverCrashes) {
    std::mt19937 rng(5);
    for (int t = 0; t < 2000; ++t) {
        Bytes junk = random_bytes(1 + rng() % 200, static_cast<unsigned>(t));
        try {
            (void)deflate_decompress(junk.data(), junk.size(), InflateLimit{1 << 16});
        } catch (const std::runtime_error&) {
        }
    }
    SUCCEED();
}
