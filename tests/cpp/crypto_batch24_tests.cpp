// Tests for the symmetric-cipher / KDF batch: ChaCha20, Salsa20, TEA, RC4, PBKDF2, SipHash.
// Reference values are the published test vectors (RFC 8439, RFC 6070-style PBKDF2-SHA256,
// classic RC4 keystreams, the SipHash paper) and round-trip / self-inverse checks.

#include <gtest/gtest.h>

#include <datamunge/algorithms/chacha20.hpp>
#include <datamunge/algorithms/pbkdf2.hpp>
#include <datamunge/algorithms/rc4.hpp>
#include <datamunge/algorithms/salsa20.hpp>
#include <datamunge/algorithms/siphash.hpp>
#include <datamunge/algorithms/tea.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

namespace {
std::string bytes_to_hex(const std::vector<std::uint8_t>& v) {
    static const char* h = "0123456789abcdef";
    std::string s;
    for (std::uint8_t b : v) { s.push_back(h[b >> 4]); s.push_back(h[b & 0xF]); }
    return s;
}
template <std::size_t N>
std::string arr_to_hex(const std::array<std::uint8_t, N>& v) {
    static const char* h = "0123456789abcdef";
    std::string s;
    for (std::uint8_t b : v) { s.push_back(h[b >> 4]); s.push_back(h[b & 0xF]); }
    return s;
}
}  // namespace

// ---------------------------------------------------------------------------- ChaCha20
TEST(ChaCha20, Rfc8439BlockVector) {
    // RFC 8439 section 2.3.2 keystream block.
    std::array<std::uint8_t, 32> key{};
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i);
    std::array<std::uint8_t, 12> nonce = {0x00, 0x00, 0x00, 0x09, 0x00, 0x00,
                                          0x00, 0x4a, 0x00, 0x00, 0x00, 0x00};
    auto block = chacha20_block(key, 1, nonce);
    EXPECT_EQ(arr_to_hex(block),
              "10f1e7e4d13b5915500fdd1fa32071c4"
              "c7d1f4c733c068030422aa9ac3d46c4e"
              "d2826446079faa0914c2d705d98b02a2"
              "b5129cd1de164eb9cbd083e8a2503c4e");
}

TEST(ChaCha20, Rfc8439EncryptSunscreen) {
    // RFC 8439 section 2.4.2: encrypt the "sunscreen" plaintext, counter starts at 1.
    std::array<std::uint8_t, 32> key{};
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i);
    std::array<std::uint8_t, 12> nonce = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x4a, 0x00, 0x00, 0x00, 0x00};
    std::string pt =
        "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for "
        "the future, sunscreen would be it.";
    std::string ct = chacha20_encrypt(key, 1, nonce, pt);
    std::vector<std::uint8_t> ctb(ct.begin(), ct.end());
    EXPECT_EQ(bytes_to_hex(ctb).substr(0, 32), "6e2e359a2568f98041ba0728dd0d6981");
    // Round-trip.
    EXPECT_EQ(chacha20_encrypt(key, 1, nonce, ct), pt);
}

// ---------------------------------------------------------------------------- Salsa20
TEST(Salsa20, BernsteinCoreVector) {
    // The Salsa20 core (doubleround*10 + feed-forward) documented example.
    std::array<std::uint8_t, 64> in = {
        211, 159, 13, 115, 76, 55, 82, 183, 3, 117, 222, 37, 191, 187, 234, 136,
        49, 237, 179, 48, 1, 106, 178, 219, 175, 199, 166, 48, 86, 16, 179, 207,
        31, 240, 32, 63, 15, 83, 93, 161, 116, 147, 48, 113, 238, 55, 204, 36,
        79, 201, 235, 79, 3, 81, 156, 47, 203, 26, 244, 243, 88, 118, 104, 54};
    std::array<std::uint8_t, 64> expected = {
        109, 42, 178, 168, 156, 240, 248, 238, 168, 196, 190, 203, 26, 110, 170, 154,
        29, 29, 150, 26, 150, 30, 235, 249, 190, 163, 251, 48, 69, 144, 51, 57,
        118, 40, 152, 157, 180, 57, 27, 94, 107, 42, 236, 35, 27, 111, 114, 114,
        219, 236, 232, 135, 111, 155, 110, 18, 24, 232, 95, 158, 179, 19, 48, 202};
    EXPECT_EQ(salsa20_core(in), expected);
}

TEST(Salsa20, EncryptRoundTrip) {
    std::array<std::uint8_t, 32> key{};
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i * 7 + 1);
    std::array<std::uint8_t, 8> nonce = {1, 2, 3, 4, 5, 6, 7, 8};
    std::string pt = "Salsa20 stream cipher round-trip over multiple 64-byte blocks!!";
    std::string ct = salsa20_encrypt(key, 0, nonce, pt);
    EXPECT_NE(ct, pt);
    EXPECT_EQ(salsa20_encrypt(key, 0, nonce, ct), pt);
}

// ---------------------------------------------------------------------------- TEA
TEST(Tea, RoundTrip) {
    std::array<std::uint32_t, 4> key = {0x01234567u, 0x89abcdefu, 0xfedcba98u, 0x76543210u};
    std::array<std::uint32_t, 2> pt = {0xdeadbeefu, 0x01234567u};
    auto ct = tea_encrypt(pt, key);
    EXPECT_NE(ct[0], pt[0]);
    auto back = tea_decrypt(ct, key);
    EXPECT_EQ(back[0], pt[0]);
    EXPECT_EQ(back[1], pt[1]);
}

TEST(Tea, ZeroKeyZeroBlockRoundTrips) {
    std::array<std::uint32_t, 4> key = {0, 0, 0, 0};
    std::array<std::uint32_t, 2> pt = {0, 0};
    auto ct = tea_encrypt(pt, key);
    EXPECT_EQ(tea_decrypt(ct, key), pt);
}

// ---------------------------------------------------------------------------- RC4
TEST(Rc4, ClassicKeystreamVectors) {
    // "Key" -> keystream, XOR "Plaintext" = BBF316E8D940AF0AD3.
    EXPECT_EQ(bytes_to_hex(rc4_keystream("Key", 3)).substr(0, 6), "eb9f77");
    std::string ct = rc4_encrypt("Key", "Plaintext");
    std::vector<std::uint8_t> ctb(ct.begin(), ct.end());
    EXPECT_EQ(bytes_to_hex(ctb), "bbf316e8d940af0ad3");
    // "Wiki" / "pedia" -> 1021BF0420.
    std::string ct2 = rc4_encrypt("Wiki", "pedia");
    std::vector<std::uint8_t> ct2b(ct2.begin(), ct2.end());
    EXPECT_EQ(bytes_to_hex(ct2b), "1021bf0420");
    // "Secret" / "Attack at dawn" -> 45A01F645FC35B383552544B9BF5.
    std::string ct3 = rc4_encrypt("Secret", "Attack at dawn");
    std::vector<std::uint8_t> ct3b(ct3.begin(), ct3.end());
    EXPECT_EQ(bytes_to_hex(ct3b), "45a01f645fc35b383552544b9bf5");
}

// ---------------------------------------------------------------------------- PBKDF2
TEST(Pbkdf2HmacSha256, StandardVectors) {
    EXPECT_EQ(pbkdf2_hmac_sha256_hex("password", "salt", 1, 32),
              "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b");
    EXPECT_EQ(pbkdf2_hmac_sha256_hex("password", "salt", 2, 32),
              "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43");
    EXPECT_EQ(pbkdf2_hmac_sha256_hex("password", "salt", 4096, 32),
              "c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a");
}

TEST(Pbkdf2HmacSha256, MultiBlockOutput) {
    // 40-byte output spans two HMAC blocks. RFC-style multi-word-salt vector.
    EXPECT_EQ(pbkdf2_hmac_sha256_hex("passwordPASSWORDpassword",
                                     "saltSALTsaltSALTsaltSALTsaltSALTsalt", 4096, 40),
              "348c89dbcbd32b2f32d814b8116e84cf2b17347ebc1800181c4e2a1fb8dd53e1c635518c7dac47e9");
}

// ---------------------------------------------------------------------------- SipHash
TEST(SipHash24, ReferencePaperVector) {
    // Key = 000102...0f, message = 000102...0e (15 bytes): 0xa129ca6149be45e5 (SipHash paper).
    std::array<std::uint8_t, 16> key{};
    for (int i = 0; i < 16; ++i) key[i] = static_cast<std::uint8_t>(i);
    std::string msg;
    for (int i = 0; i < 15; ++i) msg.push_back(static_cast<char>(i));
    EXPECT_EQ(siphash24(key, msg), 0xa129ca6149be45e5ULL);
}

TEST(SipHash24, EmptyMessageVector) {
    std::array<std::uint8_t, 16> key{};
    for (int i = 0; i < 16; ++i) key[i] = static_cast<std::uint8_t>(i);
    EXPECT_EQ(siphash24(key, ""), 0x726fdb47dd0e0e31ULL);
}

TEST(SipHash24, KeyDependence) {
    std::array<std::uint8_t, 16> k1{}, k2{};
    for (int i = 0; i < 16; ++i) { k1[i] = static_cast<std::uint8_t>(i); k2[i] = static_cast<std::uint8_t>(i); }
    k2[0] ^= 1;
    EXPECT_NE(siphash24(k1, "hash flooding"), siphash24(k2, "hash flooding"));
}
