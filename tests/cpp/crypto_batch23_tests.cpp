// Tests for the cryptographic-primitives batch: MD5, SHA-1, SHA-256, HMAC-SHA256,
// Diffie-Hellman, and Shamir's secret sharing. Reference values are the published
// test vectors (RFC 1321, FIPS 180, RFC 4231) and hand-computed number-theory checks.

#include <gtest/gtest.h>

#include <datamunge/algorithms/diffie_hellman.hpp>
#include <datamunge/algorithms/hmac.hpp>
#include <datamunge/algorithms/md5.hpp>
#include <datamunge/algorithms/sha1.hpp>
#include <datamunge/algorithms/sha256.hpp>
#include <datamunge/algorithms/shamir_secret_sharing.hpp>

#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---------------------------------------------------------------------------- MD5
TEST(Md5, Rfc1321Vectors) {
    EXPECT_EQ(md5_hex(""), "d41d8cd98f00b204e9800998ecf8427e");
    EXPECT_EQ(md5_hex("a"), "0cc175b9c0f1b6a831c399e269772661");
    EXPECT_EQ(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    EXPECT_EQ(md5_hex("message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
    EXPECT_EQ(md5_hex("abcdefghijklmnopqrstuvwxyz"), "c3fcd3d76192e4007dfb496cca67e13b");
    EXPECT_EQ(md5_hex("12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
              "57edf4a22be3c955ac49da2e2107b67a");
}

TEST(Md5, CrossesBlockBoundary) {
    // 56 bytes forces a second padding block; just check determinism + length.
    std::string s(56, 'x');
    EXPECT_EQ(md5_hex(s).size(), 32u);
    EXPECT_EQ(md5_hex(s), md5_hex(s));
}

// ---------------------------------------------------------------------------- SHA-1
TEST(Sha1, Fips180Vectors) {
    EXPECT_EQ(sha1_hex(""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(sha1_hex("abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(sha1_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
}

// ---------------------------------------------------------------------------- SHA-256
TEST(Sha256, Fips1804Vectors) {
    EXPECT_EQ(sha256_hex(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

// ---------------------------------------------------------------------------- HMAC
TEST(HmacSha256, Rfc4231Case1) {
    // Key = 0x0b x20, data = "Hi There".
    std::string key(20, '\x0b');
    EXPECT_EQ(hmac_sha256_hex(key, "Hi There"),
              "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

TEST(HmacSha256, Rfc4231Case2) {
    EXPECT_EQ(hmac_sha256_hex("Jefe", "what do ya want for nothing?"),
              "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

TEST(HmacSha256, LongKeyIsHashed) {
    std::string key(131, '\xaa');  // longer than the 64-byte block -> hashed down first
    // RFC 4231 test case 6 first block: key of 131 0xaa bytes, "Test Using Larger Than Block-Size Key - Hash Key First"
    EXPECT_EQ(hmac_sha256_hex(key, "Test Using Larger Than Block-Size Key - Hash Key First"),
              "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
}

// ---------------------------------------------------------------------------- Diffie-Hellman
TEST(DiffieHellman, SmallGroupSharedSecretMatches) {
    // Classic textbook example: p = 23, g = 5.
    const std::uint64_t p = 23, g = 5;
    const std::uint64_t a = 6, b = 15;
    std::uint64_t A = dh_public_key(g, a, p);  // 5^6 mod 23 = 8
    std::uint64_t B = dh_public_key(g, b, p);  // 5^15 mod 23 = 19
    EXPECT_EQ(A, 8u);
    EXPECT_EQ(B, 19u);
    std::uint64_t s_alice = dh_shared_secret(B, a, p);
    std::uint64_t s_bob = dh_shared_secret(A, b, p);
    EXPECT_EQ(s_alice, s_bob);
    EXPECT_EQ(s_alice, 2u);  // g^(ab) = 5^90 mod 23 = 2
}

TEST(DiffieHellman, LargerPrimeAgrees) {
    const std::uint64_t p = 2147483647ULL;  // Mersenne prime 2^31 - 1
    const std::uint64_t g = 7;
    const std::uint64_t a = 123456, b = 987654;
    std::uint64_t A = dh_public_key(g, a, p);
    std::uint64_t B = dh_public_key(g, b, p);
    EXPECT_EQ(dh_shared_secret(B, a, p), dh_shared_secret(A, b, p));
}

// ---------------------------------------------------------------------------- Shamir
TEST(Shamir, ThresholdReconstructs) {
    const std::uint64_t prime = 2087;  // small prime > secret and n
    const std::uint64_t secret = 1234;
    // threshold k = 3 -> two random coefficients.
    std::vector<std::uint64_t> coeffs = {166, 94};
    auto shares = shamir_split(secret, 6, coeffs, prime);
    ASSERT_EQ(shares.size(), 6u);

    // Any 3 shares recover the secret.
    std::vector<SecretShare> pick1 = {shares[0], shares[2], shares[4]};
    std::vector<SecretShare> pick2 = {shares[1], shares[3], shares[5]};
    EXPECT_EQ(shamir_combine(pick1, prime), secret);
    EXPECT_EQ(shamir_combine(pick2, prime), secret);

    // More than the threshold also works.
    std::vector<SecretShare> pick3 = {shares[0], shares[1], shares[2], shares[3], shares[4]};
    EXPECT_EQ(shamir_combine(pick3, prime), secret);
}

TEST(Shamir, FewerThanThresholdFails) {
    const std::uint64_t prime = 2087;
    const std::uint64_t secret = 1234;
    std::vector<std::uint64_t> coeffs = {166, 94};  // k = 3
    auto shares = shamir_split(secret, 6, coeffs, prime);
    // Two shares interpolate a line P(0) that is almost surely NOT the secret.
    std::vector<SecretShare> two = {shares[0], shares[1]};
    EXPECT_NE(shamir_combine(two, prime), secret);
}
