#pragma once

/// \file rc4.hpp
/// \brief The RC4 stream cipher (Rivest, 1987).
///
/// RC4 is a byte-oriented stream cipher built on a 256-entry permutation of the bytes
/// \f$0..255\f$. The *key-scheduling algorithm* (KSA) shuffles the identity permutation
/// under the key; the *pseudo-random generation algorithm* (PRGA) then walks the
/// permutation, swapping entries and emitting one keystream byte per step:
/// \f[
///   i\leftarrow i+1,\quad j\leftarrow j+S[i],\quad \text{swap}(S[i],S[j]),\quad
///   K\leftarrow S[\,S[i]+S[j]\,],
/// \f]
/// all arithmetic mod 256. Encryption XORs the keystream with the plaintext. RC4 is
/// spectacularly simple and was ubiquitous (WEP, early TLS), but it is *broken*: the first
/// keystream bytes are biased (the Fluhrer-Mantin-Shamir and later attacks), which sank WEP
/// and led to RC4's prohibition in TLS. It survives only as a historical example. This
/// implementation reproduces the classic published keystream vectors.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// \brief RC4 keystream generator (KSA + PRGA state).
class Rc4 {
   public:
    /// \brief Initialise the permutation from \p key via the key-scheduling algorithm.
    explicit Rc4(const std::string& key) {
        for (int k = 0; k < 256; ++k) s_[k] = static_cast<std::uint8_t>(k);
        std::uint8_t j = 0;
        for (int k = 0; k < 256; ++k) {
            j = static_cast<std::uint8_t>(j + s_[k] +
                                          static_cast<std::uint8_t>(key[k % key.size()]));
            std::swap(s_[k], s_[j]);
        }
    }

    /// \brief Emit the next keystream byte (PRGA step).
    std::uint8_t next() {
        i_ = static_cast<std::uint8_t>(i_ + 1);
        j_ = static_cast<std::uint8_t>(j_ + s_[i_]);
        std::swap(s_[i_], s_[j_]);
        return s_[static_cast<std::uint8_t>(s_[i_] + s_[j_])];
    }

   private:
    std::array<std::uint8_t, 256> s_{};
    std::uint8_t i_ = 0, j_ = 0;
};

/// \brief First \p n keystream bytes for \p key.
inline std::vector<std::uint8_t> rc4_keystream(const std::string& key, std::size_t n) {
    Rc4 rc4(key);
    std::vector<std::uint8_t> ks(n);
    for (std::size_t t = 0; t < n; ++t) ks[t] = rc4.next();
    return ks;
}

/// \brief Encrypt (or decrypt) \p data under \p key; decryption is the same operation.
inline std::string rc4_encrypt(const std::string& key, const std::string& data) {
    Rc4 rc4(key);
    std::string out = data;
    for (char& ch : out) ch = static_cast<char>(static_cast<std::uint8_t>(ch) ^ rc4.next());
    return out;
}

}  // namespace datamunge::algorithms
