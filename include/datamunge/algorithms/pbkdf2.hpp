#pragma once
/// @file pbkdf2.hpp
/// @brief Checked PBKDF2-HMAC-SHA256 with actual call counts and optional chain bytes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Complete derivation work and retained-history policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::Pbkdf2Options o;std::print("{}\n",o.max_hmac_calls); // 1000000
/// @endcode
struct Pbkdf2Options {
  std::size_t max_hmac_calls{
      1000000}; ///< Planned actual HMAC call cap; not elapsed time or SHA blocks.
  std::size_t max_trace_entries{
      1000000}; ///< Retained chain-record cap, each containing two 32-byte arrays; excludes final key.
};
/// @brief Actual HMAC result and cumulative XOR for one producing block/round.
/// Contains derived key material; owned educational storage is not securely erased.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256_trace("p","s",1,1).steps[0].round); // 1
/// @endcode
struct Pbkdf2Step {
  std::uint32_t block;            ///< One-based output block encoded big endian into U1's message.
  std::uint32_t round;            ///< One-based HMAC iteration within this block.
  std::array<std::uint8_t, 32> u; ///< Actual HMAC output U_j.
  std::array<std::uint8_t, 32> accumulator; ///< Actual running XOR after incorporating U_j.
};
/// @brief Owned derived bytes and diagnostics from one producing derivation.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256_run("p","s",2,1).hmac_calls); // 2
/// @endcode
struct Pbkdf2Run {
  std::vector<std::uint8_t> key; ///< Requested prefix of concatenated 32-byte block XORs.
  std::size_t hmac_calls{0};     ///< Actual calls performed, iterations times ceil(key length/32).
  std::vector<Pbkdf2Step> steps; ///< Optional actual chain history; empty for ordinary run.
};
/// @brief Execute PBKDF2 with HMAC-SHA256 and checked work before deriving.
/// Binary strings include embedded zeros. Empty output still validates iterations.
/// Requires c>=1 and ceil(dk_len/32)<=UINT32_MAX; length calculations cannot wrap.
/// Existing HMAC input-byte admission also applies to password plus each message.
/// O(c*ceil(dk_len/32)) HMAC calls; O(dk_len) working output beyond HMAC internals.
/// Borrowed inputs are unchanged, no references retained, no shared mutable state.
/// @param password Complete binary password, possibly empty.
/// @param salt Complete binary salt, possibly empty; no salt is generated.
/// @param iterations HMAC calls per output block, positive uint32 count.
/// @param dk_len Requested bytes, possibly zero.
/// @param options Preflight work/history policies, default one million each.
/// @return Owned key, actual HMAC call count and no chain history.
/// @throws std::invalid_argument If iterations is zero.
/// @throws std::length_error If standard length, input or work limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256_run("password","salt",2,33).hmac_calls); // 4
/// @endcode
/// @par References
/// - RFC 8018 section 5.2, https://www.rfc-editor.org/rfc/rfc8018#section-5.2
Pbkdf2Run pbkdf2_hmac_sha256_run(
    const std::string& password,
    const std::string& salt,
    std::uint32_t      iterations,
    std::size_t        dk_len,
    Pbkdf2Options      options = {});
/// @brief Retain every actual U_j and cumulative XOR from the producing derivation.
/// Same validation and algorithm as pbkdf2_hmac_sha256_run(); O(c*blocks) extra
/// records, admitted before hashing. Contains key material and performs no erasure.
/// @param password Complete binary password.
/// @param salt Complete binary salt.
/// @param iterations Positive HMAC calls per block.
/// @param dk_len Requested output bytes, possibly zero.
/// @param options Work and retained-record policies.
/// @return Owned key, actual call count and complete producing chain history.
/// @throws std::invalid_argument If iterations is zero.
/// @throws std::length_error If standard/input/work/history limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256_trace("p","s",2,1).steps.size()); // 2
/// @endcode
Pbkdf2Run pbkdf2_hmac_sha256_trace(
    const std::string& password,
    const std::string& salt,
    std::uint32_t      iterations,
    std::size_t        dk_len,
    Pbkdf2Options      options = {});
/// @brief Return only derived bytes using default checked PBKDF2 work admission.
/// @param password Complete binary password.
/// @param salt Complete binary salt.
/// @param iterations Positive HMAC calls per block.
/// @param dk_len Requested key bytes, possibly zero.
/// @return Owned derived bytes; borrows inputs without mutation.
/// @throws std::invalid_argument If iterations is zero.
/// @throws std::length_error If standard/input/default-work limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256("p","s",1,0).size()); // 0
/// @endcode
std::vector<std::uint8_t> pbkdf2_hmac_sha256(
    const std::string& password,
    const std::string& salt,
    std::uint32_t      iterations,
    std::size_t        dk_len);
/// @brief Format the derived bytes as lowercase hexadecimal, two characters per byte.
/// @param password Complete binary password.
/// @param salt Complete binary salt.
/// @param iterations Positive HMAC calls per block.
/// @param dk_len Requested bytes; hex requires twice this representable storage.
/// @return Owned lowercase hexadecimal key; empty for zero length.
/// @throws std::invalid_argument If iterations is zero.
/// @throws std::length_error If standard/input/work/string capacity limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pbkdf2_hmac_sha256_hex("password","salt",1,1)); // 12
/// @endcode
std::string pbkdf2_hmac_sha256_hex(
    const std::string& password,
    const std::string& salt,
    std::uint32_t      iterations,
    std::size_t        dk_len);
} // namespace datamunge::algorithms
