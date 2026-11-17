#pragma once
/// @file tea.hpp
/// @brief Actual32-cycle TEA word transitions with fixed-width modular arithmetic.
#include <array>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the TEAStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TEAStep r{};
/// (void)r;
/// @endcode
/// @see tea_encrypt_run()
struct TEAStep {
  std::uint32_t
      cycle{}; ///< Actual completed cycle1..32, each containing two dependent word updates.
  std::uint32_t
      sum{}; ///< Actual schedule word used for this cycle; decryption records before decrement.
  std::array<std::uint32_t, 2> before{}; ///< Actual two words before the cycle.
  std::uint32_t
      first_updated{}; ///< Actual first modified half; left for encryption, right for decryption.
  std::array<std::uint32_t, 2> after{}; ///< Actual two words after both dependent updates.
};
/// @brief Own the TEARun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TEARun r{};
/// (void)r;
/// @endcode
/// @see tea_encrypt_run()
struct TEARun {
  std::array<std::uint32_t, 2>
                       value{}; ///< Actual final two-word block, no implicit byte serialization.
  bool                 decrypting{};   ///< Actual inverse schedule selected.
  std::uint32_t        cycles{};       ///< Actual executed double-round cycles, exactly32.
  std::uint32_t        half_updates{}; ///< Actual dependent half updates, exactly64.
  std::vector<TEAStep> steps; ///< Optional actual before/intermediate/after words per cycle.
};
/// @brief Execute actual TEA encryption cycles with optional transition capture.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned final block, actual cycle/half-update counts and optional states.
/// @par Example
/// @code
/// auto r=dm::algorithms::tea_encrypt_run({0,0},{0,0,0,0});
/// std::print("{}\n",r.cycles); // 32
/// @endcode
/// @par References
/// - Wheeler and Needham, TEA, a Tiny Encryption Algorithm, FSE1994,363-366, doi:10.1007/3-540-60590-8_29.
/// @see tea_encrypt()
TEARun tea_encrypt_run(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k);
/// @brief Execute actual TEA encryption cycles with optional transition capture.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned final block, actual cycle/half-update counts and optional states.
/// @par Example
/// @code
/// auto r=dm::algorithms::tea_encrypt_trace({0,0},{0,0,0,0});
/// std::print("{}\n",r.cycles); // 32
/// @endcode
/// @par References
/// - Wheeler and Needham, TEA, a Tiny Encryption Algorithm, FSE1994,363-366, doi:10.1007/3-540-60590-8_29.
/// @see tea_encrypt()
TEARun tea_encrypt_trace(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k);
/// @brief Preserve the two-word TEA encryption signature.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned actual final two-word block.
/// @par Example
/// @code
/// auto c=dm::algorithms::tea_encrypt({0,0},{0,0,0,0});
/// auto p=dm::algorithms::tea_decrypt(c,{0,0,0,0});
/// std::print("{}\n",p); // [0, 0]
/// @endcode
/// @see tea_encrypt_run()
std::array<std::uint32_t, 2> tea_encrypt(
    std::array<std::uint32_t, 2>        v,
    const std::array<std::uint32_t, 4>& k);
/// @brief Execute actual TEA decryption cycles with optional transition capture.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned final block, actual cycle/half-update counts and optional states.
/// @par Example
/// @code
/// auto r=dm::algorithms::tea_decrypt_run({0,0},{0,0,0,0});
/// std::print("{}\n",r.cycles); // 32
/// @endcode
/// @par References
/// - Wheeler and Needham, TEA, a Tiny Encryption Algorithm, FSE1994,363-366, doi:10.1007/3-540-60590-8_29.
/// @see tea_decrypt()
TEARun tea_decrypt_run(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k);
/// @brief Execute actual TEA decryption cycles with optional transition capture.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned final block, actual cycle/half-update counts and optional states.
/// @par Example
/// @code
/// auto r=dm::algorithms::tea_decrypt_trace({0,0},{0,0,0,0});
/// std::print("{}\n",r.cycles); // 32
/// @endcode
/// @par References
/// - Wheeler and Needham, TEA, a Tiny Encryption Algorithm, FSE1994,363-366, doi:10.1007/3-540-60590-8_29.
/// @see tea_decrypt()
TEARun tea_decrypt_trace(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k);
/// @brief Preserve the two-word TEA decryption signature.
/// Two uint32 words and four uint32 key words are consumed by value/reference. Executes32 cycles, each with two dependent word updates, modulo2^32 using defined unsigned arithmetic. Encryption increments sum by0x9E3779B9 before each cycle; decryption starts0xC6EF3720 and reverses the half order before decrementing sum. No implicit endian conversion, padding, mode, authentication or key management. TEA has known equivalent-key and related-key weaknesses; XTEA/XXTEA are different algorithms and not a blanket repair of all security issues. This is an educational primitive. Fixed constant work/storage, optional32 actual records; no shared state.
/// @param v Complete two-word block.
/// @param k Borrowed complete four-word key.
/// @return Owned actual final two-word block.
/// @par Example
/// @code
/// auto c=dm::algorithms::tea_encrypt({0,0},{0,0,0,0});
/// auto p=dm::algorithms::tea_decrypt(c,{0,0,0,0});
/// std::print("{}\n",p); // [0, 0]
/// @endcode
/// @see tea_decrypt_run()
std::array<std::uint32_t, 2> tea_decrypt(
    std::array<std::uint32_t, 2>        v,
    const std::array<std::uint32_t, 4>& k);
} // namespace datamunge::algorithms
