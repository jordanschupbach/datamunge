#pragma once
/// @file verhoeff.hpp
/// @brief Checked native verhoeff and actual producing diagnostics.
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::algorithms {
namespace detail {
// Dihedral-group D5 multiplication table.
/// @brief Return the immutable canonical Verhoeff lookup table.
inline const std::array<std::array<int, 10>, 10>& verhoeff_d() {
  /// @brief Canonical immutable Verhoeff lookup coefficients.
  static const std::array<std::array<int, 10>, 10> d = {
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
       {1, 2, 3, 4, 0, 6, 7, 8, 9, 5},
       {2, 3, 4, 0, 1, 7, 8, 9, 5, 6},
       {3, 4, 0, 1, 2, 8, 9, 5, 6, 7},
       {4, 0, 1, 2, 3, 9, 5, 6, 7, 8},
       {5, 9, 8, 7, 6, 0, 4, 3, 2, 1},
       {6, 5, 9, 8, 7, 1, 0, 4, 3, 2},
       {7, 6, 5, 9, 8, 2, 1, 0, 4, 3},
       {8, 7, 6, 5, 9, 3, 2, 1, 0, 4},
       {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}}};
  return d;
}
// Position-dependent permutation table.
/// @brief Return the immutable canonical Verhoeff lookup table.
inline const std::array<std::array<int, 10>, 8>& verhoeff_p() {
  /// @brief Canonical immutable Verhoeff lookup coefficients.
  static const std::array<std::array<int, 10>, 8> p = {
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
       {1, 5, 7, 6, 2, 8, 3, 0, 9, 4},
       {5, 8, 0, 3, 7, 9, 6, 1, 4, 2},
       {8, 9, 1, 6, 0, 4, 3, 5, 2, 7},
       {9, 4, 5, 3, 1, 2, 6, 8, 7, 0},
       {4, 2, 8, 6, 5, 7, 3, 9, 0, 1},
       {2, 7, 9, 3, 8, 0, 6, 4, 1, 5},
       {7, 0, 4, 6, 9, 1, 3, 2, 5, 8}}};
  return p;
}
// Multiplicative inverse in D5.
/// @brief Return the immutable canonical Verhoeff lookup table.
inline const std::array<int, 10>& verhoeff_inv() {
  /// @brief Canonical immutable Verhoeff lookup coefficients.
  static const std::array<int, 10> inv = {0, 4, 3, 2, 1, 5, 6, 7, 8, 9};
  return inv;
}
} // namespace detail
/// @brief One actual right-to-left Verhoeff table transition.
/// @par Example
/// @code
/// dm::algorithms::VerhoeffStep r{};
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerhoeffStep {
  std::size_t position{0}; ///< Actual right-to-left offset.
  int         digit{0};    ///< Actual input decimal digit.
  int         permuted{0}; ///< Actual permutation lookup result.
  int         before{0};   ///< Actual D5 accumulator before multiplication.
  int         after{0};    ///< Actual D5 accumulator after multiplication.
};
/// @brief Actual Verhoeff producer states and result.
/// @par Example
/// @code
/// auto r=dm::algorithms::verhoeff_trace("236");
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerhoeffWork {
  std::vector<VerhoeffStep> steps;          ///< Complete actual digit transitions.
  int                       residue{0};     ///< Actual final D5 accumulator.
  int                       check_digit{0}; ///< Inverse of actual residue, for payload generation.
  bool valid{false}; ///< Validation mode only: nonempty decimal input has residue0.
};
/// @brief Run the Verhoeff D5 table producer with actual transitions.
/// Leading zeros preserved; empty payload generates0, empty full number is invalid. O(length) work/storage. No Unicode parsing, punctuation stripping or shared state.
/// @param text ASCII decimal bytes, at most1000000.
/// @param validation False generates payload check digit with permutation offset1; true validates full text with offset0.
/// @return Actual residue, check digit and complete transitions.
/// @throws std::invalid_argument Nondecimal byte.
/// @throws std::length_error Input exceeds1000000 bytes.
/// @par Example
/// @code
/// auto r=dm::algorithms::verhoeff_trace("236"); //check_digit3
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VerhoeffWork verhoeff_trace(const std::string& text, bool validation = false) {
  if (text.size() > 1000000)
    throw std::length_error("Verhoeff size");
  for (char c : text)
    if (c < '0' || c > '9')
      throw std::invalid_argument("Verhoeff decimal text");
  VerhoeffWork r;
  int          c = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    int digit    = text[text.size() - 1 - i] - '0';
    int permuted = detail::verhoeff_p()[(i + (validation ? 0 : 1)) % 8][digit];
    int next     = detail::verhoeff_d()[c][permuted];
    r.steps.push_back({i, digit, permuted, c, next});
    c = next;
  }
  r.residue     = c;
  r.check_digit = detail::verhoeff_inv()[c];
  r.valid       = validation && !text.empty() && c == 0;
  return r;
}
/// @brief Generate the ASCII decimal payload check digit.
/// @param payload Decimal text, empty allowed.
/// @return Integer digit in[0,9].
/// @throws std::invalid_argument Nondecimal text.
/// @throws std::length_error Input exceeds1000000 bytes.
/// @par Example
/// @code
/// int digit=dm::algorithms::verhoeff_check_digit("236"); //3
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline int verhoeff_check_digit(const std::string& payload) {
  return verhoeff_trace(payload).check_digit;
}
/// @brief Validate a complete ASCII decimal Verhoeff number.
/// @param full Full number including final check digit.
/// @return False for empty, nondecimal text or nonzero residue.
/// @throws std::length_error Input exceeds1000000 bytes.
/// @par Example
/// @code
/// bool valid=dm::algorithms::verhoeff_validate("2363"); //true
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline bool verhoeff_validate(const std::string& full) {
  if (full.empty())
    return false;
  for (char c : full)
    if (c < '0' || c > '9')
      return false;
  return verhoeff_trace(full, true).valid;
}
/// @brief Actual exhaustive single-substitution and adjacent-swap detection.
/// @par Example
/// @code
/// auto r=dm::algorithms::verhoeff_detection("236");
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerhoeffDetection {
  std::string full;                     ///< Payload plus actual check digit.
  std::size_t substitutions{0};         ///< Actual distinct single-digit mutations tested.
  std::size_t substitutions_caught{0};  ///< Actual rejected mutations.
  std::size_t transpositions{0};        ///< Actual unequal adjacent swaps tested.
  std::size_t transpositions_caught{0}; ///< Actual rejected swaps.
};
/// @brief Enumerate and validate the actual original error-detection cohort.
/// O(length^2) validation work; equal adjacent digits are excluded because swapping them changes nothing. Detecting these classes does not provide cryptographic integrity.
/// @param payload ASCII decimal payload, at most10000 bytes.
/// @return Owned checked number and actual detection counts.
/// @throws std::invalid_argument Nondecimal payload.
/// @throws std::length_error Payload exceeds10000 bytes.
/// @par Example
/// @code
/// auto r=dm::algorithms::verhoeff_detection("236");
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VerhoeffDetection verhoeff_detection(const std::string& payload) {
  if (payload.size() > 10000)
    throw std::length_error("detection cohort");
  VerhoeffDetection r;
  r.full = payload + char('0' + verhoeff_check_digit(payload));
  for (std::size_t i = 0; i < r.full.size(); ++i) {
    for (char c = '0'; c <= '9'; ++c)
      if (c != r.full[i]) {
        auto changed = r.full;
        changed[i]   = c;
        ++r.substitutions;
        if (!verhoeff_validate(changed))
          ++r.substitutions_caught;
      }
    if (i + 1 < r.full.size() && r.full[i] != r.full[i + 1]) {
      auto changed = r.full;
      std::swap(changed[i], changed[i + 1]);
      ++r.transpositions;
      if (!verhoeff_validate(changed))
        ++r.transpositions_caught;
    }
  }
  return r;
}

} // namespace datamunge::algorithms
