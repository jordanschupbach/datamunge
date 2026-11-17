#pragma once
/// @file luhn.hpp
/// @brief Checked ASCII-decimal Luhn checksum with overflow-free residues and actual right-to-left evidence.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Whether the supplied digits are a full checked string or a payload awaiting its appended digit.
/// @par Example
/// @code
/// auto mode=dm::algorithms::LuhnMode::Payload;
/// std::print("{}\n",int(mode)); // 1
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
enum class LuhnMode {
  Full,   ///< Rightmost supplied digit is not doubled; empty full input is invalid.
  Payload ///< Rightmost supplied payload digit is doubled; empty payload generates check digit0.
};
/// @brief Actual bounded-residue checksum result from one producing pass.
/// @par Example
/// @code
/// auto a=dm::algorithms::luhn_analyze("79927398713");
/// std::print("{} {}\n",a.valid,a.remainder); // true 0
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
struct LuhnAnalysis {
  LuhnMode    mode{LuhnMode::Full}; ///< Actual selected full/payload parity convention.
  std::size_t digits{0};  ///< Actual complete ASCII-decimal input length, preserving leading zeros.
  std::size_t doubled{0}; ///< Actual doubled-digit positions visited.
  std::size_t reductions{0}; ///< Actual doubled contributions reduced by9.
  int  remainder{0}; ///< Actual final weighted sum modulo10, stored without unbounded accumulation.
  int  check_digit{-1}; ///< Payload-only complement in0..9; minus1 in full mode.
  bool valid{false};    ///< Full-mode only: nonempty input and actual final residue0.
};
/// @brief Actual producing right-to-left digit transformation and modular accumulation.
/// @par Example
/// @code
/// auto t=dm::algorithms::luhn_trace("125");
/// std::print("{}\n",t.steps.front().contribution); // 5
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
struct LuhnStep {
  std::size_t index;   ///< Original zero-based digit index, visited in descending order.
  int         digit;   ///< Actual ASCII digit value0..9.
  bool        doubled; ///< Actual parity decision at this index.
  int         raw;     ///< Actual digit or doubled digit before optional9 subtraction, at most18.
  bool        reduced; ///< Whether raw exceeded9 and had9 subtracted.
  int         contribution; ///< Actual transformed contribution0..9.
  int         before;       ///< Actual running residue before this contribution.
  int         after;        ///< Actual running residue after this contribution modulo10.
};
/// @brief Owned original digit string and every actual producing digit step.
/// @par Example
/// @code
/// auto t=dm::algorithms::luhn_trace("12");
/// std::print("{}\n",t.steps.front().index); // 1
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
struct LuhnTrace {
  std::string           input;    ///< Independently owned original decimal bytes.
  std::vector<LuhnStep> steps;    ///< Actual right-to-left transformations, without replay.
  LuhnAnalysis          analysis; ///< Actual same-run bounded residue and mode result.
  std::size_t           captured_entries{0}; ///< Actual input bytes plus digit records, exactly2*n.
};
/// @brief Generate the decimal check digit for a payload using one shared producing residue pass.
/// @param payload ASCII decimal digits only, preserving leading zeros; empty is permitted and yields0.
/// @return The unique appended check digit in0..9 giving a nonempty full string with residue0.
/// @throws std::invalid_argument For any nondecimal byte, including signs, spaces, NUL and non-ASCII numerals.
/// @note O(n) time and constant auxiliary storage. This is a checksum, not issued-identifier validation.
/// @code
/// std::print("{}\n",dm::algorithms::luhn_check_digit("7992739871")); // 3
/// @endcode
int luhn_check_digit(const std::string& payload);
/// @brief Check a nonempty full decimal string using the same actual parity/residue producer.
/// @param full ASCII decimal digits with a final check digit; leading zeros retained.
/// @return True for nonempty residue0 input; false for empty or well-formed checksum failures.
/// @throws std::invalid_argument For any nondecimal byte, before numeric processing.
/// @note Empty full input now returns false; malformed input is rejected rather than treated as digits.
/// Length/prefix/business rules and authenticity are outside this arithmetic checksum.
/// @code
/// std::print("{}\n",dm::algorithms::luhn_validate("79927398713")); // true
/// @endcode
bool luhn_validate(const std::string& full);
/// @brief Evaluate actual selected residue/transform counts, validating the complete byte input before processing.
/// @param digits Original ASCII digits, possibly empty according to the explicit mode contract.
/// @param mode Full or Payload; any other enum value is invalid.
/// @return Actual constant-storage checksum result; full and payload meanings are kept separate.
/// @throws std::invalid_argument For malformed complete decimal input or mode.
/// @code
/// auto a=dm::algorithms::luhn_analyze("12",dm::algorithms::LuhnMode::Payload);
/// std::print("{}\n",a.check_digit); // 5
/// @endcode
LuhnAnalysis luhn_analyze(const std::string& digits, LuhnMode mode = LuhnMode::Full);
/// @brief Capture every real right-to-left digit operation from the shared producing pass.
/// @param digits Original complete ASCII-decimal input.
/// @param mode Explicit full/payload parity convention.
/// @param max_entries Optional logical capture quota, default one million; zero disables it.
/// @return Owned original bytes, actual steps and same-run result, without checksum replay.
/// @throws std::invalid_argument On invalid complete input/mode before capture.
/// @throws std::length_error Before copies if2*n logical bytes/records exceed representable geometry or cap.
/// @note Excludes fields, capacities, analysis/runtime copies and bytes of physical storage.
/// @code
/// auto t=dm::algorithms::luhn_trace("12");
/// std::print("{}\n",t.captured_entries); // 4
/// @endcode
LuhnTrace luhn_trace(
    const std::string& digits,
    LuhnMode           mode        = LuhnMode::Full,
    std::size_t        max_entries = 1000000);
/// @brief Actual generated error kind, excluding unchanged substitutions and equal-adjacent swaps.
/// @par Example
/// @code
/// auto kind=dm::algorithms::LuhnErrorKind::AdjacentSwap;
/// std::print("{}\n",int(kind)); // 1
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
enum class LuhnErrorKind {
  Substitution, ///< Replace exactly one original digit with a distinct digit.
  AdjacentSwap  ///< Exchange two distinct adjacent digits, preserving string length.
};
/// @brief One actual generated erroneous string and its actual checksum observation.
/// @par Example
/// @code
/// auto s=dm::algorithms::luhn_error_study("09");
/// std::print("{}\n",s.cases.front().detected); // true
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
struct LuhnErrorCase {
  LuhnErrorKind kind;      ///< Actual generated error kind.
  std::size_t   index;     ///< Actual substituted index or left swapped index.
  std::string   changed;   ///< Actual generated erroneous decimal string, independently owned.
  bool          detected;  ///< Actual complete full-string checksum rejection.
  int           remainder; ///< Actual resulting weighted residue0..9.
};
/// @brief Owned original payload/full string and every actual exhaustive single-error outcome.
/// @par Example
/// @code
/// auto s=dm::algorithms::luhn_error_study("09");
/// std::print("{}\n",s.substitutions==s.substitutions_detected); // true
/// @endcode
/// @par References
/// - H. P. Luhn, Computer for Verifying Numbers, US2950048A, https://patents.google.com/patent/US2950048A/en; bounded-residue ASCII-decimal implementation here.
/// @see luhn_analyze(), luhn_trace(), luhn_error_study()
struct LuhnErrorStudy {
  std::string payload; ///< Original ASCII payload, independently owned.
  std::string full;    ///< Actual payload plus its produced check digit.
  std::vector<LuhnErrorCase>
              cases; ///< All9*n substitutions and distinct-adjacent swaps of the full string.
  std::size_t substitutions{0};          ///< Actual generated substitution count.
  std::size_t substitutions_detected{0}; ///< Actual rejected substitutions.
  std::size_t swaps{0};                  ///< Actual generated unequal-adjacent swap count.
  std::size_t swaps_detected{0};         ///< Actual rejected unequal-adjacent swaps.
  std::size_t captured_entries{
      0}; ///< Payload/full bytes plus one record and changed-string bytes per error.
};
/// @brief Generate a valid full string and evaluate every prescribed single-digit and unequal-adjacent error.
/// @param payload Original complete ASCII decimal payload, with empty payload allowed.
/// @param max_entries Optional complete logical capture cap, default one million; zero disables it.
/// @return Owned full string and actual independently checked error cases, without report-local mutation logic.
/// @throws std::invalid_argument For malformed complete payload.
/// @throws std::length_error Before error-string copies if complete logical capture geometry exceeds its cap.
/// @note Counts payload/full bytes and each error record plus its changed bytes; fields/capacities/runtime copies excluded.
/// O(n*n) checksum work/output bytes for full-string length n; this is a bounded exhaustive illustration, not a Monte Carlo estimate.
/// @code
/// auto s=dm::algorithms::luhn_error_study("1230904560907");
/// std::print("{}\n",s.substitutions==s.substitutions_detected); // true
/// @endcode
LuhnErrorStudy luhn_error_study(const std::string& payload, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
