#pragma once
/// @file rabin_karp.hpp
/// @brief Exact byte matching with actual rolling hashes and character-verification records.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// @brief String algorithms and their owned histories.
namespace datamunge::algorithms {

/// @brief Hash parameters and selected-work/history bounds.
/// @par Example
/// @code
/// const dm::algorithms::RabinKarpOptions x{};
/// std::print("{}\n",x.modulus); // 1000000007
/// @endcode
/// @see rabin_karp_run()
struct RabinKarpOptions {
  std::uint32_t modulus{
      1000000007};         ///< Modulus at least two; primality is not needed for exact results.
  std::uint32_t base{256}; ///< Positive base strictly below modulus.
  std::size_t   max_operations{
      10000000}; ///< Hash updates plus visited windows plus byte comparisons; zero disables.
  std::size_t max_trace_entries{1000000}; ///< Window records retained; zero disables.
};
/// @brief One actual candidate window, before advancing the rolling hash.
/// @par Example
/// @code
/// const dm::algorithms::RabinKarpWindow x{};
/// std::print("{}\n",x.matched); // false
/// @endcode
/// @see rabin_karp_run()
struct RabinKarpWindow {
  std::size_t   start{};          ///< Byte offset of the window.
  std::uint64_t hash{};           ///< Actual modular rolling hash of this window.
  bool          candidate{};      ///< Whether the window hash equals the pattern hash.
  bool          matched{};        ///< Whether exact byte verification accepted the candidate.
  std::size_t   compared_bytes{}; ///< Exact scalar byte comparisons, including a failing byte.
};
/// @brief Owned exact matches and actual selected work.
/// @par Example
/// @code
/// const dm::algorithms::RabinKarpResult x{};
/// std::print("{}\n",x.visited_windows); // 0
/// @endcode
/// @see rabin_karp_run()
struct RabinKarpResult {
  std::vector<std::size_t> matches; ///< Ascending byte offsets, including overlapping occurrences.
  std::vector<RabinKarpWindow> windows; ///< Actual records, populated only by the trace.
  std::uint64_t pattern_hash{};         ///< Actual Horner hash; zero for an empty/too-long pattern.
  std::uint64_t high{}; ///< Actual base^(m-1) modulo modulus, zero when no hashing occurs.
  std::size_t
      visited_windows{};        ///< Windows actually considered, including empty-pattern positions.
  std::size_t hash_updates{};   ///< Initial Horner/power updates and complete rolling updates.
  std::size_t compared_bytes{}; ///< Actual exact-verification byte comparisons.
  std::size_t operations{};     ///< Sum of windows, hash updates and byte comparisons.
};
/// @brief Find exact byte matches using the shared checked rolling-hash producer.
/// Hash equality is always followed by scalar unsigned-byte verification; even composite/small moduli and collisions cannot produce false matches. Embedded NUL/high bytes are supported. Empty pattern matches all n+1 positions; longer pattern visits no windows. Typical rare-collision work O(n+m), worst O(n*m), plus owned matches. Caller strings are unchanged. Default hash is the historical base256/mod1000000007.
/// @param text Arbitrary text bytes.
/// @param pattern Arbitrary pattern bytes.
/// @param options Validated hash parameters and selected-work policy.
/// @return Exact byte offsets and actual hashes/counters.
/// @throws std::invalid_argument Modulus below2 or base outside1..modulus-1.
/// @throws std::length_error Selected work or representable match capacity exceeded.
/// @par Example
/// @code
/// auto r = dm::algorithms::rabin_karp_run("ababa","aba");
/// std::print("{} {}\n",r.matches[0],r.matches[1]); // 0 2
/// @endcode
/// @see rabin_karp_trace(), rabin_karp_search()
RabinKarpResult rabin_karp_run(
    const std::string&      text,
    const std::string&      pattern,
    const RabinKarpOptions& options = {});
/// @brief Capture every actual visited window and verification decision.
/// Same producer as rabin_karp_run(); a record is admitted before the window is visited. One record per empty-pattern position. Retains O(n) records, bounded by policy; zero disables the cap.
/// @param text Arbitrary text bytes.
/// @param pattern Arbitrary pattern bytes.
/// @param options Hash, selected-work and window-history policies.
/// @return Actual window hashes, candidates, exact decisions, matches and counters.
/// @throws std::invalid_argument Invalid hash parameters.
/// @throws std::length_error Work, history or match capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::RabinKarpOptions o; o.modulus=2; o.base=1;
/// auto r=dm::algorithms::rabin_karp_trace("ca","a",o);
/// std::print("{} {}\n",r.windows[0].candidate,r.windows[0].matched); // true false
/// @endcode
/// @see rabin_karp_run()
RabinKarpResult rabin_karp_trace(
    const std::string&      text,
    const std::string&      pattern,
    const RabinKarpOptions& options = {});

/// @brief The Rabin-Karp string-matching algorithm (Karp & Rabin 1987). Finds every position in
///        @p text at which @p pattern occurs, using a *polynomial rolling hash*. Each length-m
///        window of the text is treated as an m-digit number in some base b, taken modulo a large
///        prime p; sliding the window one character to the right updates its hash in O(1) --
///        subtract the leading digit's contribution, multiply by the base, add the incoming digit
///        -- rather than re-reading the whole window. When a window's hash equals the pattern's,
///        that only *suggests* a match (two distinct strings can collide modulo p), so the actual
///        characters are compared to confirm it; the returned positions are therefore always
///        exact. Expected running time is O(n + m) when collisions are rare, degrading to O(n*m)
///        in the worst case (adversarial or unlucky inputs that force a verification at every
///        window). Overlapping occurrences are all reported.
///
/// @param text the string to search within.
/// @param pattern the string to search for.
/// @return the sorted list of start indices at which @p pattern occurs in @p text. By convention
///         an empty @p pattern matches at every position, yielding {0, 1, ..., text.size()}
///         (mirroring std::string::find("")). If @p pattern is longer than @p text, the result is
///         empty.
/// @par Example
/// @code
/// auto p=dm::algorithms::rabin_karp_search("aaa","aa");
/// std::print("{} {}\n",p[0],p[1]); // 0 1
/// @endcode
/// @see rabin_karp_run()
std::vector<std::size_t> rabin_karp_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
