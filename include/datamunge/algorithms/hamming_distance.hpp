#pragma once
/// @file hamming_distance.hpp
/// @brief Inspectable byte/bit Hamming comparisons, finite-code geometry and Hamming shells.
#include <datamunge/algorithms/string_metrics.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Counts differing bits in two packed unsigned 64-bit words using XOR/popcount.
/// The comparison includes all 64 positions, including leading zeros. Unlike the string
/// hamming_distance(), the symbols here are individual bits, not bytes. Inputs are
/// borrowed values, there is no mutation/shared state, and work/storage are O(1).
/// @param a First packed word.
/// @param b Second packed word.
/// @return Population count of a XOR b, in [0,64].
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hamming_bit_distance(0,255)); // 8
/// @endcode
/// @par References
/// - R. W. Hamming (1950), Error Detecting and Error Correcting Codes.
///   https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_distance(), hamming_comparison()
std::size_t hamming_bit_distance(std::uint64_t a,std::uint64_t b);
/// @brief Per-byte positional comparisons and cumulative mismatch counts, without realignment.
/// Strings may contain NUL and arbitrary bytes; byte values are reported as unsigned
/// numbers, not interpreted as UTF-8 characters. Empty pairs give a zero-row typed table.
/// This and the other helpers own their results, leave inputs unchanged and have no
/// shared mutable state. Work and output storage O(n).
/// @param a First byte string.
/// @param b Equal-length second byte string.
/// @return position/a_byte/b_byte/mismatch/cumulative_distance; positions are zero-based.
/// @throws std::invalid_argument Lengths differ.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hamming_comparison("karolin","kathrin");
/// std::print("{:.0f}\n",t.double_at("cumulative_distance",6)); // 3
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_distance(), plot_hamming_comparison()
dstruct::DataFrame hamming_comparison(const std::string& a,const std::string& b);
/// @brief Aligned byte labels and agreement markers for a short Hamming comparison.
/// Printable ASCII bytes display literally; other bytes use hexadecimal labels. Work
/// and output size O(n). Call save_svg() to save a local image rather than printing XML.
/// @param a First string, nonempty and no longer than 64 bytes for legibility.
/// @param b Equal-length second string.
/// @return Owned editable plot; mismatch markers/second-row labels are red.
/// @throws std::invalid_argument Unequal lengths, empty strings or more than 64 columns.
/// @par Example
/// @code
/// dm::algorithms::plot_hamming_comparison("1011101","1001001").save_svg("hamming.svg");
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_comparison()
graphics::RPlot plot_hamming_comparison(const std::string& a,const std::string& b);
/// @brief Symmetric byte-Hamming distances for fixed-width words, in caller order.
/// Work O(m^2*n), result space O(m^2). Duplicates are allowed; an empty word list gives
/// a zero-by-zero matrix. Empty words of a common width zero are also allowed.
/// @param words Equal-byte-length strings.
/// @return Owned m-by-m distances with a zero diagonal.
/// @throws std::invalid_argument Widths differ.
/// @par Example
/// @code
/// const auto d=dm::algorithms::hamming_distance_matrix({"000","111"});
/// std::print("{:.0f}\n",d(0,1)); // 3
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_code_summary(), hamming_nearest()
linalg::DenseMatrix<double> hamming_distance_matrix(const std::vector<std::string>& words);
/// @brief Minimum separation and universal detection/correction radii of a finite code.
/// Enumerates unordered pairs in O(m^2*n) time with O(m) validation storage; does not
/// run a decoder. Detectable_errors=d_min-1 assumes code-membership rejection; a hard
/// nearest-codeword decoder can miscorrect outside its guaranteed correction radius.
/// @param codebook At least two distinct words of equal byte width.
/// @return One-row words/width/minimum_distance/detectable_errors/correctable_errors.
/// @throws std::invalid_argument Too few words, duplicates or unequal widths.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hamming_code_summary({"000","111"});
/// std::print("{:.0f}\n",t.double_at("correctable_errors",0)); // 1
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_nearest(), hamming_shells()
dstruct::DataFrame hamming_code_summary(const std::vector<std::string>& codebook);
/// @brief Every codeword's distance to a query, retaining all nearest-neighbour ties.
/// No word is silently selected and no decoding guarantee is inferred from one query.
/// Work/storage O(m*n)/O(m), with original codebook ordering preserved.
/// @param codebook Nonempty distinct equal-width byte words.
/// @param query Exactly the codebook byte width, including embedded NULs if present.
/// @return codeword/distance/nearest table; nearest is one for every tied minimum.
/// @throws std::invalid_argument Empty/duplicate/inconsistent codebook or query width mismatch.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hamming_nearest({"00","11"},"01");
/// std::print("{:.0f} {:.0f}\n",t.double_at("nearest",0),t.double_at("nearest",1)); // 1 1
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_code_summary()
dstruct::DataFrame hamming_nearest(const std::vector<std::string>& codebook,const std::string& query);
/// @brief Sizes of q-ary Hamming shells/balls and uniform-word probabilities.
/// Shell k has binomial(n,k)*(q-1)^k words; the ball is the cumulative shell sum.
/// Counts are numerical double values (not exact integers beyond 2^53); positive
/// probabilities may underflow to zero. No enumeration of q^n words is needed.
/// Work/output storage O(n); overflow is rejected before allocating the table.
/// @param length Word width n, including zero.
/// @param alphabet_size Integer q>=2; default two (bits).
/// @return distance/shell_size/ball_volume/probability/cumulative_probability, n+1 rows.
/// @throws std::invalid_argument Alphabet smaller than two.
/// @throws std::overflow_error q^n cannot be represented as a finite double.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hamming_shells(3);
/// std::print("{:.0f}\n",t.double_at("ball_volume",1)); // 4
/// @endcode
/// @par References
/// - Hamming (1950). https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see hamming_code_summary()
dstruct::DataFrame hamming_shells(std::size_t length,std::size_t alphabet_size=2);
} // namespace datamunge::algorithms
