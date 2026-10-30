/// @file hamming_distance.cpp
/// @brief Byte comparisons, packed population counts and finite Hamming-space diagnostics.
#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/hamming_distance.hpp>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private byte formatting and codebook validation.
namespace {
/// @brief Checks fixed-width, unique codewords without interpreting their bytes.
/// @param words Nonempty codebook.
/// @param minimum Minimum number of distinct words.
void validate_codebook(const std::vector<std::string>& words, std::size_t minimum) {
  if (words.size() < minimum)
    throw std::invalid_argument("hamming: too few codewords");
  std::set<std::string> seen;
  for (const auto& word : words)
    if (word.size() != words.front().size() || !seen.insert(word).second)
      throw std::invalid_argument("hamming: distinct equal-width codewords required");
}
/// @brief Human-readable label for one unsigned byte, never emitting raw control bytes.
/// @param byte Byte value from zero through 255.
/// @return Printable ASCII or a two-digit 0x-prefixed hexadecimal label.
std::string byte_label(unsigned char byte) {
  if (byte >= 33 && byte <= 126)
    return std::string(1, static_cast<char>(byte));
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
  return out.str();
}
} // namespace
std::size_t hamming_bit_distance(std::uint64_t a, std::uint64_t b) {
  return std::popcount(a ^ b);
}
dstruct::DataFrame hamming_comparison(const std::string& a, const std::string& b) {
  (void)hamming_distance(a, b);
  std::vector<double> position, aa, bb, mismatch, cumulative;
  std::size_t         count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool different = a[i] != b[i];
    count += different;
    position.push_back(i);
    aa.push_back(static_cast<unsigned char>(a[i]));
    bb.push_back(static_cast<unsigned char>(b[i]));
    mismatch.push_back(different);
    cumulative.push_back(count);
  }
  dstruct::DataFrame out;
  out.add_column("position", position);
  out.add_column("a_byte", aa);
  out.add_column("b_byte", bb);
  out.add_column("mismatch", mismatch);
  out.add_column("cumulative_distance", cumulative);
  return out;
}
graphics::RPlot plot_hamming_comparison(const std::string& a, const std::string& b) {
  if (a.empty() || a.size() > 64)
    throw std::invalid_argument("plot_hamming_comparison: use 1..64 byte positions");
  const auto          table = hamming_comparison(a, b);
  auto                p     = graphics::RPlot::create();
  std::vector<double> same, different;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool mismatch = a[i] != b[i];
    (mismatch ? different : same).push_back(i);
    p.text(i, 1.35, byte_label(static_cast<unsigned char>(a[i])), {31, 41, 55}, 18);
    p.text(
        i,
        .65,
        byte_label(static_cast<unsigned char>(b[i])),
        mismatch ? graphics::RGB{220, 38, 38} : graphics::RGB{31, 41, 55},
        18);
  }
  if (!same.empty())
    p.points(same, std::vector<double>(same.size(), 1), "agree", {22, 163, 74}, 7);
  if (!different.empty())
    p.points(different, std::vector<double>(different.size(), 1), "different", {220, 38, 38}, 8);
  return p
      .title("Hamming distance: " + std::to_string(hamming_distance(a, b)) + " differing positions")
      .x_label("zero-based byte position")
      .y_label("")
      .x_limits(-.6, double(a.size()) - .4)
      .y_limits(.25, 1.95);
}
linalg::DenseMatrix<double> hamming_distance_matrix(const std::vector<std::string>& words) {
  if (!words.empty())
    for (const auto& word : words)
      if (word.size() != words.front().size())
        throw std::invalid_argument("hamming_distance_matrix: unequal widths");
  linalg::DenseMatrix<double> out(words.size(), words.size(), 0);
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t j = i + 1; j < words.size(); ++j)
      out(i, j) = out(j, i) = hamming_distance(words[i], words[j]);
  return out;
}
dstruct::DataFrame hamming_code_summary(const std::vector<std::string>& words) {
  validate_codebook(words, 2);
  auto minimum = words.front().size();
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t j = i + 1; j < words.size(); ++j)
      minimum = std::min(minimum, hamming_distance(words[i], words[j]));
  dstruct::DataFrame out;
  out.add_column("words", std::vector<double>{double(words.size())});
  out.add_column("width", std::vector<double>{double(words.front().size())});
  out.add_column("minimum_distance", std::vector<double>{double(minimum)});
  out.add_column("detectable_errors", std::vector<double>{double(minimum - 1)});
  out.add_column("correctable_errors", std::vector<double>{double((minimum - 1) / 2)});
  return out;
}
dstruct::DataFrame hamming_nearest(
    const std::vector<std::string>& words,
    const std::string&              query) {
  validate_codebook(words, 1);
  std::vector<double> distance, nearest;
  auto                minimum = query.size();
  for (const auto& word : words) {
    const auto d = hamming_distance(word, query);
    minimum      = std::min(minimum, d);
    distance.push_back(d);
  }
  for (double d : distance)
    nearest.push_back(d == minimum);
  dstruct::DataFrame out;
  out.add_column("codeword", words);
  out.add_column("distance", distance);
  out.add_column("nearest", nearest);
  return out;
}
dstruct::DataFrame hamming_shells(std::size_t n, std::size_t q) {
  if (q < 2)
    throw std::invalid_argument("hamming_shells: alphabet_size must be at least two");
  const long double log_total = static_cast<long double>(n) * std::log(static_cast<long double>(q));
  if (log_total > std::log(static_cast<long double>(std::numeric_limits<double>::max())))
    throw std::overflow_error("hamming_shells: word-space size exceeds double range");
  const long double   total = std::pow(static_cast<long double>(q), n);
  long double         shell = 1, volume = 0;
  std::vector<double> distance, sizes, volumes, probability, cdf;
  for (std::size_t k = 0; k <= n; ++k) {
    volume += shell;
    distance.push_back(k);
    sizes.push_back(static_cast<double>(shell));
    volumes.push_back(static_cast<double>(volume));
    probability.push_back(static_cast<double>(shell / total));
    cdf.push_back(static_cast<double>(std::min(1.L, volume / total)));
    if (k < n)
      shell = shell * (n - k) / (k + 1) * (q - 1);
  }
  dstruct::DataFrame out;
  out.add_column("distance", distance);
  out.add_column("shell_size", sizes);
  out.add_column("ball_volume", volumes);
  out.add_column("probability", probability);
  out.add_column("cumulative_probability", cdf);
  return out;
}
} // namespace datamunge::algorithms
