/// @file move_to_front.cpp
/// @brief Canonical checked alphabet movement and actual rank diagnostics.
#include <cmath>
#include <datamunge/algorithms/move_to_front.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private shared encoder/decoder producer and admission helpers.
namespace {
/// @brief Validate every inverse rank before any indexing.
/// @param indices Complete rank stream.
/// @throws std::invalid_argument If any rank lies outside0..255.
void valid(const std::vector<int>& indices) {
  for (int rank : indices)
    if (rank < 0 || rank > 255)
      throw std::invalid_argument("MTF: rank must lie in0..255");
}
/// @brief Add actual work counts with overflow checking.
/// @param count Counter to update.
/// @param amount Actual new work units.
/// @throws std::length_error If addition is unrepresentable.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("MTF: work count overflow");
  count += amount;
}
/// @brief Preflight complete3n captured units before copying/transforming.
/// @param n Complete byte/rank count.
/// @param cap Optional quota, zero disables.
/// @return Checked complete3n units.
/// @throws std::length_error If geometry or quota exceeds limits.
std::size_t quota(std::size_t n, std::size_t cap) {
  if (n > std::numeric_limits<std::size_t>::max() / 3)
    throw std::length_error("MTF: capture geometry overflow");
  auto count = 3 * n;
  if (cap && count > cap)
    throw std::length_error("MTF: capture quota exceeded");
  return count;
}
/// @brief Execute actual encoder or inverse movement using one alphabet kernel.
/// @param data Optional original bytes; nonnull means encode.
/// @param indices Optional validated ranks; nonnull means decode.
/// @param steps Optional actual-state sink.
/// @return Actual output and work.
MTFAnalysis
run(const std::string* data, const std::vector<int>* indices, std::vector<MTFStep>* steps) {
  std::array<unsigned char, 256> alphabet;
  for (int i = 0; i < 256; ++i)
    alphabet[i] = static_cast<unsigned char>(i);
  const auto  n = data ? data->size() : indices->size();
  MTFAnalysis result;
  if (data)
    result.indices.reserve(n);
  else
    result.bytes.reserve(n);
  if (steps)
    steps->reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    int           rank = 0;
    unsigned char byte;
    std::size_t   comparisons = 0;
    if (data) {
      byte = static_cast<unsigned char>((*data)[i]);
      while (true) {
        ++comparisons;
        if (alphabet[rank] == byte)
          break;
        ++rank;
      }
      result.indices.push_back(rank);
    } else {
      rank = (*indices)[i];
      byte = alphabet[rank];
      result.bytes.push_back(static_cast<char>(byte));
    }
    auto before = alphabet[0];
    add(result.comparisons, comparisons);
    add(result.shifts, rank);
    for (int j = rank; j > 0; --j)
      alphabet[j] = alphabet[j - 1];
    alphabet[0] = byte;
    if (steps)
      steps->push_back({i, byte, rank, before, comparisons, std::size_t(rank)});
  }
  return result;
}
} // namespace
std::vector<int> mtf_encode(const std::string& data) {
  return run(&data, nullptr, nullptr).indices;
}
std::string mtf_decode(const std::vector<int>& indices) {
  valid(indices);
  return run(nullptr, &indices, nullptr).bytes;
}
MTFAnalysis mtf_encode_analyze(const std::string& data) {
  return run(&data, nullptr, nullptr);
}
MTFAnalysis mtf_decode_analyze(const std::vector<int>& indices) {
  valid(indices);
  return run(nullptr, &indices, nullptr);
}
MTFTrace mtf_encode_trace(const std::string& data, std::size_t cap) {
  MTFTrace t;
  t.captured_entries = quota(data.size(), cap);
  t.input_bytes      = data;
  t.result           = run(&data, nullptr, &t.steps);
  return t;
}
MTFTrace mtf_decode_trace(const std::vector<int>& indices, std::size_t cap) {
  valid(indices);
  MTFTrace t;
  t.decoding         = true;
  t.captured_entries = quota(indices.size(), cap);
  t.input_indices    = indices;
  t.result           = run(nullptr, &indices, &t.steps);
  return t;
}
MTFIndexSummary mtf_index_summary(const std::vector<int>& indices) {
  valid(indices);
  MTFIndexSummary s;
  s.count           = indices.size();
  long double total = 0;
  for (auto rank : indices) {
    ++s.histogram[rank];
    total += rank;
  }
  s.zeros = s.histogram[0];
  for (int i = 1; i <= 3; ++i)
    s.one_to_three += s.histogram[i];
  s.four_or_more = s.count - s.zeros - s.one_to_three;
  if (!s.count)
    return s;
  s.mean                 = double(total / s.count);
  std::size_t cumulative = 0;
  int         low = -1, high = 0;
  const auto  first = (s.count - 1) / 2, last = s.count / 2;
  for (int i = 0; i < 256; ++i) {
    const auto count = s.histogram[i];
    cumulative += count;
    if (low < 0 && cumulative > first)
      low = i;
    if (cumulative > last) {
      high = i;
      break;
    }
  }
  s.median = (low + high) / 2.;
  for (auto count : s.histogram)
    if (count) {
      double probability = double(count) / s.count;
      s.entropy_bits -= probability * std::log2(probability);
    }
  return s;
}
std::string mtf_demo_text() {
  return "the quick brown fox the quick brown fox the quick brown fox the quick brown fox";
}
} // namespace datamunge::algorithms
