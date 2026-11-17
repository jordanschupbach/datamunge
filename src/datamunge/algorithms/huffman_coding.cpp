/// @file huffman_coding.cpp
/// @brief Shared exact-integer Huffman heap, actual merges and fully checked byte codecs.
#include <algorithm>
#include <array>
#include <compare>
#include <cstdlib>
#include <datamunge/algorithms/huffman_coding.hpp>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private exact arithmetic, admission and byte projection helpers.
namespace huffman_detail {
/// @brief Private exact256-bit nonnegative weights/costs without an external arithmetic dependency.
/// All supported size_t/int widths are at most64 bits. A vector's uint64 counts
/// sum below2^128; even summing every possible parent costs below2^192. Thus
/// eight32-bit limbs cover the entire public input domain, not a narrowed sum cap.
struct Weight {
  std::array<std::uint32_t, 8> words{}; ///< Little-endian base2^32 limbs, exact capacity256 bits.

  /// @brief Construct an exact nonnegative uint64 value.
  /// @param value Initial value, default0.
  Weight(std::uint64_t value = 0) {
    words[0] = static_cast<std::uint32_t>(value);
    words[1] = static_cast<std::uint32_t>(value >> 32);
  }
  /// @brief Test whether every limb is zero.
  /// @return True exactly for zero; no mutation/allocation.
  bool zero() const {
    for (auto word : words)
      if (word)
        return false;
    return true;
  }
  /// @brief Add exact nonnegative limbs with carry.
  /// @param other Borrowed addend, unchanged.
  /// @return This updated value.
  /// @throws std::length_error On private256-bit overflow, unreachable for the public input bounds.
  Weight& operator+=(const Weight& other) {
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint64_t sum = std::uint64_t(words[i]) + other.words[i] + carry;
      words[i]                = static_cast<std::uint32_t>(sum);
      carry                   = sum >> 32;
    }
    if (carry)
      throw std::length_error("Huffman exact accumulator capacity");
    return *this;
  }
  /// @brief Return the exact sum of two nonnegative values.
  /// @param a Owned first addend.
  /// @param b Borrowed second addend.
  /// @return Exact sum.
  /// @throws std::length_error On private capacity overflow.
  friend Weight operator+(Weight a, const Weight& b) {
    a += b;
    return a;
  }
  /// @brief Compare exact values from most significant limb downward.
  /// @param a First borrowed value.
  /// @param b Second borrowed value.
  /// @return Exact total ordering, with no allocation/mutation.
  friend std::strong_ordering operator<=>(const Weight& a, const Weight& b) {
    for (std::size_t i = 8; i-- > 0;)
      if (a.words[i] != b.words[i])
        return a.words[i] < b.words[i] ? std::strong_ordering::less : std::strong_ordering::greater;
    return std::strong_ordering::equal;
  }
  /// @brief Compare all exact limbs for equality.
  /// @param other Borrowed comparison value.
  /// @return Whether both nonnegative values are equal.
  bool operator==(const Weight& other) const = default;
  /// @brief Multiply by one base2^32 digit with exact carry.
  /// @param factor Nonnegative32-bit multiplier.
  /// @return Exact product.
  /// @throws std::length_error On private capacity overflow.
  Weight times32(std::uint32_t factor) const {
    Weight        out;
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const std::uint64_t product = std::uint64_t(words[i]) * factor + carry;
      out.words[i]                = static_cast<std::uint32_t>(product);
      carry                       = product >> 32;
    }
    if (carry)
      throw std::length_error("Huffman exact product capacity");
    return out;
  }
  /// @brief Multiply by a complete uint64 value using two base2^32 digits.
  /// @param a Borrowed exact multiplicand.
  /// @param factor Nonnegative multiplier.
  /// @return Exact product.
  /// @throws std::length_error On private capacity overflow.
  friend Weight operator*(const Weight& a, std::uint64_t factor) {
    auto low  = a.times32(static_cast<std::uint32_t>(factor));
    auto high = a.times32(static_cast<std::uint32_t>(factor >> 32));
    if (high.words[7])
      throw std::length_error("Huffman exact shifted product capacity");
    for (std::size_t i = 7; i > 0; --i)
      high.words[i] = high.words[i - 1];
    high.words[0] = 0;
    low += high;
    return low;
  }
  /// @brief Convert to owned exact decimal digits without locale-dependent formatting.
  /// @return Exact unsigned decimal text, including"0".
  /// @throws std::bad_alloc On string allocation failure.
  std::string text() const {
    if (zero())
      return "0";
    auto        value = *this;
    std::string out;
    while (!value.zero()) {
      std::uint64_t remainder = 0;
      for (std::size_t i = 8; i-- > 0;) {
        const auto current = (remainder << 32) | value.words[i];
        value.words[i]     = static_cast<std::uint32_t>(current / 10);
        remainder          = current % 10;
      }
      out.push_back(static_cast<char>('0' + remainder));
    }
    std::reverse(out.begin(), out.end());
    return out;
  }
  /// @brief Convert the exact decimal value to finite double for a mean calculation.
  /// @return Rounded double value; every256-bit nonnegative integer fits double's exponent range.
  /// @throws std::bad_alloc On temporary decimal allocation failure.
  double as_double() const {
    const auto digits = text();
    return std::strtod(digits.c_str(), nullptr);
  }
};
static_assert(
    std::numeric_limits<std::size_t>::digits <= 64 && std::numeric_limits<int>::digits <= 63);
/// @brief Heap item ordered by exact weight, then compact node creation ID.
using Item = std::pair<Weight, std::size_t>;
/// @brief Add logical counts with representation checking.
/// @param a Existing units.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Huffman count overflow");
  return a + b;
}
/// @brief Admit complete logical units against an optional policy.
/// @param value Complete required units.
/// @param cap Maximum units, zero disabling policy.
/// @throws std::length_error On policy excess.
void admit(std::size_t value, std::size_t cap) {
  if (cap && value > cap)
    throw std::length_error("Huffman policy limit");
}
/// @brief Execute one exact-weight heap and integer-length traversal, optionally capturing actual pops.
/// @param counts Borrowed complete original-index counts.
/// @param options Input and whole merge policies.
/// @param records Optional owned merge destination.
/// @param input_entries Complete borrowed logical input size (source bytes or count positions).
/// @return Owned actual topology, lengths, exact costs and selected visits.
/// @throws std::length_error On input/merge/count/container/int-depth limits.
/// @throws std::bad_alloc On allocation failure.
HuffmanCountsRun execute(
    const std::vector<std::uint64_t>& counts,
    HuffmanOptions                    options,
    std::vector<HuffmanMerge>*        records,
    std::size_t                       input_entries) {
  admit(input_entries, options.max_input_entries);
  HuffmanCountsRun run;
  Weight           total = 0;
  for (auto count : counts) {
    total += count;
    run.active_symbols += count > 0;
    run.work.counts_visited = add(run.work.counts_visited, 1);
  }
  const auto m = run.active_symbols, merges = m ? m - 1 : 0;
  if (records)
    admit(merges, options.max_merge_records);
  const auto nodes = add(m, merges);
  if (nodes > run.nodes.max_size() || counts.size() > run.lengths.max_size())
    throw std::length_error("Huffman container size");
  run.total_weight = total.text();
  run.lengths.assign(counts.size(), 0);
  run.nodes.reserve(nodes);
  if (records)
    records->reserve(merges);
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  for (std::size_t s = 0; s < counts.size(); ++s)
    if (counts[s]) {
      const auto id = run.nodes.size();
      run.nodes.push_back({s, true, kHuffmanNoNode, kHuffmanNoNode});
      heap.push({Weight(counts[s]), id});
      run.work.heap_pushes = add(run.work.heap_pushes, 1);
    }
  Weight merge_cost = 0;
  while (heap.size() > 1) {
    const auto a = heap.top();
    heap.pop();
    const auto b = heap.top();
    heap.pop();
    run.work.heap_pops = add(run.work.heap_pops, 2);
    const auto   id    = run.nodes.size();
    const Weight sum   = a.first + b.first;
    run.nodes.push_back({0, false, a.second, b.second});
    if (records)
      records->push_back({a.second, b.second, id, a.first.text(), b.first.text(), sum.text()});
    heap.push({sum, id});
    run.work.heap_pushes = add(run.work.heap_pushes, 1);
    merge_cost += sum;
  }
  Weight weighted = 0;
  if (!heap.empty()) {
    run.root = heap.top().second;
    std::vector<std::pair<std::size_t, int>> stack{{run.root, 0}};
    while (!stack.empty()) {
      const auto [id, depth] = stack.back();
      stack.pop_back();
      run.work.tree_visits = add(run.work.tree_visits, 1);
      const auto& node     = run.nodes[id];
      if (node.leaf) {
        const int length         = depth ? depth : 1;
        run.lengths[node.symbol] = length;
        weighted += Weight(counts[node.symbol]) * length;
      } else {
        if (depth == std::numeric_limits<int>::max())
          throw std::length_error("Huffman depth exceeds int");
        stack.push_back({node.right, depth + 1});
        stack.push_back({node.left, depth + 1});
      }
    }
  }
  run.weighted_bits = weighted.text();
  run.merge_cost    = merge_cost.text();
  return run;
}
/// @brief Count actual source bytes under complete source-input admission.
/// @param text Borrowed complete arbitrary byte string.
/// @param options Complete source-byte policy.
/// @return Owned256-position unsigned-byte frequency vector.
/// @throws std::length_error On policy or count representation limits.
/// @throws std::bad_alloc On allocation failure.
std::vector<std::uint64_t> histogram(const std::string& text, HuffmanOptions options) {
  admit(text.size(), options.max_input_entries);
  std::vector<std::uint64_t> counts(256);
  for (unsigned char c : text) {
    if (counts[c] == std::numeric_limits<std::uint64_t>::max())
      throw std::length_error("Huffman byte count");
    ++counts[c];
  }
  return counts;
}
/// @brief Project the actual generic byte topology into the historical aggregate and assign path strings.
/// @param run Borrowed valid shared construction over256 positions.
/// @return Owned byte code with identical child/root IDs.
/// @throws std::bad_alloc On allocation failure.
HuffmanCode project(const HuffmanCountsRun& run) {
  HuffmanCode code;
  if (run.nodes.empty())
    return code;
  for (const auto& n : run.nodes)
    code.tree.push_back(
        {static_cast<char>(n.symbol),
         n.leaf,
         n.leaf ? -1 : static_cast<int>(n.left),
         n.leaf ? -1 : static_cast<int>(n.right)});
  code.root = static_cast<int>(run.root);
  std::vector<std::pair<int, std::string>> stack{{code.root, ""}};
  while (!stack.empty()) {
    auto [id, path] = std::move(stack.back());
    stack.pop_back();
    const auto& n = code.tree[id];
    if (n.leaf)
      code.codes[n.symbol] = path.empty() ? "0" : path;
    else {
      stack.push_back({n.right, path + "1"});
      stack.push_back({n.left, path + "0"});
    }
  }
  return code;
}
/// @brief Execute one actual byte histogram/shared construction/projection without replay.
/// @param text Borrowed source bytes.
/// @param options Complete source/merge policies.
/// @param records Optional actual merge capture destination.
/// @param frequencies Optional owned actual histogram destination.
/// @return Owned producing byte model and shared construction.
/// @throws std::length_error On policies/count/container limits.
/// @throws std::bad_alloc On allocation failure.
HuffmanBuildRun build(
    const std::string&          text,
    HuffmanOptions              options,
    std::vector<HuffmanMerge>*  records,
    std::vector<std::uint64_t>* frequencies) {
  auto            counts = histogram(text, options);
  HuffmanBuildRun run;
  run.construction                      = execute(counts, options, records, text.size());
  run.construction.work.histogram_bytes = text.size();
  run.code                              = project(run.construction);
  if (frequencies)
    *frequencies = std::move(counts);
  return run;
}
} // namespace huffman_detail
HuffmanCode huffman_build(const std::string& text) {
  return huffman_detail::build(text, {0, 0}, nullptr, nullptr).code;
}
HuffmanBuildRun huffman_build_run(const std::string& text, HuffmanOptions options) {
  return huffman_detail::build(text, options, nullptr, nullptr);
}
HuffmanBuildTrace huffman_build_trace(const std::string& text, HuffmanOptions options) {
  HuffmanBuildTrace t;
  t.run = huffman_detail::build(text, options, &t.merges, &t.frequencies);
  return t;
}
HuffmanCountsRun huffman_counts_run(
    const std::vector<std::uint64_t>& counts,
    HuffmanOptions                    options) {
  return huffman_detail::execute(counts, options, nullptr, counts.size());
}
HuffmanCountsTrace huffman_counts_trace(
    const std::vector<std::uint64_t>& counts,
    HuffmanOptions                    options) {
  HuffmanCountsTrace t;
  t.run    = huffman_detail::execute(counts, options, &t.merges, counts.size());
  t.counts = counts;
  return t;
}
std::vector<int> huffman_code_lengths(const std::vector<std::uint64_t>& counts) {
  return huffman_counts_run(counts, {0, 0}).lengths;
}
void validate_huffman_code(const HuffmanCode& code) {
  if (code.root == -1 && code.tree.empty() && code.codes.empty())
    return;
  if (code.root < 0 || static_cast<std::size_t>(code.root) >= code.tree.size() ||
      code.tree.size() > 511 || code.codes.empty())
    throw std::invalid_argument("Huffman root/empty model");
  std::array<bool, 511>                    visited{};
  std::array<bool, 256>                    symbols{};
  std::size_t                              seen = 0, leaves = 0;
  std::vector<std::pair<int, std::string>> stack{{code.root, ""}};
  while (!stack.empty()) {
    auto [id, path] = std::move(stack.back());
    stack.pop_back();
    if (id < 0 || static_cast<std::size_t>(id) >= code.tree.size() || visited[id] ||
        path.size() > 255)
      throw std::invalid_argument("Huffman repeated/invalid node");
    visited[id] = true;
    ++seen;
    const auto& node = code.tree[id];
    if (node.leaf) {
      const auto symbol = static_cast<unsigned char>(node.symbol);
      if (node.left != -1 || node.right != -1 || symbols[symbol])
        throw std::invalid_argument("Huffman leaf invariant");
      symbols[symbol] = true;
      ++leaves;
      const auto it = code.codes.find(node.symbol);
      if (it == code.codes.end() || it->second != (path.empty() ? "0" : path))
        throw std::invalid_argument("Huffman map/path mismatch");
    } else {
      if (node.left < 0 || node.right < 0 || node.left == node.right ||
          static_cast<std::size_t>(node.left) >= code.tree.size() ||
          static_cast<std::size_t>(node.right) >= code.tree.size())
        throw std::invalid_argument("Huffman child invariant");
      stack.push_back({node.right, path + "1"});
      stack.push_back({node.left, path + "0"});
    }
  }
  if (seen != code.tree.size() || leaves != code.codes.size())
    throw std::invalid_argument("Huffman disconnected/model map invariant");
}
std::string huffman_encode(const HuffmanCode& hc, const std::string& text) {
  validate_huffman_code(hc);
  std::size_t bits = 0;
  for (char ch : text)
    bits = huffman_detail::add(bits, hc.codes.at(ch).size());
  std::string out;
  if (bits > out.max_size())
    throw std::length_error("Huffman payload size");
  out.reserve(bits);
  for (char ch : text)
    out += hc.codes.at(ch);
  return out;
}
std::string huffman_decode(const HuffmanCode& hc, const std::string& bits) {
  validate_huffman_code(hc);
  for (char bit : bits)
    if (bit != '0' && bit != '1')
      throw std::invalid_argument("Huffman nonbinary payload");
  std::string out;
  if (hc.root < 0) {
    if (!bits.empty())
      throw std::invalid_argument("Huffman nonempty payload for empty model");
    return out;
  }
  if (hc.tree[hc.root].leaf) {
    for (char bit : bits)
      if (bit != '0')
        throw std::invalid_argument("Huffman singleton requires zero digits");
    out.assign(bits.size(), hc.tree[hc.root].symbol);
    return out;
  }
  int node = hc.root;
  for (char bit : bits) {
    node = bit == '0' ? hc.tree[node].left : hc.tree[node].right;
    if (hc.tree[node].leaf) {
      out.push_back(hc.tree[node].symbol);
      node = hc.root;
    }
  }
  if (node != hc.root)
    throw std::invalid_argument("Huffman incomplete payload suffix");
  return out;
}
double huffman_average_bits(const HuffmanCode& hc, const std::string& text) {
  validate_huffman_code(hc);
  huffman_detail::Weight total = 0;
  for (char c : text)
    total += hc.codes.at(c).size();
  return text.empty() ? 0 : total.as_double() / static_cast<double>(text.size());
}
} // namespace datamunge::algorithms
