/// @file karatsuba.cpp
/// @brief Original decimal Karatsuba arithmetic with checked actual-entry and leaf-work capture.
#include <algorithm>
#include <datamunge/algorithms/karatsuba.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Normalized decimal arithmetic helpers and same-execution multiplication details.
namespace karatsuba_detail {
/// @brief Normalize validated decimal bytes to owned little-endian digits.
/// @param s Borrowed validated unsigned decimal bytes; empty becomes zero.
/// @return Owned normalized digits, at least one.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container limits.
std::vector<int> to_digits(const std::string& s) {
  std::vector<int> d;
  d.reserve(s.size());
  for (auto it = s.rbegin(); it != s.rend(); ++it)
    d.push_back(*it - '0');
  while (d.size() > 1 && d.back() == 0)
    d.pop_back();
  if (d.empty())
    d.push_back(0);
  return d;
}

/// @brief Format producing little-endian digits without leading zeros.
/// @param d Owned valid decimal product digits.
/// @return Owned normalized decimal string.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container limits.
std::string from_digits(std::vector<int> d) {
  while (d.size() > 1 && d.back() == 0)
    d.pop_back();
  std::string s;
  s.reserve(d.size());
  for (auto it = d.rbegin(); it != d.rend(); ++it)
    s.push_back(static_cast<char>('0' + *it));
  return s;
}

/// @brief Add two nonnegative decimal digit vectors, including padded recursive slices.
/// @param a Borrowed first operand.
/// @param b Borrowed second operand.
/// @return Owned exact sum digits, retaining actual highest input positions unless carrying grows them.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container limits.
std::vector<int> add_digits(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  int              carry = 0;
  for (std::size_t i = 0; i < a.size() || i < b.size() || carry; ++i) {
    int s = carry;
    if (i < a.size())
      s += a[i];
    if (i < b.size())
      s += b[i];
    out.push_back(s % 10);
    carry = s / 10;
  }
  if (out.empty())
    out.push_back(0);
  return out;
}

/// @brief Subtract normalized nonnegative decimal digit vectors with a>=b.
/// @param a Borrowed valid minuend, at least b by producing algebra.
/// @param b Borrowed valid subtrahend.
/// @return Owned normalized exact difference digits.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container limits.
std::vector<int> sub_digits(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  int              borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    int s = a[i] - borrow - (i < b.size() ? b[i] : 0);
    if (s < 0) {
      s += 10;
      borrow = 1;
    } else {
      borrow = 0;
    }
    out.push_back(s);
  }
  while (out.size() > 1 && out.back() == 0)
    out.pop_back();
  return out;
}

/// @brief Shift normalized decimal digits by k positions, preserving zero normalization.
/// @param a Borrowed normalized digits.
/// @param k Checked decimal shift, zero allowed.
/// @return Owned exact shifted digits.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container limits.
std::vector<int> shift_digits(const std::vector<int>& a, std::size_t k) { // multiply by 10^k
  if (a.size() == 1 && a[0] == 0)
    return a;
  std::vector<int> out(k, 0);
  out.insert(out.end(), a.begin(), a.end());
  return out;
}

/// @brief Execute the actual schoolbook leaf product, where at least one length<=32.
/// Each coefficient sums at most32 products of digits<=9, so long-long carry
/// arithmetic is bounded independently of the other operand length.
/// @param a Borrowed normalized first digits.
/// @param b Borrowed normalized second digits.
/// @return Owned normalized exact leaf product.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On shape/container limits.
std::vector<int> schoolbook(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() > std::numeric_limits<std::size_t>::max() - b.size())
    throw std::length_error("Karatsuba leaf shape");
  std::vector<long long> acc(a.size() + b.size(), 0);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j)
      acc[i + j] += static_cast<long long>(a[i]) * b[j];
  std::vector<int> out(acc.size(), 0);
  long long        carry = 0;
  for (std::size_t i = 0; i < acc.size(); ++i) {
    long long cur = acc[i] + carry;
    out[i]        = static_cast<int>(cur % 10);
    carry         = cur / 10;
  }
  while (carry) {
    out.push_back(static_cast<int>(carry % 10));
    carry /= 10;
  }
  while (out.size() > 1 && out.back() == 0)
    out.pop_back();
  return out;
}

/// @brief Add one selected work amount with representability checking.
/// @param counter Actual selected counter to mutate.
/// @param amount Actual producing operation amount.
/// @throws std::length_error Before counter overflow.
void add_work(std::size_t& counter, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - counter)
    throw std::length_error("Karatsuba counter overflow");
  counter += amount;
}
/// @brief Multiply actual digit vectors once, optionally owning actual call-tree records.
/// @param x Borrowed valid little-endian first digits, possibly high-zero padded.
/// @param y Borrowed valid little-endian second digits, possibly high-zero padded.
/// @param out Per-run actual result/work/history state.
/// @param o Optional actual call/depth ceilings.
/// @param capture Whether to record call metadata.
/// @param parent Actual parent id, max on root.
/// @param role Actual root/low/high/sum invocation role.
/// @param depth Actual recursion depth, root0.
/// @return Owned normalized exact product digits.
/// @throws std::length_error On ceilings or size/work/container limits.
/// @throws std::bad_alloc On allocation failure.
std::vector<int> multiply(
    const std::vector<int>& x,
    const std::vector<int>& y,
    KaratsubaTrace&         out,
    KaratsubaOptions        o,
    bool                    capture,
    std::size_t             parent,
    KaratsubaRole           role,
    std::size_t             depth) {
  if ((o.max_calls && out.work.recursive_calls >= o.max_calls) ||
      (o.max_depth && depth > o.max_depth) ||
      x.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      y.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("Karatsuba actual recursion admission");
  const auto id = out.work.recursive_calls;
  add_work(out.work.recursive_calls, 1);
  out.work.max_depth = std::max(out.work.max_depth, depth);
  const bool leaf    = x.size() <= 32 || y.size() <= 32;
  const auto m       = leaf ? 0 : std::min(x.size(), y.size()) / 2;
  if (capture)
    out.calls.push_back({parent, role, depth, x.size(), y.size(), m, leaf, 0});
  std::vector<int> product;
  if (leaf) {
    add_work(out.work.schoolbook_leaves, 1);
    if (y.size() && x.size() > std::numeric_limits<std::size_t>::max() / y.size())
      throw std::length_error("Karatsuba digit count");
    add_work(out.work.digit_products, x.size() * y.size());
    product = schoolbook(x, y);
  } else {
    const std::vector<int> x0(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m)),
        x1(x.begin() + static_cast<std::ptrdiff_t>(m), x.end()),
        y0(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(m)),
        y1(y.begin() + static_cast<std::ptrdiff_t>(m), y.end());
    const auto z0   = multiply(x0, y0, out, o, capture, id, KaratsubaRole::Low, depth + 1);
    const auto z2   = multiply(x1, y1, out, o, capture, id, KaratsubaRole::High, depth + 1);
    const auto zsum = multiply(
        add_digits(x0, x1), add_digits(y0, y1), out, o, capture, id, KaratsubaRole::Sum, depth + 1);
    const auto z1 = sub_digits(sub_digits(zsum, z2), z0);
    product       = add_digits(add_digits(shift_digits(z2, 2 * m), shift_digits(z1, m)), z0);
  }
  if (capture)
    out.calls[id].product_digits = product.size();
  return product;
}
/// @brief Validate complete decimal operands and execute one exact producing multiplication.
/// @param a Borrowed original first decimal bytes.
/// @param b Borrowed original second decimal bytes.
/// @param o Complete input/actual recursion ceilings.
/// @param capture Whether to own operands and actual call records.
/// @return Owned exact product and selected work/optional history.
/// @throws std::invalid_argument On nondecimal input bytes.
/// @throws std::length_error On ceilings or arithmetic/container limits.
/// @throws std::bad_alloc On allocation failure.
KaratsubaTrace run(const std::string& a, const std::string& b, KaratsubaOptions o, bool capture) {
  for (unsigned char c : a)
    if (c < '0' || c > '9')
      throw std::invalid_argument("Karatsuba decimal input");
  for (unsigned char c : b)
    if (c < '0' || c > '9')
      throw std::invalid_argument("Karatsuba decimal input");
  if (a.size() > std::numeric_limits<std::size_t>::max() - b.size() ||
      a.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      b.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      (o.max_input_digits && a.size() + b.size() > o.max_input_digits))
    throw std::length_error("Karatsuba complete input ceiling");
  KaratsubaTrace out;
  if (capture) {
    out.left  = a;
    out.right = b;
  }
  out.product = from_digits(multiply(
      to_digits(a),
      to_digits(b),
      out,
      o,
      capture,
      std::numeric_limits<std::size_t>::max(),
      KaratsubaRole::Root,
      0));
  return out;
}
} // namespace karatsuba_detail
std::string karatsuba_multiply(const std::string& a, const std::string& b) {
  return karatsuba_detail::run(a, b, {0, 0, 0}, false).product;
}
KaratsubaTrace karatsuba_trace(const std::string& a, const std::string& b, KaratsubaOptions o) {
  return karatsuba_detail::run(a, b, o, true);
}
} // namespace datamunge::algorithms
