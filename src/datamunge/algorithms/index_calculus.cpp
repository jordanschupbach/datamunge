/// @file index_calculus.cpp
/// @brief Shared bounded index-calculus relation collection, prime-power linear algebra and target search.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/index_calculus.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/pollard_rho.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact modular arithmetic and shared preparation/query kernels.
namespace {
/// @brief Native public residue width.
using U64 = std::uint64_t;
/// @brief Exact product intermediate for every pair of uint64 residues.
using U128 = unsigned __int128;
/// @brief Owned row-major modular system coefficients.
using Matrix = std::vector<std::vector<U64>>;
/// @brief Increment a selected actual count without unsigned wrap.
/// @param value Mutable count.
/// @throws std::length_error On exhausted count range.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::length_error("index calculus count");
  ++value;
}
/// @brief Compute exact modular product using a widened intermediate.
/// @param a First residue.
/// @param b Second residue.
/// @param modulus Positive modulus.
/// @return a*b modulo modulus.
U64 multiply(U64 a, U64 b, U64 modulus) {
  return static_cast<U64>(static_cast<U128>(a) * b % modulus);
}
/// @brief Compute exact modular addition without uint64 overflow.
/// @param a First residue.
/// @param b Second residue.
/// @param modulus Positive modulus.
/// @return a+b modulo modulus.
U64 add(U64 a, U64 b, U64 modulus) {
  return static_cast<U64>((static_cast<U128>(a) + b) % modulus);
}
/// @brief Subtract two reduced residues without unsigned addition overflow.
/// @param a Reduced first residue.
/// @param b Reduced second residue.
/// @param modulus Positive common modulus.
/// @return a-b modulo modulus.
U64 subtract(U64 a, U64 b, U64 modulus) {
  return a >= b ? a - b : modulus - (b - a);
}
/// @brief Exact square-and-multiply powering.
/// @param a Base.
/// @param exponent Nonnegative exponent.
/// @param modulus Positive modulus.
/// @return a^exponent modulo modulus.
U64 power(U64 a, U64 exponent, U64 modulus) {
  U64 r = 1 % modulus;
  a %= modulus;
  while (exponent) {
    if (exponent & 1)
      r = multiply(r, a, modulus);
    a = multiply(a, a, modulus);
    exponent >>= 1;
  }
  return r;
}
/// @brief Original three-shift uint64 RNG, deterministic including zero-seed substitution.
struct XorShift {
  U64 state; ///< Actual nonzero transition state.

  /// @brief Initialize one independent stream.
  /// @param seed Initial state, zero replaced by a fixed nonzero constant.
  explicit XorShift(U64 seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Advance exactly the original 13/7/17 unsigned shift recurrence.
  /// @return Actual next uint64 state.
  U64 next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};
/// @brief Check normalized primitive-generator order using complete known factors.
/// @param g Normalized nonzero residue.
/// @param p Validated prime.
/// @param factors Complete sorted factors of p-1 with multiplicity.
/// @throws std::invalid_argument On a nongenerator.
void generator(U64 g, U64 p, const std::vector<U64>& factors) {
  U64 previous = 0;
  for (auto q : factors) {
    if (q == previous)
      continue;
    previous = q;
    if (power(g, (p - 1) / q, p) == 1)
      throw std::invalid_argument("index calculus requires a primitive generator");
  }
}
/// @brief Validate prime/generator and factor its complete group order.
/// Even p-1 is peeled by the shared factorizer, leaving an odd part <=2^63-1;
/// no new full-width domain restriction is imposed on the prime modulus.
/// @param g Mutable generator, normalized after prime validation.
/// @param p Requested modulus.
/// @return Complete sorted group-order prime factors.
/// @throws std::invalid_argument On nonprime p, zero or nongenerator g.
/// @throws std::bad_alloc On factor storage failure.
std::vector<U64> group(U64& g, U64 p) {
  if (!is_probable_prime(p))
    throw std::invalid_argument("index calculus prime modulus required");
  g %= p;
  if (!g)
    throw std::invalid_argument("index calculus nonzero generator required");
  auto factors = factorize(p - 1);
  generator(g, p, factors);
  return factors;
}
/// @brief Factor a positive residue over the complete supplied base, counting actual tests.
/// @param value Nonzero ordinary integer residue.
/// @param base Sorted factor-base primes.
/// @param exponents Owned output, replaced by complete nonnegative exponents.
/// @param tests Mutable count of actual remainder tests.
/// @return Whether the remaining cofactor is one.
/// @throws std::length_error On count overflow/capacity exhaustion.
/// @throws std::bad_alloc On exponent storage failure.
bool smooth(
    U64                     value,
    const std::vector<U64>& base,
    std::vector<U64>&       exponents,
    std::size_t&            tests) {
  exponents.assign(base.size(), 0);
  for (std::size_t i = 0; i < base.size(); ++i)
    for (;;) {
      increment(tests);
      if (value % base[i])
        break;
      value /= base[i];
      ++exponents[i];
    }
  return value == 1;
}
/// @brief Actual reduced prime-field solve outcome.
struct FieldSolution {
  std::size_t      rank{0};          ///< Actual pivot count.
  bool             consistent{true}; ///< Whether every zero row has zero RHS.
  std::vector<U64> solution; ///< Full unique canonical solution only when consistent and full rank.
};
/// @brief Solve one overdetermined system over a validated prime field.
/// @param original Borrowed rectangular coefficient matrix with K columns.
/// @param rhs Borrowed matching RHS.
/// @param q Prime field modulus, at least two.
/// @param columns Complete variable count K.
/// @param work Mutable actual pivot/row-reduction counts.
/// @return Actual rank/consistency and full solution when unique.
/// @throws std::length_error On counter/capacity overflow.
/// @throws std::bad_alloc On working-copy allocation failure.
FieldSolution field_solve(
    const Matrix&           original,
    const std::vector<U64>& rhs,
    U64                     q,
    std::size_t             columns,
    IndexCalculusWork&      work) {
  Matrix a = original;
  auto   b = rhs;
  for (auto& row : a)
    for (auto& x : row)
      x %= q;
  for (auto& x : b)
    x %= q;
  FieldSolution            out;
  std::vector<std::size_t> where(columns, a.size());
  for (std::size_t col = 0; col < columns && out.rank < a.size(); ++col) {
    std::size_t selected = out.rank;
    while (selected < a.size() && !a[selected][col])
      ++selected;
    if (selected == a.size())
      continue;
    std::swap(a[out.rank], a[selected]);
    std::swap(b[out.rank], b[selected]);
    where[col] = out.rank;
    increment(work.field_pivots);
    const auto inverse = power(a[out.rank][col], q - 2, q);
    for (auto& x : a[out.rank])
      x = multiply(x, inverse, q);
    b[out.rank] = multiply(b[out.rank], inverse, q);
    for (std::size_t r = 0; r < a.size(); ++r) {
      if (r == out.rank || !a[r][col])
        continue;
      increment(work.row_eliminations);
      const auto f = a[r][col];
      for (std::size_t c = 0; c < columns; ++c)
        a[r][c] = subtract(a[r][c], multiply(f, a[out.rank][c], q), q);
      b[r] = subtract(b[r], multiply(f, b[out.rank], q), q);
    }
    ++out.rank;
  }
  for (std::size_t r = out.rank; r < a.size(); ++r)
    if (b[r] && std::all_of(a[r].begin(), a[r].end(), [](U64 v) { return v == 0; })) {
      out.consistent = false;
      return out;
    }
  if (out.rank == columns) {
    out.solution.resize(columns);
    for (std::size_t c = 0; c < columns; ++c)
      out.solution[c] = b[where[c]];
  }
  return out;
}
/// @brief Admit one complete dimension-based relation/stage history record.
/// @param trace Optional capture destination; null disables history admission.
/// @param entries Mutable cumulative logical record entries.
/// @param columns Complete solution/exponent dimension, reserved even for a failed empty stage.
/// @param options Actual history policy.
/// @throws std::length_error On count overflow or complete history cap.
void admit(
    IndexCalculusPreparationTrace* trace,
    std::size_t&                   entries,
    std::size_t                    columns,
    IndexCalculusOptions           options) {
  if (!trace)
    return;
  const auto max = std::numeric_limits<std::size_t>::max();
  if (columns == max)
    throw std::length_error("index calculus history dimension");
  const auto cost = 1 + columns;
  if (entries > max - cost ||
      (options.max_history_entries && cost > options.max_history_entries - entries))
    throw std::length_error("index calculus complete history cap");
  entries += cost;
}
/// @brief Prepare one validated group through the shared producing kernel.
/// @param g Valid normalized primitive generator.
/// @param p Valid prime.
/// @param factors Complete group-order prime factors, moved into the output.
/// @param seed Actual initial RNG seed.
/// @param options Bound/budget/admission policies.
/// @param trace Optional actual relation/stage capture, initially empty.
/// @return Owned ready or explicitly failed table.
/// @throws std::invalid_argument On explicit invalid factor-base bound.
/// @throws std::length_error On complete admission/count/capacity overflow.
/// @throws std::bad_alloc On working/output allocation failure.
IndexCalculusTable prepare(
    U64                            g,
    U64                            p,
    std::vector<U64>               factors,
    U64                            seed,
    IndexCalculusOptions           options,
    IndexCalculusPreparationTrace* trace) {
  IndexCalculusTable table;
  table.g             = g;
  table.p             = p;
  table.order_factors = std::move(factors);
  XorShift rng(seed);
  table.query_state = rng.state;
  U64 bound         = options.factor_base_bound;
  if (!bound) {
    if (p <= 3)
      bound = 2;
    else {
      const double ln       = std::log(static_cast<double>(p));
      const double estimate = 3. * std::exp(.5 * std::sqrt(ln * std::log(ln)));
      bound                 = static_cast<U64>(std::clamp(estimate, 30., 4000.));
    }
  }
  bound = std::min(bound, p - 1);
  if (p > 2 && bound < 2)
    throw std::invalid_argument("index calculus factor base bound below two");
  table.bound    = bound;
  const auto max = std::numeric_limits<std::size_t>::max();
  if (bound >= max || (options.max_sieve_entries && bound + 1 > options.max_sieve_entries))
    throw std::length_error("index calculus complete sieve cap");
  std::vector<bool> composite(static_cast<std::size_t>(bound) + 1, false);
  for (U64 q = 2; q <= bound; ++q) {
    if (composite[q])
      continue;
    table.factor_base.push_back(q);
    if (q <= bound / q)
      for (U64 m = q * q;;) {
        composite[m] = true;
        if (m > bound - q)
          break;
        m += q;
      }
  }
  const auto k = table.factor_base.size();
  if (!k) {
    table.status = IndexCalculusStatus::ready;
    return table;
  }
  if (options.relation_surplus > max - k)
    throw std::length_error("index calculus row count");
  const auto want = k + options.relation_surplus;
  if (want > max / k || (options.max_matrix_cells && want * k > options.max_matrix_cells))
    throw std::length_error("index calculus complete matrix cap");
  Matrix           a;
  std::vector<U64> b, exponents;
  std::size_t      history_entries = 0;
  for (std::size_t attempt = 0; attempt < options.relation_attempts && a.size() < want; ++attempt) {
    increment(table.work.relation_attempts);
    const U64 exponent = 1 + rng.next() % (p - 2), value = power(g, exponent, p);
    if (!smooth(value, table.factor_base, exponents, table.work.divisibility_tests))
      continue;
    admit(trace, history_entries, k, options);
    if (trace)
      trace->relations.push_back({table.work.relation_attempts, exponent, value, exponents});
    a.push_back(exponents);
    b.push_back(exponent);
    increment(table.work.accepted_relations);
  }
  table.query_state = rng.state;
  if (a.size() < k)
    return table;
  std::vector<U64> assembled(k, 0);
  U64              combined = 1;
  for (std::size_t first = 0; first < table.order_factors.size();) {
    const auto  q    = table.order_factors[first];
    std::size_t last = first;
    while (last < table.order_factors.size() && table.order_factors[last] == q)
      ++last;
    U64  modulus = q;
    auto solved  = field_solve(a, b, q, k, table.work);
    admit(trace, history_entries, k, options);
    if (trace)
      trace->stages.push_back({q, modulus, solved.rank, solved.solution});
    if (!solved.consistent) {
      table.status = IndexCalculusStatus::verification_failed;
      return table;
    }
    if (solved.rank < k) {
      table.status = IndexCalculusStatus::rank_deficient;
      return table;
    }
    auto x = std::move(solved.solution);
    for (std::size_t level = first + 1; level < last; ++level) {
      const U64        next = static_cast<U64>(static_cast<U128>(modulus) * q);
      std::vector<U64> residual(a.size());
      for (std::size_t r = 0; r < a.size(); ++r) {
        U64 lhs = 0;
        for (std::size_t c = 0; c < k; ++c)
          lhs = add(lhs, multiply(a[r][c], x[c], next), next);
        const U64 difference = subtract(b[r] % next, lhs, next);
        if (difference % modulus) {
          table.status = IndexCalculusStatus::verification_failed;
          return table;
        }
        residual[r] = difference / modulus;
      }
      auto correction = field_solve(a, residual, q, k, table.work);
      if (!correction.consistent || correction.rank < k) {
        table.status = IndexCalculusStatus::verification_failed;
        return table;
      }
      for (std::size_t c = 0; c < k; ++c)
        x[c] = static_cast<U64>(static_cast<U128>(modulus) * correction.solution[c] + x[c]);
      modulus = next;
      admit(trace, history_entries, k, options);
      if (trace)
        trace->stages.push_back({q, modulus, correction.rank, x});
    }
    const U64 phi = modulus - modulus / q, inverse = power(combined % modulus, phi - 1, modulus);
    for (std::size_t c = 0; c < k; ++c) {
      const U64 offset =
          multiply(subtract(x[c], assembled[c] % modulus, modulus), inverse, modulus);
      assembled[c] = static_cast<U64>(static_cast<U128>(combined) * offset + assembled[c]);
    }
    combined = static_cast<U64>(static_cast<U128>(combined) * modulus);
    first    = last;
  }
  for (std::size_t c = 0; c < k; ++c)
    if (power(g, assembled[c], p) != table.factor_base[c]) {
      table.status = IndexCalculusStatus::verification_failed;
      return table;
    }
  table.logarithms = std::move(assembled);
  table.status     = IndexCalculusStatus::ready;
  return table;
}
/// @brief Completely validate a borrowed public table before target arithmetic.
/// @param table Borrowed editable value table.
/// @throws std::invalid_argument On malformed group/factor data/base/logarithms/status.
void validate_table(const IndexCalculusTable& table) {
  const auto p = table.p, g = table.g;
  if (!is_probable_prime(p) || !g || g >= p || !table.bound || table.bound >= p ||
      (p > 2 && table.bound < 2))
    throw std::invalid_argument("index calculus malformed table group");
  U64 remaining = p - 1, previous = 0;
  for (auto q : table.order_factors) {
    if (q < previous || !is_probable_prime(q) || remaining % q)
      throw std::invalid_argument("index calculus malformed order factors");
    remaining /= q;
    previous = q;
  }
  if (remaining != 1)
    throw std::invalid_argument("index calculus incomplete order factors");
  generator(g, p, table.order_factors);
  previous = 0;
  for (auto q : table.factor_base) {
    if (q <= previous || q > table.bound || q >= p || !is_probable_prime(q))
      throw std::invalid_argument("index calculus malformed factor base");
    previous = q;
  }
  switch (table.status) {
    case IndexCalculusStatus::ready:
      if (table.logarithms.size() != table.factor_base.size())
        throw std::invalid_argument("index calculus table log dimensions");
      for (std::size_t i = 0; i < table.logarithms.size(); ++i)
        if (table.logarithms[i] >= p - 1 ||
            power(g, table.logarithms[i], p) != table.factor_base[i])
          throw std::invalid_argument("index calculus unverified table logarithm");
      break;
    case IndexCalculusStatus::relation_limit:
    case IndexCalculusStatus::rank_deficient:
    case IndexCalculusStatus::verification_failed:
      if (!table.logarithms.empty())
        throw std::invalid_argument("index calculus failed table has logs");
      break;
    default:
      throw std::invalid_argument("index calculus invalid table status");
  }
}
/// @brief Perform the shared target search on a validated table and normalized nonzero target.
/// @param table Valid ready or failed table.
/// @param h Valid nonzero normalized target.
/// @param attempts Actual target trial budget, zero for none.
/// @return Verified exponent and actual successful relation/work, or bounded failure.
/// @throws std::length_error On counter/capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
IndexCalculusQuery query(const IndexCalculusTable& table, U64 h, std::size_t attempts) {
  IndexCalculusQuery out;
  if (h == 1) {
    out.result = {0, true};
    return out;
  }
  if (h == table.g) {
    out.result = {1, true};
    return out;
  }
  if (table.status != IndexCalculusStatus::ready)
    return out;
  XorShift         rng(table.query_state);
  std::vector<U64> exponents;
  const auto       order = table.p - 1;
  for (std::size_t i = 0; i < attempts; ++i) {
    increment(out.attempts);
    const U64 shift = rng.next() % order,
              value = multiply(h, power(table.g, shift, table.p), table.p);
    if (!smooth(value, table.factor_base, exponents, out.divisibility_tests))
      continue;
    U64 x = 0;
    for (std::size_t j = 0; j < exponents.size(); ++j)
      x = add(x, multiply(exponents[j], table.logarithms[j], order), order);
    x = subtract(x, shift, order);
    if (power(table.g, x, table.p) == h) {
      out.result  = {x, true};
      out.shift   = shift;
      out.residue = value;
      out.powers  = std::move(exponents);
      return out;
    }
  }
  return out;
}
} // namespace
IndexCalculusTable index_calculus_prepare(
    std::uint64_t        g,
    std::uint64_t        p,
    std::uint64_t        seed,
    IndexCalculusOptions options) {
  auto factors = group(g, p);
  return prepare(g, p, std::move(factors), seed, options, nullptr);
}
IndexCalculusPreparationTrace index_calculus_prepare_trace(
    std::uint64_t        g,
    std::uint64_t        p,
    std::uint64_t        seed,
    IndexCalculusOptions options) {
  auto                          factors = group(g, p);
  IndexCalculusPreparationTrace out;
  out.table = prepare(g, p, std::move(factors), seed, options, &out);
  return out;
}
IndexCalculusQuery
index_calculus_query(const IndexCalculusTable& table, std::uint64_t h, std::size_t max_attempts) {
  validate_table(table);
  h %= table.p;
  if (!h)
    throw std::invalid_argument("index calculus nonzero target required");
  return query(table, h, max_attempts);
}
DiscreteLogResult
index_calculus(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t seed) {
  auto factors = group(g, p);
  h %= p;
  if (!h)
    throw std::invalid_argument("index calculus nonzero target required");
  if (h == 1)
    return {0, true};
  if (h == g)
    return {1, true};
  auto table = prepare(g, p, std::move(factors), seed, {}, nullptr);
  return query(table, h, 4000000).result;
}
} // namespace datamunge::algorithms
