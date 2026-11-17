/// @file lucas_primality.cpp
/// @brief Shared exact bounded Lucas factor/order producer and actual observations.
#include <datamunge/algorithms/lucas_primality.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private actual-record admission and shared exact producer.
namespace {
/// @brief Admit one actual logical capture record before appending it.
/// @param trace Optional actual capture output.
/// @param quota Optional whole capture cap, zero disables it.
void admit(LucasPrimalityTrace* trace, std::size_t quota) {
  if (!trace)
    return;
  if (trace->captured_entries == std::numeric_limits<std::size_t>::max() ||
      (quota && trace->captured_entries >= quota))
    throw std::length_error("Lucas: trace quota");
  ++trace->captured_entries;
}
/// @brief Factor and search once, with optional actual emitted history and explicit incomplete outcomes.
/// @param n Original candidate.
/// @param trials Actual divisor-test budget.
/// @param bases Actual base budget.
/// @param trace Optional actual capture output.
/// @param quota Optional capture cap.
/// @return Actual bounded same-run decision and selected work.
LucasPrimalityAnalysis produce(
    std::uint64_t        n,
    std::size_t          trials,
    std::size_t          bases,
    LucasPrimalityTrace* trace,
    std::size_t          quota) {
  LucasPrimalityAnalysis out;
  out.n = n;
  admit(trace, quota);
  if (n < 2) {
    out.stop       = LucasPrimalityStop::SmallComposite;
    out.conclusive = true;
    return out;
  }
  if (n == 2) {
    out.stop       = LucasPrimalityStop::SmallPrime;
    out.conclusive = out.prime = out.factorization_complete = true;
    out.remaining                                           = 1;
    out.witness                                             = 1;
    return out;
  }
  if (n % 2 == 0) {
    out.stop       = LucasPrimalityStop::EvenComposite;
    out.conclusive = true;
    return out;
  }
  out.remaining = n - 1;
  for (std::uint64_t d = 2; d <= out.remaining / d; ++d) {
    if (out.work.trial_divisors == trials) {
      out.stop = LucasPrimalityStop::FactorBudget;
      return out;
    }
    ++out.work.trial_divisors;
    const auto before   = out.remaining;
    unsigned   exponent = 0;
    while (out.remaining % d == 0) {
      out.remaining /= d;
      ++exponent;
      ++out.work.successful_divisions;
    }
    if (exponent)
      out.factors.push_back({d, exponent});
    if (trace) {
      admit(trace, quota);
      trace->factor_steps.push_back({d, before, out.remaining, exponent});
    }
  }
  if (out.remaining > 1)
    out.factors.push_back({out.remaining, 1});
  out.remaining              = 1;
  out.factorization_complete = true;
  auto power                 = [&](std::uint64_t base, std::uint64_t q) {
    const auto exponent = q ? (n - 1) / q : n - 1;
    const auto residue  = mod_pow(base, exponent, n);
    ++out.work.powers;
    const bool passed = q ? residue != 1 : residue == 1;
    if (trace) {
      admit(trace, quota);
      trace->powers.push_back({base, q, exponent, residue, passed});
    }
    return passed;
  };
  for (std::uint64_t a = 2; a < n; ++a) {
    if (out.work.bases == bases) {
      out.stop = LucasPrimalityStop::BaseBudget;
      return out;
    }
    ++out.work.bases;
    const bool    fermat   = power(a, 0);
    std::uint64_t rejected = 0;
    bool          accepted = fermat;
    if (fermat)
      for (const auto& factor : out.factors)
        if (!power(a, factor.prime)) {
          rejected = factor.prime;
          accepted = false;
          break;
        }
    if (trace) {
      admit(trace, quota);
      trace->bases.push_back({a, fermat, rejected, accepted});
    }
    if (!fermat) {
      out.stop           = LucasPrimalityStop::CompositeFermat;
      out.composite_base = a;
      out.conclusive     = true;
      return out;
    }
    if (accepted) {
      out.stop    = LucasPrimalityStop::PrimeCertificate;
      out.witness = a;
      out.prime = out.conclusive = true;
      return out;
    }
  }
  throw std::logic_error("Lucas: exhaustive base search contradicted exact order criterion");
}
} // namespace
LucasPrimalityAnalysis
lucas_primality_analyze(std::uint64_t n, std::size_t max_trial_divisors, std::size_t max_bases) {
  return produce(n, max_trial_divisors, max_bases, nullptr, 0);
}
LucasPrimalityTrace lucas_primality_trace(
    std::uint64_t n,
    std::size_t   max_trial_divisors,
    std::size_t   max_bases,
    std::size_t   max_entries) {
  LucasPrimalityTrace t;
  t.analysis = produce(n, max_trial_divisors, max_bases, &t, max_entries);
  return t;
}
bool lucas_primality_test(std::uint64_t n) {
  auto a = lucas_primality_analyze(n);
  if (!a.conclusive)
    throw std::length_error("Lucas: work budget exhausted, no primality decision");
  return a.prime;
}
std::vector<LucasPrimalityAnalysis> lucas_primality_census(
    std::uint64_t first,
    std::uint64_t last) {
  if (first > last || last - first >= 10000)
    throw std::invalid_argument("Lucas: interval must contain1..10000 candidates");
  std::vector<LucasPrimalityAnalysis> out;
  out.reserve(last - first + 1);
  for (auto n = first;; ++n) {
    out.push_back(lucas_primality_analyze(n));
    if (n == last)
      break;
  }
  return out;
}
} // namespace datamunge::algorithms
