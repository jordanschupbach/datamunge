/// @file trial_division.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <datamunge/algorithms/trial_division.hpp>
namespace datamunge::algorithms {
/// @brief Implement the checked canonical trial_division_work producer; contracts are declared in the matching header.
TrialDivisionWork trial_division_work(std::uint64_t n, std::size_t cap) {
  if (!n)
    throw std::invalid_argument("trial division: zero");
  TrialDivisionWork r;
  auto              divides = [&](std::uint64_t d) {
    if (r.tests == cap)
      throw std::length_error("trial division: work limit");
    ++r.tests;
    return n % d == 0;
  };
  while (n > 1 && divides(2)) {
    r.factors.push_back(2);
    n /= 2;
  }
  for (std::uint64_t d = 3; d <= n / d; d += 2) {
    ++r.odd_candidates;
    while (divides(d)) {
      r.factors.push_back(d);
      n /= d;
    }
  }
  if (n > 1)
    r.factors.push_back(n);
  return r;
}
/// @brief Implement the checked canonical trial_division_factorize producer; contracts are declared in the matching header.
std::vector<std::uint64_t> trial_division_factorize(std::uint64_t n) {
  return trial_division_work(n).factors;
}
} // namespace datamunge::algorithms
