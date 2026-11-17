/// @file kahan.cpp
/// @brief Original-order double Kahan stream and complete actual arithmetic capture.
#include <datamunge/algorithms/kahan.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
KahanStep KahanAccumulator::add(double value) {
  if (count_ == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Kahan count overflow");
  KahanStep s;
  s.index             = count_;
  s.value             = value;
  s.sum_before        = sum_;
  s.correction_before = correction_;
  s.adjusted          = value - correction_;
  s.total             = sum_ + s.adjusted;
  s.correction_after  = (s.total - sum_) - s.adjusted;
  correction_         = s.correction_after;
  sum_                = s.total;
  ++count_;
  return s;
}
void KahanAccumulator::add_repeated(double value, std::size_t n) {
  if (n > std::numeric_limits<std::size_t>::max() - count_)
    throw std::length_error("Kahan complete repeat count");
  for (std::size_t i = 0; i < n; ++i)
    (void)add(value);
}
double kahan_sum(const std::vector<double>& xs) {
  KahanAccumulator a;
  for (double x : xs)
    (void)a.add(x);
  return a.sum();
}
KahanTrace kahan_trace(const std::vector<double>& xs, std::size_t cap) {
  KahanTrace t;
  if ((cap && xs.size() > cap) || xs.size() > t.steps.max_size())
    throw std::length_error("Kahan complete record ceiling");
  t.steps.reserve(xs.size());
  KahanAccumulator a;
  for (double x : xs)
    t.steps.push_back(a.add(x));
  t.sum = a.sum();
  return t;
}
} // namespace datamunge::algorithms
