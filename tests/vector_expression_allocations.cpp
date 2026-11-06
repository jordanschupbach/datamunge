/// @file vector_expression_allocations.cpp
/// @brief Allocation regression for fused Doppler arithmetic, sine, noise addition and reductions.
#include <datamunge/vector.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool tracking = false; ///< Only record measured operations.
std::size_t calls = 0; ///< Number of measured allocations.
std::size_t largest = 0; ///< Largest measured allocation in bytes.
}
/// @brief Intercept ordinary allocations made by vector storage.
/// @param size Allocation size.
/// @return Owned allocation.
/// @throws std::bad_alloc On allocation failure.
void* operator new(std::size_t size) {
  if (tracking) { ++calls; largest = std::max(largest, size); }
  if (void* result = std::malloc(size ? size : 1)) return result;
  throw std::bad_alloc();
}
/// @brief Free an intercepted allocation.
/// @param value Owned memory.
void operator delete(void* value) noexcept { std::free(value); }
/// @brief Free an intercepted sized allocation.
/// @param value Owned memory.
/// @param size Unused size.
void operator delete(void* value, std::size_t size) noexcept { (void)size; std::free(value); }

/// @brief Verify zero-allocation formation and one-allocation fused materialisation.
/// @return Zero when values and budgets hold, otherwise one.
int main() {
  namespace dm = datamunge;
  constexpr std::size_t n = 100000;
  dm::set_seed(42);
  const auto x = dm::runif(0, 1, n);
  auto noise = dm::rnorm(0, 1, n);
  const double expected = std::sin(20 / (x[0] + 0.25)) + noise[0];
  const auto doppler = [](auto&& input) {
    return dm::sin(20 / (std::forward<decltype(input)>(input) + 0.25));
  };
  tracking = true;
  const auto expression = doppler(x) + std::move(noise);
  tracking = false;
  if (calls != 0) return 1;
  std::printf("Doppler/sine/addition construction: %zu allocations\n", calls);
  calls = largest = 0; tracking = true;
  const dm::Vector<> snapshot = expression;
  tracking = false;
  if (calls != 1 || largest != n * sizeof(double) || snapshot[0] != expected) return 1;
  std::printf("Fused evaluation: %zu allocation, %zu bytes\n", calls, largest);
  calls = largest = 0; tracking = true;
  const std::vector<double> standard = expression;
  tracking = false;
  if (calls != 1 || largest != n * sizeof(double) || standard[0] != expected) return 1;
  std::printf("Standard-vector consumer: %zu allocation\n", calls);
  calls = largest = 0; tracking = true;
  const double sum = dm::sum(expression), mean = dm::mean(expression), sd = dm::sd(expression);
  tracking = false;
  if (calls != 0 || !std::isfinite(sum) || !std::isfinite(mean) || !std::isfinite(sd)) return 1;
  std::printf("Streaming sum/mean/sd: %zu allocations\n", calls);
  calls = largest = 0; tracking = true;
  const auto fresh_noise = doppler(x) + dm::rnorm(0, 1, n);
  tracking = false;
  if (calls != 1 || largest != n * sizeof(double)) return 1;
  // Only the random sample owns storage; the arithmetic still has no result array.
  std::printf("Construction with rnorm: %zu allocation for sampled noise\n", calls);
  return fresh_noise.size() != n;
}
