/// @file matrix_allocations.cpp
/// @brief Allocation-budget regression for deferred inverse-action product chains.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <datamunge/linalg/bindings.hpp>
#include <datamunge/linalg/matrix.hpp>
#include <new>
#include <vector>

namespace {
bool        tracking = false; ///< Only count the expression formation/evaluation under test.
std::size_t calls    = 0;     ///< Number of allocations in the measured section.
std::size_t largest  = 0;     ///< Largest single allocation in bytes.
} // namespace

/// @brief Intercept ordinary allocations only to measure the expression's workspace.
/// @param size Allocation size in bytes.
/// @return Owned storage released by operator delete.
/// @throws std::bad_alloc On allocation failure.
void* operator new(std::size_t size) {
  if (tracking) {
    ++calls;
    largest = std::max(largest, size);
  }
  if (void* value = std::malloc(size ? size : 1))
    return value;
  throw std::bad_alloc();
}
/// @brief Release intercepted ordinary storage.
/// @param value Previously allocated block.
void operator delete(void* value) noexcept {
  std::free(value);
}
/// @brief Release a sized ordinary block.
/// @param value Previously allocated block.
/// @param size Unused allocation size.
void operator delete(void* value, std::size_t size) noexcept {
  (void)size;
  std::free(value);
}

/// @brief Check that lazy construction allocates nothing and solving never creates the wide
/// intermediate.
/// @return Zero when results and both allocation budgets hold, otherwise one.
int main() {
  datamunge::Mat<> a = {{2, 0}, {0, 4}};
  datamunge::Mat<> b(2, 20000), c(20000, 1);
  b.set(0, 0, 6);
  b.set(1, 19999, 8);
  c.set(0, 0, 1);
  c.set(19999, 0, 1);
  tracking        = true;
  auto expression = datamunge::solve(a) * b * c;
  tracking        = false;
  if (calls != 0) {
    std::fprintf(stderr, "Expression construction allocated %zu times\n", calls);
    return 1;
  }
  tracking          = true;
  const auto result = expression.eval();
  tracking          = false;
  // A solved 2-by-20000 intermediate would allocate 320000 bytes. Every required
  // coefficient, RHS, pivot and triangular-solve buffer fits comfortably below 1 KiB.
  if (largest >= 1024 || result(0, 0) != 3 || result(1, 0) != 2) {
    std::fprintf(
        stderr, "Solve allocated a %zu-byte block or returned incorrect coefficients\n", largest);
    return 1;
  }
  std::printf("Lazy construction: 0 allocations; largest solve allocation: %zu bytes\n", largest);

  /// @brief Verify that both integer and floating-point -1 powers retain solve fusion.
  /// @param exponent Either -1 or -1.0.
  /// @return True for allocation-free construction and a narrow, correct solve.
  const auto check_power = [&](auto exponent) {
    calls = largest = 0;
    tracking = true;
    const auto powered = (a ^ exponent) * b * c;
    tracking = false;
    if (calls != 0)
      return false;
    tracking = true;
    const auto value = powered.eval();
    tracking = false;
    return largest < 1024 && value(0, 0) == 3 && value(1, 0) == 2;
  };
  if (!check_power(-1) || !check_power(-1.0))
    return 1;
  std::printf("Power -1/-1.0: 0 construction allocations; fused solve below 1 KiB\n");

  datamunge::Mat<> design(20000, 2), response(20000, 1);
  for (std::size_t i = 0; i < design.rows(); ++i) {
    design.set(i, 0, 1);
    design.set(i, 1, static_cast<double>(i % 2));
    response.set(i, 0, 1 + 2 * static_cast<double>(i % 2));
  }
  calls = largest    = 0;
  tracking           = true;
  auto least_squares = datamunge::solve(design.t() * design) * design.t() * response;
  tracking           = false;
  if (calls != 0)
    return 1;
  tracking        = true;
  const auto beta = least_squares.eval();
  tracking        = false;
  // No materialised design transpose, pseudoinverse or rows-by-rows orthogonal factor.
  if (largest >= 1024 || beta(0, 0) != 1 || beta(1, 0) != 2)
    return 1;
  std::printf("20000-row least squares: largest evaluation allocation: %zu bytes\n", largest);

  std::vector<double> x_values(40000), y_values(20000);
  for (std::size_t i = 0; i < 20000; ++i) {
    x_values[2 * i]     = 1;
    x_values[2 * i + 1] = static_cast<double>(i % 2);
    y_values[i]         = 1 + 2 * static_cast<double>(i % 2);
  }
  datamunge::LinalgMatrix x_handle(x_values, 20000, 2), y_handle(y_values, 20000, 1);
  calls = largest = 0;
  tracking        = true;
  auto bound      = x_handle.transpose()
                   .multiply(x_handle)
                   .inverse_action()
                   .multiply(x_handle.transpose())
                   .multiply(y_handle);
  auto bound_beta = bound.values();
  tracking        = false;
  // Runtime nodes allocate small handles, never a design-sized transpose or solved intermediate.
  if (largest >= 1024 || bound_beta[0] != 1 || bound_beta[1] != 2)
    return 1;
  std::printf(
      "Binding least squares: largest construction/evaluation allocation: %zu bytes\n", largest);
}
