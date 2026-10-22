// Small runtime checks used by the executable nn examples. They stay on in
// Release builds and make the example exit nonzero on failure, so `just
// run-all-nn` doubles as a regression test against the recorded TF numbers.
#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace check {

inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline void Near(double actual, double expected, double tolerance, const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance)
    throw std::runtime_error(message + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
}

/// Expects `f` to throw; returns the message.
template <typename F>
std::string Throws(F&& f, const std::string& message) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  throw std::runtime_error(message + ": expected an exception");
}

}  // namespace check

inline void Header(const std::string& title) {
  const std::size_t pad = title.size() < 60 ? 60 - title.size() : 0;
  std::cout << "\n== " << title << " " << std::string(pad, '=') << "\n";
}
