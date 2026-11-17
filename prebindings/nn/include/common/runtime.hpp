// Bounded demonstrations by default; opt into convergence experiments explicitly.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace example_runtime {

inline bool Full() {
  const char* value = std::getenv("DATAMUNGE_EXAMPLES_FULL");
  return value && std::string(value) == "1";
}

template <typename T> T Limit(T requested, T quick_limit) {
  return Full() ? requested : std::min(requested, quick_limit);
}

inline void Describe() {
  if (!Full())
    std::cout << "Quick demonstration: bounded training; convergence metrics require "
                 "DATAMUNGE_EXAMPLES_FULL=1.\n";
}

}  // namespace example_runtime
