#pragma once

#include <cstdint>

namespace datamunge::random {

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed);

  std::uint64_t next_u64();
  double        uniform01();
  double        normal();

 private:
  std::uint64_t state_;
  bool          has_spare_{false};
  double        spare_{0.0};
};

} // namespace datamunge::random
