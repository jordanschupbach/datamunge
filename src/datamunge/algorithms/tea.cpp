/// @file tea.cpp
/// @brief Actual modulo2^32 TEA cycles; capture shares the encrypt/decrypt kernels.
#include <datamunge/algorithms/tea.hpp>
namespace datamunge::algorithms {
/// @brief Private fixed-work invocation kernel.
namespace {
/// @brief Execute the actual dependent half updates and capture their real intermediate.
TEARun execute(
    std::array<std::uint32_t, 2>        v,
    const std::array<std::uint32_t, 4>& k,
    bool                                decrypt,
    bool                                capture) {
  TEARun r;
  r.decrypting           = decrypt;
  std::uint32_t       v0 = v[0], v1 = v[1], sum = decrypt ? 0xC6EF3720u : 0;
  const std::uint32_t delta = 0x9E3779B9u;
  for (std::uint32_t i = 0; i < 32; ++i) {
    auto          before = std::array{v0, v1};
    std::uint32_t first, used;
    if (decrypt) {
      used = sum;
      v1 -= ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
      first = v1;
      v0 -= ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
      sum -= delta;
    } else {
      sum += delta;
      used = sum;
      v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
      first = v0;
      v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }
    ++r.cycles;
    r.half_updates += 2;
    if (capture)
      r.steps.push_back({i + 1, used, before, first, {v0, v1}});
  }
  r.value = {v0, v1};
  return r;
}
} // namespace
TEARun tea_encrypt_run(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k) {
  return execute(v, k, false, false);
}
TEARun tea_encrypt_trace(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k) {
  return execute(v, k, false, true);
}
TEARun tea_decrypt_run(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k) {
  return execute(v, k, true, false);
}
TEARun tea_decrypt_trace(std::array<std::uint32_t, 2> v, const std::array<std::uint32_t, 4>& k) {
  return execute(v, k, true, true);
}
std::array<std::uint32_t, 2> tea_encrypt(
    std::array<std::uint32_t, 2>        v,
    const std::array<std::uint32_t, 4>& k) {
  return tea_encrypt_run(v, k).value;
}
std::array<std::uint32_t, 2> tea_decrypt(
    std::array<std::uint32_t, 2>        v,
    const std::array<std::uint32_t, 4>& k) {
  return tea_decrypt_run(v, k).value;
}
} // namespace datamunge::algorithms
