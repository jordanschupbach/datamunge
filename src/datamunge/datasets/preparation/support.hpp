#pragma once
/// @file support.hpp
/// @brief Private bounded native preparation utilities and reproducible metadata RNG.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <datamunge/datasets/assets.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
namespace datamunge::datasets::detail {
namespace fs = std::filesystem;
using Json   = nlohmann::ordered_json; ///< Provenance keeps explicit insertion order.

/// @brief Own a uniquely created staging directory; remove it even on exceptions.
struct Temporary {
  fs::path path; ///< Directory owned by this instance.

  /// @brief Create a unique directory below parent.
  /// @param parent Existing staging parent.
  explicit Temporary(const fs::path& parent) {
    /// @brief Process-local uniqueness counter for staging names.
    static std::atomic<std::uint64_t> serial{0};
    for (int i = 0; i < 100; ++i) {
      path = parent / (".native-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                       "-" + std::to_string(serial++));
      if (fs::create_directory(path))
        return;
    }
    throw std::runtime_error("cannot create native staging directory");
  }
  /// @brief Staging ownership cannot be copied.
  /// @param other Unused source, since copying is prohibited.
  Temporary(const Temporary& other) = delete;
  /// @brief Clean up unpublished staging files.
  ~Temporary() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};
/// @brief Read a metadata file as bytes.
/// @param p Source metadata path.
/// @return Owned contents, with no text transcoding.
inline std::string read(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot read " + p.string());
  std::string s{std::istreambuf_iterator<char>(f), {}};
  if (f.bad())
    throw std::runtime_error("read failed: " + p.string());
  return s;
}
/// @brief Atomically write owned metadata bytes.
/// @param p Destination path.
/// @param s Exact bytes.
inline void write(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  Temporary     stage(p.parent_path());
  std::ofstream f(stage.path / "file", std::ios::binary);
  f.write(s.data(), s.size());
  f.close();
  if (!f)
    throw std::runtime_error("write failed: " + p.string());
  fs::rename(stage.path / "file", p);
}
/// @brief Decode an owned provenance JSON document.
/// @param p Source path.
/// @return Parsed JSON value.
inline Json json(const fs::path& p) {
  return Json::parse(read(p));
}
/// @brief Write readable provenance with a terminating newline.
/// @param p Destination path.
/// @param j Provenance value.
inline void save(const fs::path& p, const Json& j) {
  write(p, j.dump(2, ' ', false, Json::error_handler_t::replace) + "\n");
}
/// @brief Python-compatible integer-seeded MT19937 for preserving published row identities.
/// @details Implements the init_by_array seed and getrandbits rejection shuffle used
/// by the former data scripts; no host interpreter or std::shuffle variability.
struct MetadataRandom {
  std::array<std::uint32_t, 624> state{};  ///< MT19937 state words.
  std::size_t                    at = 624; ///< Next state word.

  /// @brief Seed using the original one-word nonnegative integer protocol.
  /// @param seed Pinned report metadata seed.
  explicit MetadataRandom(std::uint32_t seed) {
    state[0] = 19650218;
    for (std::size_t i = 1; i < 624; ++i)
      state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    std::size_t i = 1;
    for (int k = 0; k < 624; ++k) {
      state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525U)) + seed;
      if (++i >= 624) {
        state[0] = state[623];
        i        = 1;
      }
    }
    for (int k = 0; k < 623; ++k) {
      state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941U)) - i;
      if (++i >= 624) {
        state[0] = state[623];
        i        = 1;
      }
    }
    state[0] = 0x80000000U;
  }
  /// @brief Generate one tempered MT19937 word.
  /// @return Uniform 32-bit word.
  std::uint32_t word() {
    if (at == 624) {
      for (std::size_t i = 0; i < 624; ++i) {
        auto y   = (state[i] & 0x80000000U) | (state[(i + 1) % 624] & 0x7fffffffU);
        state[i] = state[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0);
      }
      at = 0;
    }
    auto y = state[at++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    return y ^ (y >> 18);
  }
  /// @brief Shuffle metadata with bit-length rejection sampling.
  /// @tparam T Owned metadata element type.
  /// @param values Mutable ordered metadata vector.
  template<class T>
  void shuffle(std::vector<T>& values) {
    for (std::size_t n = values.size(); n > 1; --n) {
      unsigned k = 0;
      for (auto x = n; x; x >>= 1)
        ++k;
      std::uint32_t r;
      do {
        r = word() >> (32 - k);
      } while (r >= n);
      std::swap(values[n - 1], values[r]);
    }
  }
};
/// @brief Prepare the canonical local memory or balanced MNIST asset pools.
/// @param name "memory" or "mnist-assets" adapter name.
/// @param root Parent cache directory.
void prepare_basic(const std::string& name, const fs::path& root);
/// @brief Execute a codec process directly without shell interpolation.
/// @param args Executable name and exact arguments.
void run(const std::vector<std::string>& args);
} // namespace datamunge::datasets::detail
