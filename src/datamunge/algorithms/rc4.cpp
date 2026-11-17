/// @file rc4.cpp
/// @brief Shared classic RC4 schedule/step producer and actual deterministic bias-study generator.
#include <datamunge/algorithms/rc4.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
/// @brief Canonical historical byte-cipher implementation.
namespace datamunge::algorithms {
Rc4::Rc4(const std::string& key, std::vector<Rc4KsaStep>* scheduling) {
  if (key.empty())
    throw std::invalid_argument("RC4: key must be nonempty");
  for (std::size_t k = 0; k < 256; ++k)
    s_[k] = static_cast<std::uint8_t>(k);
  std::uint8_t j = 0;
  for (std::size_t k = 0; k < 256; ++k) {
    const auto before = j, key_byte = static_cast<std::uint8_t>(key[k % key.size()]);
    j = static_cast<std::uint8_t>(j + s_[k] + key_byte);
    const Rc4KsaStep record{k, before, j, key_byte, s_[k], s_[j]};
    std::swap(s_[k], s_[j]);
    if (scheduling)
      scheduling->push_back(record);
  }
}
Rc4Step Rc4::next_step() {
  Rc4Step step;
  step.i_before     = i_;
  step.j_before     = j_;
  i_                = static_cast<std::uint8_t>(i_ + 1);
  j_                = static_cast<std::uint8_t>(j_ + s_[i_]);
  step.i_after      = i_;
  step.j_after      = j_;
  step.left_before  = s_[i_];
  step.right_before = s_[j_];
  std::swap(s_[i_], s_[j_]);
  step.output_index = static_cast<std::uint8_t>(s_[i_] + s_[j_]);
  step.output       = s_[step.output_index];
  return step;
}
std::uint8_t Rc4::next() {
  return next_step().output;
}
Rc4State Rc4::state() const {
  return {s_, i_, j_};
}
Rc4Trace rc4_trace(const std::string& key, std::size_t n, const Rc4Options& o) {
  if (key.empty())
    throw std::invalid_argument("RC4: key must be nonempty");
  if ((o.max_bytes && n > o.max_bytes) || n > std::numeric_limits<std::size_t>::max() - 772 ||
      (o.max_trace_entries && n + 772 > o.max_trace_entries))
    throw std::length_error("RC4: output/history policy");
  Rc4Trace t;
  Rc4      stream(key, &t.scheduling);
  t.scheduled = stream.state();
  t.steps.reserve(n);
  t.bytes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto step = stream.next_step();
    t.bytes.push_back(step.output);
    t.steps.push_back(step);
  }
  t.final = stream.state();
  return t;
}
std::vector<std::uint8_t> rc4_keystream(const std::string& key, std::size_t n) {
  Rc4 stream(key);
  if (n > 1000000)
    throw std::length_error("RC4: default output policy");
  std::vector<std::uint8_t> out(n);
  for (auto& x : out)
    x = stream.next();
  return out;
}
std::string rc4_encrypt(const std::string& key, const std::string& data) {
  Rc4 stream(key);
  if (data.size() > 1000000)
    throw std::length_error("RC4: default output policy");
  std::string out = data;
  for (char& x : out)
    x = static_cast<char>(static_cast<std::uint8_t>(x) ^ stream.next());
  return out;
}
Rc4BiasStudy rc4_second_byte_study(std::size_t count, std::size_t max_keys) {
  if (count > std::numeric_limits<std::uint32_t>::max() || (max_keys && count > max_keys))
    throw std::length_error("RC4 bias: key policy");
  Rc4BiasStudy out;
  out.keys = count;
  std::unordered_set<std::string> distinct;
  for (std::size_t k = 0; k < count; ++k) {
    const auto raw = static_cast<std::uint32_t>(k) * std::uint32_t{2654435761U};
    const auto key = "k" + std::to_string(raw % 1000000U);
    distinct.insert(key);
    Rc4 stream(key);
    stream.next();
    const auto byte = stream.next();
    ++out.counts[byte];
    ++out.groups[byte >> 4];
  }
  out.unique_keys = distinct.size();
  return out;
}
} // namespace datamunge::algorithms
