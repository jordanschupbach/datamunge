/// @file safetensors.cpp
/// @brief Strict bounded Float32 SafeTensors decoder; never executes serialized code.
#include <algorithm>
#include <bit>
#include <datamunge/nn/safetensors.hpp>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
namespace datamunge::nn {
NamedTensors load_safetensors(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("cannot open SafeTensors file");
  auto size = in.tellg();
  if (size < 10 || size > std::streamoff(1073741824))
    throw std::invalid_argument("SafeTensors file size outside bounds");
  in.seekg(0);
  unsigned char bytes[8];
  in.read(reinterpret_cast<char*>(bytes), 8);
  std::uint64_t length = 0;
  for (int i = 0; i < 8; ++i)
    length |= std::uint64_t(bytes[i]) << (8 * i);
  if (length < 2 || length > 16777216 || length > std::uint64_t(size) - 8)
    throw std::invalid_argument("invalid SafeTensors header size");
  std::string header(length, ' ');
  in.read(header.data(), length);
  nlohmann::json meta;
  try {
    std::map<int, std::set<std::string>> keys;
    meta = nlohmann::json::parse(
        header, [&](int depth, nlohmann::json::parse_event_t event, nlohmann::json& value) {
          if (event == nlohmann::json::parse_event_t::object_start)
            keys[depth + 1].clear();
          if (event == nlohmann::json::parse_event_t::key &&
              !keys[depth].insert(value.get<std::string>()).second)
            throw std::invalid_argument("duplicate SafeTensors JSON key");
          return true;
        });
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid SafeTensors JSON");
  }
  if (!meta.is_object())
    throw std::invalid_argument("SafeTensors header must be an object");
  NamedTensors                                         out;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  const std::uint64_t                                  payload = std::uint64_t(size) - 8 - length;
  for (auto it = meta.begin(); it != meta.end(); ++it) {
    if (it.key() == "__metadata__")
      continue;
    const auto& entry = it.value();
    if (!entry.is_object() || !entry.contains("dtype") || !entry["dtype"].is_string() ||
        entry["dtype"] != "F32" || !entry.contains("shape") || !entry["shape"].is_array() ||
        !entry.contains("data_offsets") || !entry["data_offsets"].is_array() ||
        entry["data_offsets"].size() != 2)
      throw std::invalid_argument("unsupported SafeTensors entry");
    Shape         shape;
    std::uint64_t count = 1;
    for (const auto& d : entry["shape"]) {
      if (!d.is_number_integer() || d.get<std::int64_t>() < 0)
        throw std::invalid_argument("invalid SafeTensors shape");
      auto n = d.get<std::uint64_t>();
      if (n && count > payload / 4 / n)
        throw std::invalid_argument("SafeTensors element count overflow");
      count *= n;
      shape.push_back(n);
    }
    auto offsets = entry["data_offsets"];
    for (auto& v : offsets)
      if (!v.is_number_integer() || v.get<std::int64_t>() < 0)
        throw std::invalid_argument("invalid SafeTensors offset");
    auto begin = offsets[0].get<std::uint64_t>(), end = offsets[1].get<std::uint64_t>();
    if (begin > end || end > payload || end - begin != count * 4)
      throw std::invalid_argument("SafeTensors data range mismatch");
    ranges.emplace_back(begin, end);
    std::vector<float> values(count);
    in.seekg(8 + length + begin);
    in.read(reinterpret_cast<char*>(values.data()), count * 4);
    if (!in)
      throw std::runtime_error("truncated SafeTensors data");
    if constexpr (std::endian::native != std::endian::little)
      for (auto& v : values)
        v = std::bit_cast<float>(std::byteswap(std::bit_cast<std::uint32_t>(v)));
    out.emplace_back(it.key(), from_vector(values, shape));
  }
  std::sort(ranges.begin(), ranges.end());
  std::uint64_t end = 0;
  for (auto range : ranges) {
    if (range.first != end)
      throw std::invalid_argument("SafeTensors gaps or overlapping entries");
    end = range.second;
  }
  if (end != payload)
    throw std::invalid_argument("SafeTensors trailing payload");
  return out;
}
} // namespace datamunge::nn
