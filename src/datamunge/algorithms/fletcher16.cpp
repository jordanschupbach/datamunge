/// @file
/// @brief Canonical modulo255 Fletcher byte folds, composition and complete corruption scans.
#include <datamunge/algorithms/fletcher16.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
/// @brief Private raw-state validation, shared sequential folds and complete-budget checks.
namespace {
/// @brief Canonical modulo255 residues before packing into the public raw tag.
struct FletcherState {
  std::uint16_t a = 0; ///< First residue,always below255.
  std::uint16_t b = 0; ///< Second residue,always below255.
};
/// @brief Validate and unpack a canonical raw tag without touching message data.
/// @param tag Packed high-B/low-A raw state.
/// @return Canonical residues.
/// @throws std::invalid_argument If either lane equals255.
FletcherState fletcher_state(std::uint16_t tag) {
  FletcherState state{static_cast<std::uint16_t>(tag & 255U), static_cast<std::uint16_t>(tag >> 8)};
  if (state.a == 255 || state.b == 255)
    throw std::invalid_argument("Fletcher state lanes must be below255");
  return state;
}
/// @brief Pack canonical raw residues with B high and A low.
/// @param state Valid canonical residues.
/// @return Raw16-bit tag,without complements or protocol check-byte insertion.
std::uint16_t fletcher_pack(FletcherState state) {
  return static_cast<std::uint16_t>((state.b << 8) | state.a);
}
/// @brief Fold every actual unsigned byte through one shared modulo255 recurrence.
/// @param state Valid starting residues.
/// @param data Valid byte view,never retained.
/// @param steps Optional pre-admitted actual-event destination.
/// @return Canonical final residues.
/// @throws std::bad_alloc If an optional record append cannot allocate storage.
/// @throws std::length_error If an optional event container exceeds capacity.
FletcherState fletcher_fold(
    FletcherState                    state,
    std::string_view                 data,
    std::vector<Fletcher16ByteStep>* steps = nullptr) {
  std::size_t index = 0;
  for (unsigned char byte : data) {
    const auto before = state;
    state.a           = static_cast<std::uint16_t>((state.a + byte) % 255U);
    state.b           = static_cast<std::uint16_t>((state.b + state.a) % 255U);
    if (steps)
      steps->push_back({index, byte, before.a, before.b, state.a, state.b});
    ++index;
  }
  return state;
}
/// @brief Compute a canonical zero-initialised tag through the production byte kernel.
/// @param data Valid read-only byte view.
/// @return Complete raw Fletcher-16 tag.
std::uint16_t fletcher_raw(std::string_view data) {
  return fletcher_pack(fletcher_fold({}, data));
}
/// @brief Validate complete copies*n byte folds and the optional cap before allocation.
/// @param n Bytes per complete fold.
/// @param copies Number of complete message folds.
/// @param cap Optional byte-fold limit,zero disables it.
/// @return Exact representable total folds.
/// @throws std::length_error For product overflow or exceeded nonzero cap.
std::size_t fletcher_work(std::size_t n, std::size_t copies, std::size_t cap) {
  if (copies && n > std::numeric_limits<std::size_t>::max() / copies)
    throw std::length_error("Fletcher complete work overflow");
  const auto work = n * copies;
  if (cap && work > cap)
    throw std::length_error("Fletcher complete work cap");
  return work;
}
/// @brief Require that a complete scratch/copy message can fit string storage.
/// @param n Complete byte count.
/// @throws std::length_error If n exceeds string capacity.
void fletcher_string_capacity(std::size_t n) {
  if (n > std::string{}.max_size())
    throw std::length_error("Fletcher message copy capacity");
}
/// @brief Subtract two canonical residues modulo255 without unsigned underflow.
/// @param changed New residue.
/// @param original Baseline residue.
/// @return Canonical modular change in0..254.
std::uint16_t fletcher_delta(std::uint16_t changed, std::uint16_t original) {
  return static_cast<std::uint16_t>((changed + 255U - original) % 255U);
}
} // namespace
std::uint16_t fletcher16(const std::string& data) {
  return fletcher_raw(data);
}
std::uint16_t fletcher16_update(std::uint16_t checksum, std::string_view data) {
  return fletcher_pack(fletcher_fold(fletcher_state(checksum), data));
}
std::uint16_t
fletcher16_combine(std::uint16_t left, std::uint16_t right, std::uint64_t right_length) {
  const auto a = fletcher_state(left), b = fletcher_state(right);
  const auto first  = static_cast<std::uint16_t>((a.a + b.a) % 255U);
  const auto second = static_cast<std::uint16_t>((a.b + b.b + (right_length % 255U) * a.a) % 255U);
  return fletcher_pack({first, second});
}
Fletcher16Trace fletcher16_trace(std::string_view data, std::size_t max_bytes) {
  if ((max_bytes && data.size() > max_bytes) ||
      data.size() > std::vector<Fletcher16ByteStep>{}.max_size())
    throw std::length_error("Fletcher complete trace capacity");
  Fletcher16Trace out;
  out.steps.reserve(data.size());
  const auto state = fletcher_fold({}, data, &out.steps);
  out.sum1         = state.a;
  out.sum2         = state.b;
  out.checksum     = fletcher_pack(state);
  return out;
}
Fletcher16FlipScan fletcher16_flip_scan(std::string_view data, std::size_t max_bytes_processed) {
  if (data.size() > std::numeric_limits<std::size_t>::max() / 8)
    throw std::length_error("Fletcher bit count overflow");
  const auto bits = data.size() * 8;
  const auto work = fletcher_work(data.size(), bits + 1, max_bytes_processed);
  if (bits > std::vector<Fletcher16Flip>{}.max_size())
    throw std::length_error("Fletcher bit record capacity");
  fletcher_string_capacity(data.size());
  Fletcher16FlipScan out;
  out.bytes_processed = work;
  out.flips.reserve(bits);
  std::string scratch(data);
  const auto  baseline = fletcher_fold({}, data);
  out.checksum         = fletcher_pack(baseline);
  for (std::size_t bit = 0; bit < bits; ++bit) {
    const auto index    = bit / 8;
    const auto before   = static_cast<unsigned char>(scratch[index]);
    scratch[index]      = static_cast<char>(before ^ (1U << (bit % 8)));
    const auto state    = fletcher_fold({}, scratch);
    scratch[index]      = static_cast<char>(before);
    const auto checksum = fletcher_pack(state);
    const bool detected = checksum != out.checksum;
    out.flips.push_back(
        {bit,
         checksum,
         fletcher_delta(state.a, baseline.a),
         fletcher_delta(state.b, baseline.b),
         detected});
    if (detected)
      ++out.detected;
  }
  return out;
}
Fletcher16ReplacementScan fletcher16_replacement_scan(
    std::string_view data,
    std::size_t      index,
    std::size_t      max_bytes_processed) {
  if (index >= data.size())
    throw std::invalid_argument("Fletcher replacement index outside message");
  const auto work = fletcher_work(data.size(), 257, max_bytes_processed);
  fletcher_string_capacity(data.size());
  if (std::vector<Fletcher16Replacement>{}.max_size() < 256)
    throw std::length_error("Fletcher replacement record capacity");
  Fletcher16ReplacementScan out;
  out.index           = index;
  out.original        = static_cast<std::uint8_t>(data[index]);
  out.bytes_processed = work;
  out.replacements.reserve(256);
  out.checksum = fletcher_raw(data);
  std::string scratch(data);
  for (unsigned value = 0; value < 256; ++value) {
    scratch[index]      = static_cast<char>(value);
    const auto checksum = fletcher_raw(scratch);
    const bool changed = value != out.original, detected = checksum != out.checksum;
    out.replacements.push_back({static_cast<std::uint8_t>(value), checksum, changed, detected});
    if (changed)
      ++out.changed;
    if (detected)
      ++out.detected;
    if (changed && !detected)
      ++out.missed;
  }
  return out;
}
std::uint16_t
fletcher16_transposition_delta(std::uint64_t distance, std::uint8_t left, std::uint8_t right) {
  return static_cast<std::uint16_t>(((distance % 255U) * ((right + 255U - left) % 255U)) % 255U);
}
std::uint32_t byte_sum32(std::string_view data) {
  std::uint32_t sum = 0;
  for (unsigned char byte : data)
    sum += byte;
  return sum;
}
std::string checksum_swap_bytes(std::string_view data, std::size_t first, std::size_t second) {
  if (first >= data.size() || second >= data.size())
    throw std::invalid_argument("checksum swap index outside message");
  fletcher_string_capacity(data.size());
  std::string out(data);
  std::swap(out[first], out[second]);
  return out;
}
} // namespace datamunge::algorithms
