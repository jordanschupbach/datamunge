/// @file
/// @brief Write monophonic notes and rests to a Standard MIDI File.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace datamunge {

/// @brief A timed monophonic note or rest for write_midi().
///
/// Events occupy consecutive time intervals.
/// @par Example
/// @code
/// const dm::MidiEvent middle_c{60, 480};
/// @endcode
/// @par References
/// - MIDI Association, Standard MIDI Files, https://midi.org/standard-midi-files
/// @see write_midi()
struct MidiEvent {
  int pitch; ///< MIDI pitch in [0, 127], or -1 for a rest.
  std::uint32_t ticks; ///< Positive duration in ticks at 480 ticks per quarter note.
};

/// @brief Write sequential notes and rests as one format-0 Standard MIDI track.
///
/// Uses channel 1, program 0, velocity 90, 480 ticks per quarter and 120 BPM.
/// Preserves leading, intervening and trailing silence. An empty sequence
/// writes a valid track with an immediate end. Opens the file after validation.
/// @param path Destination file. Existing contents are overwritten.
/// @param events Sequential note and rest events in playback order.
/// @return Number of note-on events written.
/// @throws std::invalid_argument For invalid pitches, zero durations or a delta above 0x0fffffff.
/// @throws std::length_error If the encoded track exceeds the 32-bit chunk length.
/// @throws std::runtime_error If the file cannot be opened, written or closed.
/// @par Example
/// @code
/// const std::array notes{dm::MidiEvent{60, 480}, dm::MidiEvent{-1, 240}, dm::MidiEvent{64, 480}};
/// const auto count = dm::write_midi("phrase.mid", notes); // 2
/// @endcode
/// @par References
/// - MIDI Association, Standard MIDI Files, https://midi.org/standard-midi-files
/// @see MidiEvent
inline std::size_t write_midi(const std::filesystem::path &path,
                              std::span<const MidiEvent> events) {
  // Tempo and acoustic grand piano program, both at time zero.
  std::vector<std::uint8_t> track{0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20, 0, 0xc0, 0};
  const auto append_delta = [&](std::uint32_t ticks) {
    std::array<std::uint8_t, 4> bytes{};
    std::size_t first = bytes.size() - 1;
    bytes[first] = static_cast<std::uint8_t>(ticks & 127);
    while ((ticks >>= 7) != 0)
      bytes[--first] = static_cast<std::uint8_t>((ticks & 127) | 128);
    track.insert(track.end(), bytes.begin() + first, bytes.end());
  };

  constexpr std::uint32_t max_delta = 0x0fffffff;
  std::uint32_t silence = 0;
  std::size_t notes = 0;
  for (const auto &[pitch, ticks] : events) {
    if (pitch < -1 || pitch > 127 || ticks == 0 || ticks > max_delta)
      throw std::invalid_argument("invalid MIDI pitch or duration");
    if (pitch == -1) {
      if (ticks > max_delta - silence)
        throw std::invalid_argument("MIDI rest exceeds maximum delta");
      silence += ticks;
      continue;
    }
    const auto key = static_cast<std::uint8_t>(pitch);
    append_delta(silence);
    track.insert(track.end(), {0x90, key, 90});
    append_delta(ticks);
    track.insert(track.end(), {0x80, key, 0});
    silence = 0;
    ++notes;
  }
  append_delta(silence);
  track.insert(track.end(), {0xff, 0x2f, 0});
  if (track.size() > std::numeric_limits<std::uint32_t>::max() ||
      track.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::length_error("MIDI track is too large");

  std::ofstream file(path, std::ios::binary);
  const unsigned char header[] = {'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 0, 0, 1,
                                  1, 0xe0, 'M', 'T', 'r', 'k'};
  file.write(reinterpret_cast<const char *>(header), sizeof(header));
  for (int shift : {24, 16, 8, 0})
    file.put(static_cast<char>((track.size() >> shift) & 255));
  file.write(reinterpret_cast<const char *>(track.data()),
             static_cast<std::streamsize>(track.size()));
  file.close();
  if (!file)
    throw std::runtime_error("could not write MIDI file");
  return notes;
}

} // namespace datamunge
