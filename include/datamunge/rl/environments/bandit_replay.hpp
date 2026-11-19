#pragma once

/// @file bandit_replay.hpp
/// @brief Record selected arms and observed payouts from an existing bandit rollout.

#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/cad/video.hpp"

namespace datamunge::rl::environments {

/// @brief Encode a bandit rollout with the selected arm and observed reward highlighted.
///
/// Every frame displays the pull index and immediate reward. The selected arm
/// is green when its reward reaches the supplied threshold and red otherwise.
/// The function reads the caller's trace without drawing more random numbers.
/// Requires ffmpeg with libx264 on PATH.
/// @param arms Number of arms, from 1 to 20.
/// @param pulls Nonempty chronological pairs of zero-based arm and finite reward.
/// @param destination MP4 path with an existing parent directory.
/// @param success_threshold Reward at or above which to mark a successful pull. Defaults to 1.
/// @param fps One frame per pull, at this integer frame rate. Defaults to 4.
/// @throws std::invalid_argument Invalid arm count, trace, threshold or output settings.
/// @throws std::runtime_error Encoder startup, write or finalization failure.
/// @par Example
/// @code
/// dm::rl::environments::save_bandit_replay(2, {{0,0.f},{1,1.f}}, "pulls.mp4");
/// @endcode
/// @see cad::VideoWriter
inline void save_bandit_replay(int arms, const std::vector<std::pair<int, float>>& pulls,
                               const std::string& destination, float success_threshold = 1.f,
                               int fps = 4) {
  if (arms < 1 || arms > 20 || pulls.empty() || !std::isfinite(success_threshold))
    throw std::invalid_argument("invalid bandit replay geometry or threshold");
  for (const auto& [arm, reward] : pulls)
    if (arm < 0 || arm >= arms || !std::isfinite(reward))
      throw std::invalid_argument("invalid arm or reward in bandit replay");

  cad::RenderResult frame;
  frame.width = 640;
  frame.height = 240;
  frame.rgba.resize(frame.width * frame.height * 4);
  const auto panel = [&frame](int x, int y, int width, int height,
                              unsigned char red, unsigned char green, unsigned char blue) {
    for (int j = y; j < y + height; ++j) {
      for (int i = x; i < x + width; ++i) {
        auto* pixel = frame.rgba.data() + 4 * (j * frame.width + i);
        pixel[0] = red;
        pixel[1] = green;
        pixel[2] = blue;
        pixel[3] = 255;
      }
    }
  };
  cad::VideoWriter video(destination, frame.width, frame.height, fps);
  const int columns = (arms + 1) / 2;
  const int cell = 600 / columns;
  for (std::size_t step = 0; step < pulls.size(); ++step) {
    const auto [selected, reward] = pulls[step];
    panel(0, 0, frame.width, frame.height, 15, 23, 42);
    cad::draw_text(frame, std::format("Pull {} of {}", step + 1, pulls.size()), 20, 18, 3);
    cad::draw_text(frame, std::format("Arm {}   Reward {:+.3g}", selected, reward), 20, 52);
    for (int arm = 0; arm < arms; ++arm) {
      const int x = 20 + (arm % columns) * cell;
      const int y = 96 + (arm / columns) * 61;
      if (arm != selected) panel(x, y, cell - 6, 48, 51, 65, 85);
      else if (reward >= success_threshold) panel(x, y, cell - 6, 48, 21, 128, 61);
      else panel(x, y, cell - 6, 48, 185, 28, 28);
      const auto label = std::to_string(arm);
      cad::draw_text(frame, label, x + (cell - 6 - int(label.size()) * 12) / 2, y + 17);
    }
    video.append(frame);
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
