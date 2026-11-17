#pragma once
/// @file atari_frames.hpp
/// @brief Source box-filter grayscale resize, two-frame pooling and owning four-frame stack; no
/// emulator.
#include <algorithm>
#include <datamunge/rl/environments/interface.hpp>
#include <deque>
namespace datamunge::rl::environments {
/// @brief Own four source-preprocessed grayscale frames without an emulator dependency.
///
/// Frame-major Float32 observations scale bytes by255. Integer box averages use source floor
/// division, including upsampling by nearest source cell. Copies own independent frame deques.
/// Mutations require external synchronization. RGB conversion and emulator frame skipping belong to
/// the caller.
///
/// @par Example
/// @code
/// dm::rl::environments::AtariFrameStack frames(2,2);
/// for(int i=0;i<4;++i)frames.PushFrame({0,64,128,255});auto obs=frames.StackedObs();
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236 ; source integer box-filter resize variant retained.
/// @see PushFrame(), StackedObs()
class AtariFrameStack {
 public:
  static constexpr int kSize  = 84; ///< Resized square side in pixels.
  static constexpr int kStack = 4;  ///< Required temporal frame count.

  /// @brief Configure positive bounded source screen geometry.
  ///
  /// @param raw_w Screen width in pixels, default160.
  /// @param raw_h Screen height in pixels, default210; product at most16million.
  /// @throws std::invalid_argument Nonpositive or excessive geometry.
  /// @par Example
  /// @code
  /// dm::rl::environments::AtariFrameStack frames(2,2);
  /// for(int i=0;i<4;++i)frames.PushFrame({0,64,128,255});auto obs=frames.StackedObs();
  /// @endcode
  /// @see PushFrame()
  AtariFrameStack(int raw_w = 160, int raw_h = 210) : raw_w_(raw_w), raw_h_(raw_h) {
    if (raw_w < 1 || raw_h < 1 || int64_t(raw_w) * raw_h > 16000000)
      throw std::invalid_argument("positive bounded raw frame geometry required");
  }
  /// @brief Discard all owned frames while retaining screen dimensions.
  ///
  /// Borrowed last_frame references are invalidated. Four pushes are needed before StackedObs.
  ///
  /// @par Example
  /// @code
  /// dm::rl::environments::AtariFrameStack frames(2,2);
  /// for(int i=0;i<4;++i)frames.PushFrame({0,64,128,255});auto obs=frames.StackedObs();
  /// frames.Clear();
  /// @endcode
  /// @see StackedObs()
  void Clear() { frames_.clear(); }
  // Max over the last two frames removes Atari flicker (`prev` may be empty).
  /// @brief Apply source elementwise byte maximum to the current screen.
  ///
  /// @param cur Mutable current screen, any width; owns its bytes.
  /// @param prev Borrowed prior screen of the same size, or empty for no pooling.
  /// @throws std::invalid_argument A nonempty previous screen has different size.
  /// @par Example
  /// @code
  /// std::vector<unsigned char>
  /// current{1,4};dm::rl::environments::AtariFrameStack::MaxPool(current,{3,2});
  /// @endcode
  /// @see PushFrame()
  static void MaxPool(std::vector<unsigned char>& cur, const std::vector<unsigned char>& prev) {
    if (!prev.empty() && prev.size() != cur.size())
      throw std::invalid_argument("max-pool frame geometry mismatch");
    if (!prev.empty())
      for (size_t i = 0; i < cur.size(); ++i)
        cur[i] = std::max(cur[i], prev[i]);
  }
  // Box-filter resize of the grayscale screen to 84x84.
  /// @brief Box-average a source screen and append its owning84-square bytes.
  ///
  /// Keeps only the last four frames. Coordinates and truncating integer averages preserve the
  /// source protocol. Borrowed previous last_frame references may be invalidated.
  ///
  /// @param gray Borrowed row-major bytes with exactly raw_w*raw_h entries.
  /// @throws std::invalid_argument The input does not match configured screen size.
  /// @par Example
  /// @code
  /// dm::rl::environments::AtariFrameStack frames(2,2);
  /// for(int i=0;i<4;++i)frames.PushFrame({0,64,128,255});auto obs=frames.StackedObs();
  /// @endcode
  /// @see StackedObs()
  void PushFrame(const std::vector<unsigned char>& gray) {
    if (gray.size() != size_t(raw_w_) * raw_h_)
      throw std::invalid_argument("raw grayscale frame size mismatch");
    std::vector<unsigned char> small(kSize * kSize);
    for (int y = 0; y < kSize; ++y)
      for (int x = 0; x < kSize; ++x) {
        const int y0 = y * raw_h_ / kSize, y1 = std::max(y0 + 1, (y + 1) * raw_h_ / kSize);
        const int x0 = x * raw_w_ / kSize, x1 = std::max(x0 + 1, (x + 1) * raw_w_ / kSize);
        int64_t   sum = 0, n = 0;
        for (int yy = y0; yy < y1; ++yy)
          for (int xx = x0; xx < x1; ++xx)
            sum += gray[yy * raw_w_ + xx], ++n;
        small[y * kSize + x] = static_cast<unsigned char>(sum / n);
      }
    frames_.push_back(std::move(small));
    while (static_cast<int>(frames_.size()) > kStack)
      frames_.pop_front();
  }
  /// @brief Read flattened four-frame observation size.
  ///
  /// @return 4*84*84=28224 coordinates.
  /// @see StackedObs()
  int obs_size() const { return kStack * kSize * kSize; }
  /// @brief Copy four frames to an owning normalized Float32 observation.
  ///
  /// Throws std::logic_error until four frames have been pushed, including after Clear.
  ///
  /// @return Frame-major28224 coordinates; byte values divided by255.f.
  /// @par Example
  /// @code
  /// dm::rl::environments::AtariFrameStack frames(2,2);
  /// for(int i=0;i<4;++i)frames.PushFrame({0,64,128,255});auto obs=frames.StackedObs();
  /// @endcode
  /// @see PushFrame()
  Obs StackedObs() const {
    if (frames_.size() != kStack)
      throw std::logic_error("four frames required before reading observation");
    Obs o(obs_size());
    for (int f = 0; f < kStack; ++f)
      for (int i = 0; i < kSize * kSize; ++i)
        o[f * kSize * kSize + i] = frames_[f][i] / 255.f;
    return o;
  }
  /// @brief Borrow the newest resized byte frame.
  ///
  /// Throws std::logic_error when empty. Caller synchronizes against mutation.
  ///
  /// @return Reference with7056 bytes; invalidated by clear/destruction and potentially by later
  /// pushes.
  /// @see PushFrame()
  const std::vector<unsigned char>& last_frame() const {
    if (frames_.empty())
      throw std::logic_error("no preprocessed frame");
    return frames_.back();
  }

 private:
  int                                    raw_w_;  ///< Configured screen width.
  int                                    raw_h_;  ///< Configured screen height.
  std::deque<std::vector<unsigned char>> frames_; ///< Owned recent resized screens.
};

} // namespace datamunge::rl::environments
