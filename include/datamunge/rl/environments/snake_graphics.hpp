// datamunge::nn port helper: Snake boards as RGB frames and a PNG filmstrip.
//
// tflow films Snake on a vsim 3D board (vsim_worlds.h SnakeWorld) and streams it into ffmpeg;
// both are out of scope for the datamunge ports. Instead the board is drawn as a small RGB
// image (one pixel per cell: head red, body green, food gold, empty dark) and an episode's
// frames are tiled into one PNG with nn::save_image_grid.
#pragma once
/// @file snake_graphics.hpp
/// @brief Native snake graphics simulation components for explicit caller-owned rollouts.

#include <datamunge/rl/environments/snake.hpp>
#include <string>
#include <vector>

#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::rl::environments {

// The board as height x width x 3 floats in [0, 1].
/// @brief Copy a snake board to row-major RGB Float32 pixels in[0,1].
/// @param env Reset native Snake instance; its state is read without mutation.
/// @return Owning height*width*3 RGB Float32 values in[0,1].
///
/// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references expire
/// at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid explicit
/// bounds/actions throw std::invalid_argument or std::out_of_range before their checked access.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Snake board;board.reset(1);auto rgb=e::SnakeFrameRgb(board);
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, StepResult
inline std::vector<float> SnakeFrameRgb(const Snake& env) {
  const int h =
      env.height(); ///< Owned source simulation state; units and geometry follow the owning class.
  const int w =
      env.width(); ///< Owned source simulation state; units and geometry follow the owning class.
  std::vector<float> img(static_cast<size_t>(h) * w * 3, 0.f);
  auto               paint = [&](int r, int c, float R, float G, float B) {
    if (r < 0 || r >= h || c < 0 || c >= w)
      return;
    float* p = &img[(static_cast<size_t>(r) * w + c) * 3];
    p[0] = R, p[1] = G, p[2] = B;
  };
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c)
      paint(r, c, 0.08f, 0.09f, 0.12f);
  paint(env.food().first, env.food().second, 1.f, 0.77f, 0.27f);
  const auto& body = env.body();
  for (size_t i = 1; i < body.size(); ++i)
    paint(body[i].first, body[i].second, 0.30f, 0.75f, 0.40f);
  if (!body.empty())
    paint(body.front().first, body.front().second, 0.92f, 0.25f, 0.25f);
  return img;
}

// Collects frames of one board size and writes them as a grid PNG.
/// @brief Own the SnakeFilmstrip simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Snake board;board.reset(1);e::SnakeFilmstrip
/// obj;obj.Add(board);
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class SnakeFilmstrip {
 public:
  /// @brief Append one independently copied snake RGB frame; subsequent frames must share board
  /// geometry.
  /// @param env Reset native Snake instance; its state is read without mutation.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake board;board.reset(1);e::SnakeFilmstrip
  /// obj;obj.Add(board);obj.Add(board);
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Add(const Snake& env) {
    if (frames_ && (h_ != env.height() || w_ != env.width()))
      throw std::invalid_argument("filmstrip board geometry changed");
    h_ = env.height(), w_ = env.width();
    const std::vector<float> f = SnakeFrameRgb(env);
    pixels_.insert(pixels_.end(), f.begin(), f.end());
    ++frames_;
  }
  /// @brief Return the number of captured frames.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake board;board.reset(1);e::SnakeFilmstrip
  /// obj;obj.Add(board);auto value=obj.frames();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int frames() const { return frames_; }
  // [frames, height, width, 3] float32.
  /// @brief Copy captured RGB pixels into an owning Float32 NHWC tensor; an empty filmstrip is
  /// rejected.
  /// @return Owning Float32 CPU[frames,height,width,3] tensor.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::logic_error Required reset/captured-frame state is absent.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake board;board.reset(1);e::SnakeFilmstrip
  /// obj;obj.Add(board);auto value=obj.tensor();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  datamunge::nn::Tensor tensor() const {
    if (frames_ == 0)
      throw std::logic_error("empty snake filmstrip");
    return datamunge::nn::from_vector(pixels_, {frames_, h_, w_, 3});
  }
  /// @brief Write captured frames through the native image-grid PNG API.
  /// @param path Destination PNG filename; image-writer errors propagate.
  /// @param cols Positive board column count, at most1024.
  /// @param scale Positive integer pixel enlargement, default6.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake board;board.reset(1);e::SnakeFilmstrip
  /// obj;obj.Add(board);std::filesystem::create_directories("build");obj.Save("build/public-snake.png");
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Save(const std::string& path, int cols = 16, int scale = 6) const {
    datamunge::nn::save_image_grid(
        tensor(), path, cols, 0.0, 1.0, /*padding=*/1, /*pad_value=*/0.5, scale);
  }

 private:
  int h_ = 0; ///< Owned snake/filmstrip board height.
  int w_ = 0; ///< Owned snake/filmstrip board width or contextual weight matrix, according to its

  ///< declared type.
  int                frames_ = 0; ///< Captured frame count.
  std::vector<float> pixels_;     ///< Owned concatenated NHWC RGB pixels in[0,1].
};

} // namespace datamunge::rl::environments
