// datamunge::nn port helper: Snake boards as RGB frames and a PNG filmstrip.
//
// tflow films Snake on a vsim 3D board (vsim_worlds.h SnakeWorld) and streams it into ffmpeg;
// both are out of scope for the datamunge ports. Instead the board is drawn as a small RGB
// image (one pixel per cell: head red, body green, food gold, empty dark) and an episode's
// frames are tiled into one PNG with nn::save_image_grid.
#pragma once

#include <string>
#include <vector>

#include "common/rl/snake.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/tensor.hpp"

namespace rl {

// The board as height x width x 3 floats in [0, 1].
inline std::vector<float> SnakeFrameRgb(const Snake& env) {
  const int h = env.height(), w = env.width();
  std::vector<float> img(static_cast<size_t>(h) * w * 3, 0.f);
  auto paint = [&](int r, int c, float R, float G, float B) {
    if (r < 0 || r >= h || c < 0 || c >= w) return;
    float* p = &img[(static_cast<size_t>(r) * w + c) * 3];
    p[0] = R, p[1] = G, p[2] = B;
  };
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c) paint(r, c, 0.08f, 0.09f, 0.12f);
  paint(env.food().first, env.food().second, 1.f, 0.77f, 0.27f);
  const auto& body = env.body();
  for (size_t i = 1; i < body.size(); ++i) paint(body[i].first, body[i].second, 0.30f, 0.75f, 0.40f);
  if (!body.empty()) paint(body.front().first, body.front().second, 0.92f, 0.25f, 0.25f);
  return img;
}

// Collects frames of one board size and writes them as a grid PNG.
class SnakeFilmstrip {
 public:
  void Add(const Snake& env) {
    h_ = env.height(), w_ = env.width();
    const std::vector<float> f = SnakeFrameRgb(env);
    pixels_.insert(pixels_.end(), f.begin(), f.end());
    ++frames_;
  }
  int frames() const { return frames_; }
  // [frames, height, width, 3] float32.
  datamunge::nn::Tensor tensor() const {
    return datamunge::nn::from_vector(pixels_, {frames_, h_, w_, 3});
  }
  void Save(const std::string& path, int cols = 16, int scale = 6) const {
    datamunge::nn::save_image_grid(tensor(), path, cols, 0.0, 1.0, /*padding=*/1, /*pad_value=*/0.5, scale);
  }

 private:
  int h_ = 0, w_ = 0, frames_ = 0;
  std::vector<float> pixels_;
};

}  // namespace rl
