#pragma once

/// @file lunar_lander_replay.hpp
/// @brief Replay rendered from a recorded LunarLander trajectory.

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/cad/video.hpp"

namespace datamunge::rl::environments {

/// @brief Record the position and attitude of a simulated LunarLander above its landing pad.
///
/// Writes one frame per recorded pose and fits a fixed camera around the
/// trajectory. Requires ffmpeg with libx264 on PATH.
/// @param poses Nonempty chronological [x,y,angle] triples in model units and radians.
/// @param destination MP4 path with an existing parent directory.
/// @param fps Frame rate from 1 to 240, default 30.
/// @throws std::invalid_argument Empty trajectory, nonfinite pose or invalid encoder settings.
/// @throws std::runtime_error Rendering or encoder failure.
/// @par Example
/// @code
/// dm::rl::environments::save_lunar_lander_replay({{0,1,0},{0,0,0}}, "landing.mp4");
/// @endcode
/// @see LunarLander, cad::VideoWriter
inline void save_lunar_lander_replay(const std::vector<std::array<float, 3>>& poses,
                                     const std::string& destination, int fps = 30) {
  if (poses.empty())
    throw std::invalid_argument("empty LunarLander trajectory");
  double extent = 1.2, bottom = 0, top = 1.5;
  for (const auto& [x,y,angle] : poses) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(angle))
      throw std::invalid_argument("nonfinite LunarLander pose");
    extent = std::max(extent, std::abs(double(x)));
    bottom = std::min(bottom, double(y));
    top = std::max(top, double(y));
  }
  const auto box = [](cad::Scene& scene, cad::Vec3f pos, cad::Vec3f size, cad::RgbaColorF color) {
    const int id = cad::AddPrimitiveToScene(&scene, cad::PrimitiveKind::Cube);
    auto& object = *scene.FindObject(id);
    object.position = pos;
    object.scale = size;
    object.color = color;
    return id;
  };
  cad::Scene scene;
  box(scene, {0,0,-.09f}, {float(2. * (extent + 1.)),.54f,.14f}, {.32f,.36f,.41f,1});
  box(scene, {0,0,-.018f}, {.5f,.49f,.035f}, {.21f,.71f,.38f,1});
  const cad::PreparedScene terrain(std::move(scene));
  cad::Scene craft;
  const int cabin = box(craft, {}, {.24f,.20f,.21f}, {.22f,.5f,.86f,1});
  const int base = box(craft, {}, {.29f,.21f,.075f}, {.86f,.65f,.24f,1});
  const int left = box(craft, {}, {.05f,.18f,.10f}, {.42f,.48f,.56f,1});
  const int right = box(craft, {}, {.05f,.18f,.10f}, {.42f,.48f,.56f,1});
  cad::Camera camera;
  camera.target = {0,0,(top+bottom)/2.};
  camera.position = {0,-6.,camera.target.z + .65};
  camera.projection = cad::Projection::Orthographic;
  camera.ortho_height = std::max(top-bottom+.65,(extent+.35)*1.25);
  cad::RenderOptions options;
  options.width = 480;
  options.height = 300;
  options.threads = 2;
  options.shadows = false;
  cad::VideoWriter video(destination, options.width, options.height, fps);
  for (const auto& [x,y,angle] : poses) {
    constexpr float pi = std::numbers::pi_v<float>;
    const float tilt = std::remainder(angle, 2.f * pi);
    const float c = std::cos(tilt), s = std::sin(tilt);
    const auto place = [&](int id, float horizontal, float vertical) {
      auto& object = *craft.FindObject(id);
      object.position = {x + horizontal*c - vertical*s, 0, y + horizontal*s + vertical*c};
      object.rotation_deg.y = -tilt * (180.f/pi);
    };
    place(cabin, 0,.235f);
    place(base, 0,.12f);
    place(left, -.12f,.05f);
    place(right, .12f,.05f);
    video.append(terrain.render(craft, camera, options));
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
