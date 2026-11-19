#pragma once

/// @file mountain_car_replay.hpp
/// @brief Native animated replay of discrete and continuous Mountain Car trajectories.

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/cad/video.hpp"
#include "datamunge/rl/environments/classic_control.hpp"

namespace datamunge::rl::environments {

/// @brief Render a recorded Mountain Car rollout as an H.264 MP4.
///
/// The car moves over a sinusoidal hill and aligns with the local slope.
/// Uses one frame per recorded position, including the reset position if
/// supplied. The caller owns the trajectory and no simulation state changes.
/// Requires ffmpeg with libx264 on PATH.
/// @param positions Nonempty chronological finite position observations in [-1.2, 0.6].
/// @param destination MP4 path with an existing parent directory.
/// @param goal Finite flag position inside the track. Defaults to 0.5 for discrete control.
/// @param fps Output rate in frames per second, from 1 to 240. Defaults to 24.
/// @throws std::invalid_argument Invalid position, goal, trajectory or output parameters.
/// @throws std::runtime_error Rendering or encoder failure.
/// @par Example
/// @code
/// namespace e = dm::rl::environments;
/// e::MountainCar car;
/// const auto initial = car.reset(1);
/// const auto next = car.step(2);
/// e::save_mountain_car_replay({initial[0], next.obs[0]}, "hill.mp4");
/// @endcode
/// @see MountainCar, MountainCarContinuous, cad::VideoWriter
inline void save_mountain_car_replay(const std::vector<float>& positions,
                                    const std::string& destination, float goal = .5f,
                                    int fps = 24) {
  const auto in_track = [](float x) { return std::isfinite(x) && x >= -1.2f && x <= .6f; };
  if (positions.empty() || !in_track(goal))
    throw std::invalid_argument("mountain car replay needs positions and an in-track goal");
  for (float value : positions)
    if (!in_track(value))
      throw std::invalid_argument("mountain car replay position is outside the track");

  const auto height = [](float x) { return 1.5f * (1.f + std::sin(3.f * x)); };
  const auto box = [](cad::Scene& scene, cad::Vec3f position, cad::Vec3f size,
                      cad::RgbaColorF color) {
    const int id = cad::AddPrimitiveToScene(&scene, cad::PrimitiveKind::Cube);
    auto& object = *scene.FindObject(id);
    object.position = position;
    object.scale = size;
    object.color = color;
    return id;
  };
  const auto degrees = [](float radians) { return radians * (180.f / std::numbers::pi_v<float>); };

  cad::Scene hill;
  constexpr int segments = 56;
  for (int i = 0; i < segments; ++i) {
    const float a = -1.25f + 1.9f * i / segments;
    const float b = -1.25f + 1.9f * (i + 1) / segments;
    const float dx = 4.f * (b - a), dz = height(b) - height(a);
    const float angle = std::atan2(dz, dx);
    const int id = box(hill, {2.f * (a + b), 0, .5f * (height(a) + height(b)) - .08f},
                       {std::hypot(dx, dz) + .015f, .95f, .18f}, {.2f, .57f, .3f, 1});
    hill.FindObject(id)->rotation_deg.y = -degrees(angle);
  }
  box(hill, {4.f * goal, 0, height(goal) + .38f}, {.04f,.04f,.76f}, {.25f,.28f,.34f,1});
  box(hill, {4.f * goal + .14f, 0, height(goal) + .67f}, {.28f,.045f,.18f}, {.92f,.25f,.22f,1});
  const cad::PreparedScene scenery(std::move(hill));

  cad::Scene car;
  const int body = box(car, {}, {.46f,.33f,.20f}, {.15f,.4f,.85f,1});
  const int cabin = box(car, {}, {.23f,.25f,.16f}, {.56f,.76f,.94f,1});
  const int front = box(car, {}, {.13f,.38f,.13f}, {.13f,.17f,.22f,1});
  const int back = box(car, {}, {.13f,.38f,.13f}, {.13f,.17f,.22f,1});
  cad::Camera camera;
  camera.target = {-1.2,0,1.7};
  camera.position = {-1.2,-11.,5.1};
  camera.projection = cad::Projection::Orthographic;
  camera.ortho_height = 5.;
  cad::RenderOptions options;
  options.width = 480;
  options.height = 270;
  options.threads = 2;
  options.shadows = false;
  cad::VideoWriter video(destination, options.width, options.height, fps);
  for (float x : positions) {
    const float angle = std::atan2(4.5f * std::cos(3.f * x), 4.f);
    const float cosine = std::cos(angle), sine = std::sin(angle);
    const auto place = [&](int id, float along, float up) {
      auto& object = *car.FindObject(id);
      object.position = {4.f * x + along * cosine - up * sine, 0,
                         height(x) + along * sine + up * cosine};
      object.rotation_deg.y = -degrees(angle);
    };
    place(body, 0, .21f);
    place(cabin, -.035f, .37f);
    place(front, .15f, .075f);
    place(back, -.15f, .075f);
    video.append(scenery.render(car, camera, options));
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
