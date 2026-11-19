#pragma once

/// @file linkage_replay.hpp
/// @brief Native replay of classic-control linkage trajectories.

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/cad/video.hpp"

namespace datamunge::rl::environments {

/// @brief Select the mechanism and geometry of a linkage replay.
/// @par Example
/// @code
/// const auto mode = dm::rl::environments::LinkageReplay::CartPole;
/// @endcode
/// @see save_linkage_replay()
enum class LinkageReplay {
  CartPole, ///< Translating pivot on a rail, with one pole.
  Pendulum, ///< Fixed pivot and one swinging link.
  Acrobot   ///< Fixed pivot and two swinging links.
};

/// @brief Encode a recorded classic-control trajectory as an H.264 MP4.
///
/// Writes one frame per sample, without mutating an environment. CartPole
/// angles are measured clockwise from upright, Pendulum angles from upright,
/// and Acrobot angles from hanging down. Requires ffmpeg with libx264 on PATH.
/// @param mechanism Mechanism and observation convention.
/// @param samples Nonempty chronological finite triples. CartPole uses
/// [cart_x,theta,0], Pendulum [0,theta,0], and Acrobot [0,theta1,theta2] where
/// theta2 is relative to the first link. Angles are in radians.
/// @param destination MP4 path with an existing parent directory.
/// @param fps Frames per second, in [1,240]. Defaults to 30.
/// @throws std::invalid_argument Malformed samples, mechanism or encoder geometry.
/// @throws std::runtime_error Rendering or encoding failure.
/// @par Example
/// @code
/// namespace e = dm::rl::environments;
/// e::save_linkage_replay(e::LinkageReplay::Pendulum, {{0,0,0},{0,.2f,0}}, "swing.mp4");
/// @endcode
/// @see cad::VideoWriter
inline void save_linkage_replay(LinkageReplay mechanism,
                                const std::vector<std::array<float, 3>>& samples,
                                const std::string& destination, int fps = 30) {
  const bool cart = mechanism == LinkageReplay::CartPole;
  const bool acrobot = mechanism == LinkageReplay::Acrobot;
  if (samples.empty() || (!cart && !acrobot && mechanism != LinkageReplay::Pendulum))
    throw std::invalid_argument("linkage replay requires a mechanism and samples");
  for (const auto& sample : samples)
    for (float value : sample)
      if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite linkage pose");

  const auto box = [](cad::Scene& scene, cad::Vec3f position, cad::Vec3f scale, cad::RgbaColorF color) {
    const int id = cad::AddPrimitiveToScene(&scene, cad::PrimitiveKind::Cube);
    auto& object = *scene.FindObject(id);
    object.position = position;
    object.scale = scale;
    object.color = color;
    return id;
  };
  cad::Scene background;
  if (cart)
    box(background, {0,0,-.16f}, {5.4f,.4f,.14f}, {.31f,.37f,.46f,1});
  else
    box(background, {}, {.17f,.24f,.17f}, {.26f,.33f,.44f,1});
  if (acrobot)
    box(background, {0,.22f,1}, {4.4f,.04f,.035f}, {.21f,.63f,.37f,1});
  const cad::PreparedScene scenery(std::move(background));
  cad::Scene actors;
  const int carriage = cart ? box(actors, {}, {.56f,.35f,.25f}, {.15f,.47f,.72f,1}) : -1;
  const int first = box(actors, {}, {.10f,.15f,1}, {.91f,.45f,.12f,1});
  const int second = acrobot ? box(actors, {}, {.10f,.15f,1}, {.19f,.54f,.84f,1}) : -1;
  cad::Camera camera;
  camera.target = {0,0,cart ? .65 : 0.};
  camera.position = {0,-8.,camera.target.z + .7};
  camera.projection = cad::Projection::Orthographic;
  camera.ortho_height = cart ? 3.6 : acrobot ? 4.6 : 2.6;
  cad::RenderOptions options;
  options.width = 480;
  options.height = 300;
  options.threads = 2;
  options.shadows = false;
  cad::VideoWriter video(destination, options.width, options.height, fps);
  constexpr float pi = std::numbers::pi_v<float>;
  for (const auto& [x, first_angle, second_angle] : samples) {
    if (cart) actors.FindObject(carriage)->position = {x,0,.05f};
    const auto link = [&](int id, cad::Vec3f pivot, float theta) {
      const float dx = std::sin(theta), dz = std::cos(theta);
      auto& object = *actors.FindObject(id);
      object.position = {pivot.x + .5f * dx,0,pivot.z + .5f * dz};
      object.rotation_deg.y = theta * (180.f / pi);
      return cad::Vec3f{pivot.x + dx,0,pivot.z + dz};
    };
    const auto elbow = link(first, {cart ? x : 0,0,cart ? .2f : 0},
                            acrobot ? pi - first_angle : first_angle);
    if (acrobot) link(second, elbow, pi - first_angle - second_angle);
    video.append(scenery.render(actors, camera, options));
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
