#pragma once

/// @file grid_replay.hpp
/// @brief Native 3D replay for tabular grid environments.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/cad/video.hpp"
#include "datamunge/rl/environments/grid_world.hpp"

namespace datamunge::rl::environments {

/// @brief Encode the visited states of a GridWorld or CliffWalking episode as an H.264 MP4.
///
/// The board is built once with raised walls, a green goal marker and red
/// cliff cells. An orange token represents each visited state in sequence.
/// A jump back to the start is visible on the very next frame. The environment
/// is read without mutation. The first and last frames receive a one-second
/// hold. This CPU renderer needs ffmpeg with libx264 on PATH.
/// @param env Board geometry. The environment need not be reset.
/// @param states Nonempty episode states in chronological order, including the reset state.
/// @param destination MP4 path with an existing parent directory.
/// @param fps Integer output rate and moves per second, from 1 to 240.
/// @throws std::invalid_argument Invalid state, empty replay or invalid output parameters.
/// @throws std::runtime_error Rendering or encoder failure.
/// @par Example
/// @code
/// namespace e = dm::rl::environments;
/// e::CliffWalking board;
/// e::save_grid_replay(board, {36,24,25,36}, "cliff.mp4", 5);
/// @endcode
/// @see cad::VideoWriter, CliffWalking, GridWorld
inline void save_grid_replay(const GridBase& env, const std::vector<int>& states,
                             const std::string& destination, int fps = 5) {
  if (states.empty())
    throw std::invalid_argument("grid replay requires at least one state");
  for (int state : states)
    if (state < 0 || state >= env.num_states() || env.IsWall(state / env.width(), state % env.width()))
      throw std::invalid_argument("grid replay state must be a traversable cell");

  const auto box = [](cad::Scene& scene, cad::Vec3f position, cad::Vec3f scale,
                      cad::RgbaColorF color) {
    const int id = cad::AddPrimitiveToScene(&scene, cad::PrimitiveKind::Cube);
    auto* object = scene.FindObject(id);
    object->position = position;
    object->scale = scale;
    object->color = color;
    return id;
  };

  const auto* cliff = dynamic_cast<const CliffWalking*>(&env);
  cad::Scene board;
  for (int r = 0; r < env.height(); ++r) {
    for (int c = 0; c < env.width(); ++c) {
      const float x = c, y = -r;
      const cad::RgbaColorF color = cliff && cliff->IsCliff(r, c)
          ? cad::RgbaColorF{.83f,.15f,.2f,1} : cad::RgbaColorF{.55f,.65f,.76f,1};
      box(board, {x,y,-.1f}, {.96f,.96f,.18f}, color);
      if (env.IsWall(r, c))
        box(board, {x,y,.36f}, {.88f,.88f,.74f}, {.23f,.3f,.39f,1});
      if (env.IsGoal(r, c)) {
        box(board, {x+.2f,y,.3f}, {.06f,.06f,.6f}, {.2f,.36f,.28f,1});
        box(board, {x+.02f,y,.52f}, {.36f,.09f,.23f}, {.12f,.72f,.32f,1});
      }
    }
  }
  cad::RenderOptions options;
  options.width = 640;
  options.height = 400;
  options.threads = 2;
  cad::Camera camera;
  camera.target = {(env.width()-1)/2., -(env.height()-1)/2., 0.};
  camera.position = {camera.target.x + .5, camera.target.y - 7., 13.};
  camera.projection = cad::Projection::Orthographic;
  camera.ortho_height = std::max(env.height() + 2., (env.width() + 2.) * options.height / options.width);
  const cad::PreparedScene scenery(std::move(board));
  cad::Scene actors;
  const int agent = box(actors, {}, {.58f,.58f,.54f}, {1,.63f,.08f,1});
  cad::VideoWriter video(destination, options.width, options.height, fps);
  for (std::size_t i = 0; i < states.size(); ++i) {
    const auto [r, c] = env.Coords(states[i]);
    actors.FindObject(agent)->position = {float(c), -float(r), .27f};
    const auto frame = scenery.render(actors, camera, options);
    const int holds = i == 0 || i == states.size() - 1 ? fps : 1;
    for (int j = 0; j < holds; ++j) video.append(frame);
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
