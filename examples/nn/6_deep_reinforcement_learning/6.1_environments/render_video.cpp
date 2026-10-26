// CMD: just run-nn render_video
// Rollout "video" of the hand-written 6.1 controllers.
//
// Port of tflow's render_video. tflow renders every built-in environment's 3D
// world offscreen with vsim's NVRHI/Vulkan renderer and streams the frames,
// with a HUD overlay, into ffmpeg (output/videos/*.mp4). vsim rendering and
// ffmpeg video writing are out of scope for the datamunge ports, so this port
// plays exactly the episodes tflow films (same controllers, same reset seeds,
// same Hud bookkeeping; common/rl/playback.hpp) and prints each one's steps
// and return, which match tflow's "recorded ..." lines. The only frames
// rendered are Snake's, whose board is itself an image: they are written as a
// PNG filmstrip to output/nn/frames/snake_greedy.png
// (common/rl/snake_frames.hpp). Checks are added.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/lunar_lander.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/snake.hpp"
#include "common/rl/snake_frames.hpp"

int main() {
  try {
    std::printf("== Playing hand-written controllers (tflow records them to output/videos/; video omitted)\n");
    rl::Playback cart, pend, lander, snake;
    {
      std::printf("-- CartPole: push toward the lean\n");
      rl::CartPole env;
      cart = rl::PlayEpisode(env, [](const rl::Obs& o) {
        const float lean = o[2] + 0.5f * o[3];
        return rl::Note(rl::Action(lean > 0 ? 1 : 0), "theta + 0.5 theta_dot = " + rl::HudSigned(lean));
      }, "cartpole_controller", 5);
    }
    {
      std::printf("-- Pendulum: energy pumping, then a linear regulator near the top\n");
      rl::Pendulum env;
      auto swing_up = [](const rl::Obs& o) {
        const float theta = std::atan2(o[1], o[0]), thdot = o[2];
        float u;
        if (std::abs(theta) < 0.5f && std::abs(thdot) < 3.f) u = std::max(-2.f, std::min(2.f, -8.f * theta - 2.f * thdot));
        else u = (thdot * thdot / 6.f + 5.f * (o[0] - 1.f) < 0 ? 2.f : -2.f) * (thdot >= 0 ? 1.f : -1.f);
        return rl::Action({u});
      };
      pend = rl::PlayEpisode(env, swing_up, "pendulum_swing_up", 2);
    }
    bool landed = false;
    {
      std::printf("-- LunarLander: the heuristic pilot (hover over the pad, keep upright)\n");
      rl::LunarLander env;
      auto pilot = [](const rl::Obs& o) {
        const float x = o[0], y = o[1], vx = o[2], vy = o[3], angle = o[4], ang_vel = o[5];
        const float angle_todo = (std::max(-0.4f, std::min(0.4f, 0.6f * x + vx)) - angle) * 2.f - ang_vel;
        const float hover_todo = (0.5f * std::abs(x) - y) * 0.5f - vy;
        if (hover_todo > std::abs(angle_todo) && hover_todo > 0.05f) return 2;
        if (angle_todo < -0.05f) return 3;
        if (angle_todo > 0.05f) return 1;
        return 0;
      };
      lander = rl::PlayEpisode(env, pilot, "lunar_lander_heuristic", 4);
      landed = env.state().landed;
    }
    rl::SnakeFilmstrip strip;
    {
      std::printf("-- Snake: greedy toward the food, avoiding the body\n");
      rl::Snake env(8, 8);
      static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
      auto greedy = [&](const rl::Obs&) {
        auto [hr, hc] = env.head();
        auto [fr, fc] = env.food();
        int prefer[6], n = 0;
        if (std::abs(fr - hr) >= std::abs(fc - hc)) prefer[n++] = fr < hr ? 0 : 2, prefer[n++] = fc < hc ? 3 : 1;
        else prefer[n++] = fc < hc ? 3 : 1, prefer[n++] = fr < hr ? 0 : 2;
        for (int d = 0; d < 4; ++d) prefer[n++] = d;
        for (int i = 0; i < n; ++i) {
          const int r = hr + dr[prefer[i]], c = hc + dc[prefer[i]];
          if (r >= 0 && r < env.height() && c >= 0 && c < env.width() && !env.Occupied(r, c)) return prefer[i];
        }
        return prefer[0];
      };
      snake = rl::PlayEpisode(env, greedy, "snake_greedy", 3, [&](const rl::Snake& e, const rl::Hud&) { strip.Add(e); });
      const std::string png = (paths::OutputDir("frames") / "snake_greedy.png").string();
      strip.Save(png);
      std::printf("  wrote %s (%d frames)\n", png.c_str(), strip.frames());
    }

    check::Require(cart.steps == 500 && cart.ret == 500.f, "the CartPole controller should balance for all 500 steps");
    check::Require(pend.steps == 200 && pend.ret > -300.f, "the pendulum swing-up should score above -300");
    check::Require(landed && lander.ret > 200.f, "the heuristic pilot should land with a return above 200");
    check::Require(snake.ret > 10.f && strip.frames() == snake.steps + 1, "greedy Snake should eat > 10 food and film every step");
    std::cout << "\nchecks passed (vsim/ffmpeg videos omitted; Snake frames written as a PNG)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
