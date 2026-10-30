# Native Python NN example. Rollout "video" of the hand-written 6.1 controllers.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/6_deep_reinforcement_learning/6.1_environments/render_video.py
"""Rollout "video" of the hand-written 6.1 controllers.

Port of tflow's render_video. tflow renders every built-in environment's 3D
world offscreen with vsim's NVRHI/Vulkan renderer and streams the frames,
with a HUD overlay, into ffmpeg (output/videos/*.mp4). vsim rendering and
ffmpeg video writing are out of scope for the datamunge ports, so this port
plays exactly the episodes tflow films (same controllers, same reset seeds,
same Hud bookkeeping; common/rl/playback.hpp) and prints each one's steps
and return, which match tflow's "recorded ..." lines. The only frames
rendered are Snake's, whose board is itself an image: they are written as a
PNG filmstrip to output/nn/frames/snake_greedy.png
(common/rl/snake_frames.hpp). Checks are added.
"""
import os

from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_hud = helper("rl_hud")
rl_playback = helper("rl_playback")
rl_classic_control = helper("rl_classic_control")
rl_lunar_lander = helper("rl_lunar_lander")
rl_snake = helper("rl_snake")
rl_snake_frames = helper("rl_snake_frames")
f32, fma = rl_env.f32, rl_env.fma


def cartpole_controller(obs):
    lean = fma(0.5, obs[3], obs[2])
    return rl_hud.note(1 if lean > 0 else 0, "theta + 0.5 theta_dot = " + rl_hud.hud_signed(lean))


def pendulum_swing_up(obs):
    theta, thdot = rl_env.atan2f(obs[1], obs[0]), obs[2]
    if abs(theta) < 0.5 and abs(thdot) < 3.0:
        u = max(-2.0, min(2.0, fma(-8.0, theta, -2.0 * thdot)))
    else:
        energy = fma(5.0, f32(obs[0] - 1.0), f32(f32(thdot * thdot) / 6.0))
        u = (2.0 if energy < 0 else -2.0) * (1.0 if thdot >= 0 else -1.0)
    return [u]


def lunar_pilot(obs):
    x, y, vx, vy, angle, ang_vel = obs[:6]
    angle_todo = fma(f32(max(f32(-0.4), min(f32(0.4), fma(f32(0.6), x, vx))) - angle), 2.0, -ang_vel)
    hover_todo = fma(f32(0.5 * abs(x) - y), 0.5, -vy)
    if hover_todo > abs(angle_todo) and hover_todo > f32(0.05):
        return 2
    if angle_todo < f32(-0.05):
        return 3
    if angle_todo > f32(0.05):
        return 1
    return 0


def snake_greedy(env):
    dr, dc = (-1, 0, 1, 0), (0, 1, 0, -1)

    def policy(obs):
        hr, hc = env.head()
        fr, fc = env.food()
        vertical, horizontal = (0 if fr < hr else 2), (3 if fc < hc else 1)
        prefer = [vertical, horizontal] if abs(fr - hr) >= abs(fc - hc) else [horizontal, vertical]
        for d in prefer + [0, 1, 2, 3]:
            r, c = hr + dr[d], hc + dc[d]
            if 0 <= r < env.height() and 0 <= c < env.width() and not env.occupied(r, c):
                return d
        return prefer[0]

    return policy


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    print("== Playing hand-written controllers (tflow records them to output/videos/; video omitted)")
    print("-- CartPole: push toward the lean")
    cart = rl_playback.play_episode(rl_classic_control.CartPole(), cartpole_controller, "cartpole_controller", 5)
    print("-- Pendulum: energy pumping, then a linear regulator near the top")
    pend = rl_playback.play_episode(rl_classic_control.Pendulum(), pendulum_swing_up, "pendulum_swing_up", 2)
    print("-- LunarLander: the heuristic pilot (hover over the pad, keep upright)")
    lander_env = rl_lunar_lander.LunarLander()
    lander = rl_playback.play_episode(lander_env, lunar_pilot, "lunar_lander_heuristic", 4)
    landed = lander_env.state().landed
    print("-- Snake: greedy toward the food, avoiding the body")
    strip = rl_snake_frames.SnakeFilmstrip()
    snake_env = rl_snake.Snake(8, 8)
    snake = rl_playback.play_episode(snake_env, snake_greedy(snake_env), "snake_greedy", 3,
                                     lambda e, hud: strip.add(e))
    png = checks.output_dir("frames") / "snake_greedy.png"
    strip.save(png)
    print("  wrote %s (%d frames)" % (png, strip.frames()))

    checks.require(cart.steps == 500 and cart.ret == 500.0, "the CartPole controller should balance for all 500 steps")
    checks.require(pend.steps == 200 and pend.ret > -300.0, "the pendulum swing-up should score above -300")
    checks.require(landed and lander.ret > 200.0, "the heuristic pilot should land with a return above 200")
    checks.require(snake.ret > 10.0 and strip.frames() == snake.steps + 1,
                   "greedy Snake should eat > 10 food and film every step")
    print("\nchecks passed (vsim/ffmpeg videos omitted; Snake frames written as a PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
