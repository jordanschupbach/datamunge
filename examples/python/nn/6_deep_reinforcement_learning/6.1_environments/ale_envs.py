# Native Python NN example. Atari through ALE with DQN preprocessing.
"""Atari through ALE with DQN preprocessing.

CMD: just python-example ale_envs

Port of tflow's ale_envs. What a DQN agent sees: every agent step repeats
the action for 4 emulator frames, max-pools the last two to remove flicker,
converts to grayscale, resizes to 84x84 and stacks the last 4 frames so
velocities are visible. This program shows the pipeline on the games
available (Tetris by default, anything in data/roms/), measures throughput,
and checks that motion is visible in the stack.

Deviations: ALE is not a datamunge dependency (the plan keeps emulators out
of scope), so the emulator part only runs when the example is compiled with
ALE on the include path (common/rl/ale_env.hpp, DATAMUNGE_EXAMPLES_HAVE_ALE)
and linked with libale; `just run-nn` builds without it. Without ALE the
program lists the ROMs and runs the same preprocessing code
(rl::AtariFrameStack) on a synthetic 160x210 test screen with a flickering,
moving sprite instead. The video (raw screen next to the 4 stacked frames)
is omitted in both cases. Options: game=<name>.
"""
import os
from pathlib import Path

from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_ale_env = helper("rl_ale_env")
f32 = rl_env.f32
AtariFrameStack = rl_ale_env.AtariFrameStack


def stack_difference(obs):
    """Mean |newest - oldest| over the 84x84 pixels of a stacked observation."""
    pixels = 84 * 84
    total = 0.0
    for i in range(pixels):
        total += abs(f32(obs[3 * pixels + i] - obs[i]))
    return total / pixels


def test_screen(frame):
    """A 160x210 grayscale "screen": dark background, a bright 8x8 sprite moving 3 px per
    emulator frame, drawn only on even frames (Atari-style flicker)."""
    screen = [20] * (160 * 210)
    if frame % 2 == 0:
        x0, y0 = (10 + 3 * frame) % 150, 100
        for y in range(y0, y0 + 8):
            screen[y * 160 + x0:y * 160 + x0 + 8] = [230] * 8
    return screen


def synthetic_pipeline():
    print("== ALE is not built in: the emulator part is skipped (see the header).\n"
          "   The DQN preprocessing runs on a synthetic 160x210 screen instead\n"
          "   (an 8x8 sprite moving 3 px per frame, visible only on even frames).")
    stack = AtariFrameStack(160, 210)
    for _ in range(AtariFrameStack.STACK):
        stack.push_frame(test_screen(0))
    obs = stack.stacked_obs()
    frame = 0
    min_peak_pooled = min_peak_plain = 1.0
    for _ in range(8):
        prev = []
        for k in range(4):  # frame skip 4
            if k == 2:
                prev = test_screen(frame)
            frame += 1
        cur = test_screen(frame - 1)  # the last emulated frame (odd: the sprite is not drawn)
        plain = list(cur)
        AtariFrameStack.max_pool(cur, prev)
        stack.push_frame(cur)
        obs = stack.stacked_obs()
        single = AtariFrameStack(160, 210)
        single.push_frame(plain)
        min_peak_pooled = min(min_peak_pooled, max(stack.last_frame()) / 255.0)
        min_peak_plain = min(min_peak_plain, max(single.last_frame()) / 255.0)
    print("  obs: %d floats = %d frames of 84x84, values in [%.3f, %.3f]"
          % (len(obs), AtariFrameStack.STACK, min(obs), max(obs)))
    print("  brightest pixel of the last frame alone: %.3f, after max-pooling the last two frames: %.3f -> flicker removed"
          % (min_peak_plain, min_peak_pooled))
    diff = stack_difference(obs)
    print("  frame stack difference (newest - oldest, mean |diff| over pixels): %.4f -> motion is visible to the network" % diff)
    checks.require(len(obs) == 4 * 84 * 84, "the observation is 4 stacked 84x84 frames")
    checks.require(min_peak_plain < 0.1 and min_peak_pooled > 0.5, "max-pooling the last two frames should keep the flickering sprite")
    checks.require(diff > 0.001, "the stack should show the sprite's motion")


def main(arguments=None):
    args = options(arguments, dict(game="tetris", full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    roms = ""
    for directory in rl_ale_env.rom_dirs():
        if Path(directory).exists():
            for entry in os.scandir(directory):
                if entry.name.endswith(".bin"):
                    roms += " " + entry.name[:-4]
    print("== available ROMs:" + roms + "   (pass game=<name> to pick one)")
    # No ALE binding: like the C++ built without ALE, the emulator demo for args["game"] is
    # skipped and the preprocessing runs on a synthetic screen.
    assert not rl_ale_env.HAVE_ALE
    synthetic_pipeline()
    print("\nchecks passed (video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
