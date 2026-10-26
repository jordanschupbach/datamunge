"""Atari DQN preprocessing and ROM lookup (examples/nn/common/rl/ale_env.hpp).

ALE is not a datamunge dependency, and the ports have no emulator binding, so this helper
carries the emulator-free part of the C++ header: AtariFrameStack (max-pool of the last two
frames, box-filter resize to 84x84, a stack of the last 4 frames, scaling to [0, 1]) and the
ROM search path. HAVE_ALE is False: the AtariEnv class itself is not available.
"""
import os
from collections import deque
from pathlib import Path

from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32 = rl_env.f32

HAVE_ALE = False


def rom_dirs():
    """Directories searched for <game>.bin, in order."""
    dirs = ["data/roms"]
    if os.environ.get("TFLOW_DATA_DIR"):
        dirs.append(os.environ["TFLOW_DATA_DIR"] + "/roms")
    if os.environ.get("ALE_ROM_DIR"):
        dirs.append(os.environ["ALE_ROM_DIR"])
    if os.environ.get("HOME"):
        dirs.append(os.environ["HOME"] + "/projects/tflow_examples/data/roms")
    return dirs


def find_rom(game):
    for directory in rom_dirs():
        path = directory + "/" + game + ".bin"
        if Path(path).exists():
            return path
    raise RuntimeError("ROM " + game + ".bin not found in data/roms/ or $ALE_ROM_DIR")


class AtariOptions:
    def __init__(self):
        self.frame_skip = 4
        self.clip_rewards = False
        self.noop_max = 30
        self.max_frames = 27000
        self.repeat_action_probability = f32(0.25)


class AtariFrameStack:
    """The DQN frame pipeline over raw grayscale screens of raw_w x raw_h (lists of 0..255)."""

    SIZE, STACK = 84, 4

    def __init__(self, raw_w=160, raw_h=210):
        self.raw_w, self.raw_h = raw_w, raw_h
        self.frames = deque()
        size = AtariFrameStack.SIZE
        # The box filter's source rows/columns per output pixel (fixed for a screen size).
        self.row_spans = [(y * raw_h // size, max(y * raw_h // size + 1, (y + 1) * raw_h // size)) for y in range(size)]
        self.col_spans = [(x * raw_w // size, max(x * raw_w // size + 1, (x + 1) * raw_w // size)) for x in range(size)]

    def clear(self):
        self.frames.clear()

    @staticmethod
    def max_pool(cur, prev):
        """Max over the last two frames removes Atari flicker (prev may be empty); in place."""
        if prev:
            for i in range(len(cur)):
                if prev[i] > cur[i]:
                    cur[i] = prev[i]

    def push_frame(self, gray):
        """Box-filter resize of the grayscale screen to 84x84."""
        small = []
        w = self.raw_w
        for y0, y1 in self.row_spans:
            rows = [gray[yy * w:(yy + 1) * w] for yy in range(y0, y1)]
            for x0, x1 in self.col_spans:
                total = sum(sum(row[x0:x1]) for row in rows)
                small.append(total // ((y1 - y0) * (x1 - x0)))
        self.frames.append(small)
        while len(self.frames) > AtariFrameStack.STACK:
            self.frames.popleft()

    def obs_size(self):
        return AtariFrameStack.STACK * AtariFrameStack.SIZE * AtariFrameStack.SIZE

    def stacked_obs(self):
        obs = []
        for frame in self.frames:
            obs.extend(f32(v / 255.0) for v in frame)
        return obs

    def last_frame(self):
        return self.frames[-1]
