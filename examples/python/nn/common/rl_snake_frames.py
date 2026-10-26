"""Snake boards as RGB frames and a PNG filmstrip (examples/nn/common/rl/snake_frames.hpp).

One pixel per cell (head red, body green, food gold, empty dark); an episode's frames are
tiled into one PNG with nn.save_image_grid.
"""
from pydatamunge import nn


def snake_frame_rgb(env):
    """The board as height x width x 3 floats in [0, 1]."""
    h, w = env.height(), env.width()
    img = [0.0] * (h * w * 3)

    def paint(r, c, red, green, blue):
        if 0 <= r < h and 0 <= c < w:
            p = (r * w + c) * 3
            img[p:p + 3] = [red, green, blue]

    for r in range(h):
        for c in range(w):
            paint(r, c, 0.08, 0.09, 0.12)
    paint(env.food()[0], env.food()[1], 1.0, 0.77, 0.27)
    body = list(env.body())
    for r, c in body[1:]:
        paint(r, c, 0.30, 0.75, 0.40)
    if body:
        paint(body[0][0], body[0][1], 0.92, 0.25, 0.25)
    return img


class SnakeFilmstrip:
    """Collects frames of one board size and writes them as a grid PNG."""

    def __init__(self):
        self.h = self.w = self._frames = 0
        self.pixels = []

    def add(self, env):
        self.h, self.w = env.height(), env.width()
        self.pixels.extend(snake_frame_rgb(env))
        self._frames += 1

    def frames(self):
        return self._frames

    def tensor(self):
        """[frames, height, width, 3] float32."""
        return nn.from_vector(self.pixels, [self._frames, self.h, self.w, 3])

    def save(self, path, cols=16, scale=6):
        nn.save_image_grid(self.tensor(), str(path), cols, 0.0, 1.0, 1, 0.5, scale)
