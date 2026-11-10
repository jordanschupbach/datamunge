"""Tiny asymmetric masked autoencoder (include/datamunge/nn/masked_autoencoder.hpp).

He et al. MAE on 16 non-overlapping 7x7 MNIST patches: a visible-only width-32
transformer encoder (4 of 16 tokens), a width-16 mask-token decoder, fixed 2-D
sin/cos positions and a hidden-only loss. Parameter names match tflow's. Model
pieces, hash-derived patch selections and native metrics only: training steps
(forward, mae_loss, nn.grad / optimizer) are written out in each example.
"""

import math

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

ae = helper("autoencoder")
require = ae.require


mae_position = nn.MaePosition
mae_block = nn.MaeBlock
mae_patches = nn.MaePatches
masked_autoencoder = nn.MaskedAutoencoder
mae_loss = nn.MaeLoss


class MaskedAutoencoderOutput:
    """Compatibility value for restoration helpers that fill encoder/decoder fields."""
    patches = visible = mask = code = prediction = encoder_attention = decoder_attention = None


def mae_encode(p, image, selector, out):
    result = nn.MaeEncode(p, image, selector)
    out.patches, out.visible, out.mask = result.patches, result.visible, result.mask
    out.code, out.encoder_attention = result.code, result.encoder_attention


def mae_decode(p, code, selector, out):
    result = nn.MaeDecode(p, code, selector)
    out.prediction, out.decoder_attention = result.prediction, result.decoder_attention


def _initial_selector():
    one = [0.0] * 64
    for k in range(4):
        one[k * 16 + k] = 1.0
    return nn.from_vector(one, [1, 4, 16])


class MaskedAutoencoderModel:
    """Parameters (seed 1877) + optional Adam; forward pieces only."""

    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1877)
        with nn.no_grad():
            masked_autoencoder(self.parameters, nn.zeros([1, 28, 28, 1]), _initial_selector())
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def forward(self, images, selection):
        return masked_autoencoder(self.parameters, images, selection)

    def loss(self, images, selection):
        n = self.forward(images, selection)
        return mae_loss(n.prediction, n.patches, n.mask)

    def evaluate(self, images, selection):
        """[code [B, 4, 32], prediction [B, 16, 49], hidden-only MSE]"""
        with nn.no_grad():
            n = self.forward(images, selection)
            return [n.code, n.prediction, mae_loss(n.prediction, n.patches, n.mask)]

    def decode(self, codes, selection):
        with nn.no_grad():
            out = MaskedAutoencoderOutput()
            mae_decode(self.parameters, codes, selection, out)
            return out.prediction

    def probe(self, images, selection):
        """[patches, visible, mask, encoder attention, decoder attention]"""
        with nn.no_grad():
            n = self.forward(images, selection)
            return [n.patches, n.visible, n.mask, n.encoder_attention, n.decoder_attention]


def mae_selection_values(ids, epoch=0, seed=1907):
    """Native deterministic selectors as flat values, for existing display helpers."""
    source = nn.cast(nn.from_vector_double(list(ids), [len(ids)]), nn.DType.Int64)
    return list(nn.MaeSelection(source, epoch, seed).values())


mae_selection = nn.MaeSelection
mae_pixel = nn.MaePixel
PIXEL = [[mae_pixel(p, i) for i in range(49)] for p in range(16)]


def mae_image_values(values, n):
    return list(nn.MaeImage(nn.from_vector(list(values), [n, 16, 49])).values())


mae_image = nn.MaeImage


def mae_hidden(selection, b, patch):
    """Whether patch `patch` of example b is hidden (flat [n, 4, 16] selection values)."""
    return all(selection[b * 64 + k * 16 + patch] == 0 for k in range(4))


def mae_mse(prediction, original, selection):
    """Independent native hidden-only MSE."""
    pixels = mae_image(prediction).values()
    o, s = original.contiguous().values(), selection.contiguous().values()
    total, count = 0.0, 0
    for b in range(prediction.size(0)):
        for p in range(16):
            if mae_hidden(s, b, p):
                for j in PIXEL[p]:
                    index = b * 784 + j
                    y = pixels[index]
                    require(math.isfinite(y) and 0 <= y <= 1, "bounded finite patch prediction")
                    e = y - o[index]
                    total += e * e
                    count += 1
    require(count == prediction.size(0) * 12 * 49, "exact75percent hidden loss denominator")
    return total / count
