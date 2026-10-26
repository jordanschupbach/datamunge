"""Tiny asymmetric masked autoencoder (examples/nn/common/masked_autoencoder.hpp).

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


def mae_position(width):
    """Fixed 2-D sin/cos positions [16, width]: first half the patch row, second the column."""
    out = [0.0] * (16 * width)
    for patch in range(16):
        for axis in range(2):
            for k in range(width // 4):
                a = (patch % 4 if axis else patch // 4) * math.pow(10000.0, -2.0 * k / (width // 2))
                out[patch * width + axis * width // 2 + k] = math.sin(a)
                out[patch * width + axis * width // 2 + width // 4 + k] = math.cos(a)
    return nn.from_vector(out, [16, width])


class MaeBlockOutput:
    def __init__(self, output, attention):
        self.output, self.attention = output, attention


def mae_block(p, name, x, width):
    """Pre-norm block: x + MHA(LN x) (2 heads, no biases), then + FF(LN .) ratio 2, tanh-GELU."""
    normalized = nn.layer_norm(p, name + "/ln1", x)
    options = nn.MultiHeadOptions()
    options.need_weights = True
    attended = nn.multi_head_attention(p, name + "/attention", normalized, normalized, 2, options)
    residual = nn.add(x, attended.output)
    ff = nn.dense(p, name + "/ff1", nn.layer_norm(p, name + "/ln2", residual), 2 * width, nn.Activation.GeluTanh)
    return MaeBlockOutput(nn.add(residual, nn.dense(p, name + "/ff2", ff, width)), attended.weights)


class MaskedAutoencoderOutput:
    patches = visible = mask = code = prediction = encoder_attention = decoder_attention = None


def mae_patches(image):
    """[B, 28, 28, 1] -> [B, 16, 49] row-major 7x7 patches."""
    b = image.size(0)
    return nn.reshape(nn.transpose(nn.reshape(image, [b, 4, 7, 4, 7]), [0, 1, 3, 2, 4]), [b, 16, 49])


def mae_encode(p, image, selector, out):
    """Encoder half; selector [B, 4, 16] one-hot gather: hidden pixels never enter the encoder."""
    out.patches = mae_patches(image)
    out.visible = nn.matmul(selector, out.patches)
    out.mask = nn.sub(1.0, nn.sum(selector, [1]))
    embedded = nn.dense(p, "patch", out.visible, 32)
    position = nn.matmul(selector, nn.expand_dims(mae_position(32), 0))
    encoder = mae_block(p, "encoder", nn.add(embedded, position), 32)
    out.code = nn.layer_norm(p, "encoder/norm", encoder.output)
    out.encoder_attention = encoder.attention


def mae_decode(p, code, selector, out):
    """Decoder half: from the code and patch locations only (no image/target)."""
    mask = nn.sub(1.0, nn.sum(selector, [1]))
    projected = nn.dense(p, "decoder_projection", code, 16)
    restored = nn.matmul(selector, projected, True)
    token = p.get("mask_token", [16], nn.init.zeros())
    decoder_input = nn.add(nn.add(restored, nn.mul(nn.expand_dims(mask, 2), token)), mae_position(16))
    decoder = mae_block(p, "decoder", decoder_input, 16)
    out.decoder_attention = decoder.attention
    out.prediction = nn.dense(p, "prediction", nn.layer_norm(p, "decoder/norm", decoder.output), 49,
                              nn.Activation.Sigmoid)


def masked_autoencoder(p, image, selector):
    out = MaskedAutoencoderOutput()
    mae_encode(p, image, selector, out)
    mae_decode(p, out.code, selector, out)
    return out


def mae_loss(prediction, patches, mask):
    """Squared error over hidden patches only, / (49 x number of hidden patches)."""
    error = nn.square(nn.sub(prediction, patches))
    return nn.div(nn.sum(nn.mul(error, nn.expand_dims(mask, 2))), nn.mul(nn.sum(mask), 49.0))


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


def mae_mix(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def mae_selection_values(ids, epoch=0, seed=1907):
    """Flat [n, 4, 16] selection values for a list of source ids."""
    out = [0.0] * (len(ids) * 64)
    for b, identifier in enumerate(ids):
        base = seed ^ ((int(identifier) * 0x9E3779B9) & 0xFFFFFFFF) ^ ((epoch * 0x85EBCA6B) & 0xFFFFFFFF)
        keys = sorted((mae_mix(base ^ ((i * 0xC2B2AE35) & 0xFFFFFFFF)), i) for i in range(16))
        for k in range(4):
            out[b * 64 + k * 16 + keys[k][1]] = 1.0
    return out


def mae_selection(source, epoch=0, seed=1907):
    """Reproducible per-example 4-of-16 visible patch selection [n, 4, 16]."""
    ids = [int(v) for v in source.contiguous().values()]
    return nn.from_vector(mae_selection_values(ids, epoch, seed), [len(ids), 4, 16])


def mae_pixel(patch, within):
    """Flat pixel index of element `within` of 7x7 patch `patch`."""
    return ((patch // 4) * 7 + within // 7) * 28 + (patch % 4) * 7 + within % 7


PIXEL = [[mae_pixel(p, i) for i in range(49)] for p in range(16)]


def mae_image_values(values, n):
    out = [0.0] * (n * 784)
    for b in range(n):
        for p in range(16):
            base = (b * 16 + p) * 49
            for i, j in enumerate(PIXEL[p]):
                out[b * 784 + j] = values[base + i]
    return out


def mae_image(patches):
    """[n, 16, 49] patches -> [n, 784] image."""
    n = patches.size(0)
    return nn.from_vector(mae_image_values(patches.contiguous().values(), n), [n, 784])


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
