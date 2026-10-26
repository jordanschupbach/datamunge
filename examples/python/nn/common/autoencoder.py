"""Autoencoder support mirroring examples/nn/common/autoencoder.hpp.

Saliency MNIST pools (1000 fit / 200 validation / 200 test from
$DATAMUNGE_DATA_DIR/saliency/saliency_data.npz), tensor/byte utilities, figure
writers, the convolutional / variational / dense undercomplete autoencoder
components, tflow's MT19937 Box-Muller Gaussian banks and denoising corruption.
Only model pieces and numerical utilities live here: training loops stay in the
example scripts (there is deliberately no update/train method).
"""
import math
import random
import shutil
import struct
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ---- data ------------------------------------------------------------------------------

class SaliencyData:
    """images float [n, 28, 28, 1] in [0, 1]; labels int32 [n]; source ids int32 [n]."""

    def __init__(self, images, labels, source):
        self.images, self.labels, self.source = images, labels, source


def rows(data, indices):
    indices = [int(i) for i in indices]
    return SaliencyData(nn.take_rows(data.images, indices), nn.take_rows(data.labels, indices),
                        nn.take_rows(data.source, indices))


def find(arrays, name):
    """The tensor called `name` in a dict or a list of (name, tensor) pairs."""
    items = arrays.items() if isinstance(arrays, dict) else arrays
    for key, value in items:
        if key == name:
            return value
    raise RuntimeError("missing array " + name)


def load_saliency(root):
    """[fit, validation, test] SaliencyData with tflow's integrity checks."""
    path = Path(root) / "saliency_data.npz"
    require(path.exists(), str(path) + " missing; run `just autoencoder-assets` in tflow_examples or pass data=")
    arrays = nn.load_npz(str(path))
    result, ids, seen = [], set(), set()
    for split in ("fit", "validation", "test"):
        x = find(arrays, split + "_images").contiguous()
        n = 1000 if split == "fit" else 200
        require(list(x.shape()) == [n, 28, 28, 1], "source28x28 grayscale digit images")
        labels = find(arrays, split + "_labels").contiguous()
        source = find(arrays, split + "_source_ids").contiguous()
        require(labels.dtype() == nn.DType.Int32 and source.dtype() == nn.DType.Int32
                and list(labels.shape()) == [n] and list(source.shape()) == [n], "aligned source/label vectors")
        pixels = [int(v) for v in x.values()]
        counts = [0] * 10
        for b, (label, identifier) in enumerate(zip(labels.values(), source.values())):
            label, identifier = int(label), int(identifier)
            require(0 <= label < 10 and identifier not in ids, "source IDs disjoint,labels valid")
            ids.add(identifier)
            require(60000 <= identifier < 70000 if split == "test" else 0 <= identifier < 60000,
                    "original test/train boundary")
            counts[label] += 1
            row = pixels[b * 784:(b + 1) * 784]
            require(all(0 <= v < 256 for v in row), "uint8 pixels")
            key = bytes(row)
            require(key not in seen, "no duplicated input bytes across all pools")
            seen.add(key)
        require(all(c == n // 10 for c in counts), "balanced digit pools")
        result.append(SaliencyData(nn.div(nn.cast(x, nn.DType.Float32), 255.0), labels, source))
    return result


def copy_manifests(data, root):
    """Copies tflow's provenance files next to the outputs when they exist."""
    data, root = Path(data), Path(root)
    if (data / "manifest.json").exists():
        shutil.copyfile(data / "manifest.json", root / "data_manifest.json")
    source = data_root() / "autoencoder" / "source_manifest.json"
    if source.exists():
        shutil.copyfile(source, root / "source_manifest.json")


# ---- tensor helpers ----------------------------------------------------------------------

def tensor_bytes(tensor):
    """Raw float32 bytes, for exact-equality checks."""
    values = tensor.contiguous().values()
    return struct.pack(f"<{len(values)}f", *values)


def max_abs_diff(a, b, count):
    x, y = a.contiguous().values(), b.contiguous().values()
    return max((abs(x[i] - y[i]) for i in range(count)), default=0.0)


def mse(prediction, original):
    """Double-precision MSE; the prediction must be finite in [0, 1]."""
    require(prediction.numel() == original.numel(), "aligned reconstruction pixels")
    p, o = prediction.contiguous().values(), original.contiguous().values()
    total = 0.0
    for x, y in zip(p, o):
        require(math.isfinite(x) and 0 <= x <= 1, "finite bounded sigmoid reconstruction")
        total += (x - y) * (x - y)
    return total / len(p)


def pixel_mean(fit):
    """Per-pixel mean [784] of the fit images (float32 of the double mean)."""
    x = fit.contiguous().values()
    n = fit.size(0)
    out = []
    for i in range(784):
        total = 0.0
        for b in range(n):
            total += x[b * 784 + i]
        out.append(total / n)
    return nn.from_vector(out, [784])


def constant(mean, n):
    """`mean` [784] repeated n times as [n, 784]."""
    return nn.tile(nn.reshape(mean, [1, 784]), [n, 1]).contiguous()


def arrays(store):
    """Deep copies of every parameter, in creation order."""
    return [(name, store.at(name).detach().clone()) for name in store.names()]


def snapshot(store):
    return [tensor_bytes(store.at(name)) for name in store.names()]


def check_gradients(store, gradients, finite, nonzero):
    """Checks every gradient is finite and nonzero; returns them as (name, tensor) pairs."""
    require(len(gradients) == store.size(), "one gradient per parameter")
    out = []
    for name, gradient in zip(store.names(), gradients):
        require(gradient.defined(), nonzero)
        values = gradient.contiguous().values()
        require(all(math.isfinite(v) for v in values), finite)
        require(sum(v * v for v in values) > 0, nonzero)
        out.append((name, gradient))
    return out


# ---- figures ------------------------------------------------------------------------------

def panel_row(path, panels):
    """One row of 28x28 panels scaled 4x, 4 px apart on gray 220 (tflow's digit<k>.png)."""
    tiles = [nn.reshape(p, [28, 28, 1]) for p in panels]
    nn.save_image_grid(nn.stack(tiles, 0), str(path), len(tiles), 0.0, 1.0, 4, 220.0 / 255.0, 4)


def row(x, b):
    """The b-th 784-pixel row of [n, 784] / [n, 28, 28, 1] values."""
    return nn.reshape(nn.slice_axis(x, 0, b, b + 1), [784])


# ---- convolutional autoencoder -------------------------------------------------------------

class ConvAeOutput:
    conv1 = pool1 = conv2 = code = decoder1 = reconstruction = None


def _relu_conv_options():
    options = nn.ConvLayerOptions()
    options.init = nn.init.he_uniform()
    options.activation = nn.Activation.Relu
    return options


def conv_ae_decode(p, code, out=None):
    """nearest-repeat2, Conv3x3 4->8 ReLU, nearest-repeat2, Conv3x3 8->1 sigmoid."""
    d1 = nn.conv2d(p, "decoder1", nn.upsample_nearest(code, 2), 8, 3, _relu_conv_options())
    if out is not None:
        out.decoder1 = d1
    output = nn.ConvLayerOptions()
    output.activation = nn.Activation.Sigmoid
    return nn.conv2d(p, "decoder2", nn.upsample_nearest(d1, 2), 1, 3, output)


def convolutional_autoencoder(p, image):
    """Conv3x3 1->8 ReLU, AvgPool2, Conv3x3 8->4 ReLU, AvgPool2 (7x7x4 code), then the decoder."""
    out = ConvAeOutput()
    out.conv1 = nn.conv2d(p, "encoder1", image, 8, 3, _relu_conv_options())
    out.pool1 = nn.avg_pool2d(out.conv1, 2, 2)
    out.conv2 = nn.conv2d(p, "encoder2", out.pool1, 4, 3, _relu_conv_options())
    out.code = nn.avg_pool2d(out.conv2, 2, 2)
    out.reconstruction = conv_ae_decode(p, out.code, out)
    return out


def make_adam(store, learning_rate=0.003):
    """Adam(.003), the optimiser of every short autoencoder example."""
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    return nn.Optimizer(store, learning_rate, config)


class ConvolutionalAutoencoderModel:
    """Parameters (seed 1801) + optional Adam; forward pieces only."""

    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1801)
        with nn.no_grad():
            convolutional_autoencoder(self.parameters, nn.zeros([1, 28, 28, 1]))
        self.optimizer = make_adam(self.parameters) if training else None

    @staticmethod
    def loss(reconstruction, image):
        return nn.mean(nn.square(nn.sub(reconstruction, image)))

    def forward(self, images):
        return convolutional_autoencoder(self.parameters, images)

    def evaluate(self, images):
        """[code, reconstruction, loss]"""
        with nn.no_grad():
            net = self.forward(images)
            return [net.code, net.reconstruction, self.loss(net.reconstruction, images)]

    def activations(self, images):
        with nn.no_grad():
            n = self.forward(images)
            return [n.conv1, n.pool1, n.conv2, n.code, n.decoder1, n.reconstruction]

    def decode(self, codes):
        with nn.no_grad():
            return conv_ae_decode(self.parameters, codes)


# ---- variational autoencoder ---------------------------------------------------------------

def vae_reparameterize(mu, log_variance, epsilon):
    return nn.add(mu, nn.mul(nn.exp(nn.mul(log_variance, 0.5)), epsilon))


def vae_kl(mu, log_variance):
    """KL(N(mu, exp(lv)) || N(0, I)) per row."""
    inner = nn.sub(nn.sub(nn.add(nn.square(mu), nn.exp(log_variance)), 1.0), log_variance)
    return nn.mul(nn.sum(inner, [1]), 0.5)


def vae_pixel_nll(logits, target):
    """Bernoulli NLL summed over pixels, stable for large |logits|."""
    return nn.sum(nn.sub(nn.softplus(logits), nn.mul(target, logits)), [1])


class VaeOutput:
    mu = log_variance = z = logits = probability = reconstruction_nll = kl = loss = None


def vae_decode_logits(p, z):
    return nn.dense(p, "output", nn.dense(p, "decoder", z, 64, nn.Activation.Relu), 784)


def variational_autoencoder(p, image, epsilon):
    """784 -> 64 ReLU -> (mu, log variance) [2]; z = mu + exp(lv/2) eps; 2 -> 64 ReLU -> 784 logits."""
    out = VaeOutput()
    h = nn.dense(p, "encoder", image, 64, nn.Activation.Relu)
    out.mu = nn.dense(p, "mu", h, 2)
    out.log_variance = nn.dense(p, "log_variance", h, 2)
    out.z = vae_reparameterize(out.mu, out.log_variance, epsilon)
    out.logits = vae_decode_logits(p, out.z)
    out.probability = nn.sigmoid(out.logits)
    out.reconstruction_nll = vae_pixel_nll(out.logits, image)
    out.kl = vae_kl(out.mu, out.log_variance)
    out.loss = nn.mean(nn.add(out.reconstruction_nll, out.kl), [0])
    return out


class VariationalAutoencoderModel:
    """Parameters (seed 1951) + optional Adam; forward pieces only."""

    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1951)
        with nn.no_grad():
            variational_autoencoder(self.parameters, nn.zeros([1, 784]), nn.zeros([1, 2]))
        self.optimizer = make_adam(self.parameters) if training else None

    @staticmethod
    def flatten(images):
        return nn.reshape(images, [images.size(0), 784])

    def forward(self, images, noise):
        return variational_autoencoder(self.parameters, self.flatten(images), noise)

    def evaluate(self, images, noise):
        """[mu, log_variance, z, probability, reconstruction_nll, kl, loss], all checked finite."""
        with nn.no_grad():
            n = self.forward(images, noise)
            out = [t.contiguous() for t in (n.mu, n.log_variance, n.z, n.probability, n.reconstruction_nll,
                                            n.kl, n.loss)]
        for t in out:
            require(all(math.isfinite(v) for v in t.values()), "ALL posterior/output/objective values finite")
        return out

    def decode(self, codes):
        with nn.no_grad():
            return nn.sigmoid(vae_decode_logits(self.parameters, codes))


def mt19937_raw(seed):
    """A random.Random whose getrandbits(32) is std::mt19937(seed)() (init_genrand state)."""
    state = [seed & 0xFFFFFFFF]
    for i in range(1, 624):
        previous = state[-1]
        state.append((1812433253 * (previous ^ (previous >> 30)) + i) & 0xFFFFFFFF)
    generator = random.Random()
    generator.setstate((3, tuple(state) + (624,), None))
    return generator


def gaussian_values(count, seed):
    """tflow's Box-Muller stream from std::mt19937(seed): `count` doubles."""
    rng = mt19937_raw(seed)
    tau = 6.283185307179586476925286766559
    bank = [0.0] * count
    for i in range(0, count, 2):
        u = (rng.getrandbits(32) + 0.5) / 4294967296.0
        v = (rng.getrandbits(32) + 0.5) / 4294967296.0
        radius = math.sqrt(-2 * math.log(u))
        bank[i] = radius * math.cos(tau * v)
        if i + 1 < count:
            bank[i + 1] = radius * math.sin(tau * v)
    return bank


def gaussian_noise(shape, samples, seed):
    """tflow's nn::GaussianNoise as float64 [samples, shape...]."""
    full = [samples] + list(shape)
    count = 1
    for d in full:
        count *= d
    return nn.from_vector_double(gaussian_values(count, seed), full)


class VaeNoiseBank:
    def __init__(self, standard, epsilon):
        self.standard, self.epsilon = standard, epsilon  # float64 / float32 [samples, n, 2]


def vae_noise(n, samples, seed):
    standard = gaussian_noise([n, 2], samples, seed)
    return VaeNoiseBank(standard, nn.cast(standard, nn.DType.Float32).contiguous())


def vae_noise_at(bank, k):
    return nn.select(bank.epsilon, 0, k).contiguous()


def vae_score(model, images, bank):
    """Monte Carlo [reconstruction NLL, KL, negative ELBO] per image over every bank sample."""
    reconstruction = kl = 0.0
    n, samples = images.size(0), bank.epsilon.size(0)
    for k in range(samples):
        out = model.evaluate(images, vae_noise_at(bank, k))
        nll, kls = out[4].values(), out[5].values()
        for b in range(n):
            reconstruction += nll[b]
            if k == 0:
                kl += kls[b]
    reconstruction /= n * samples
    kl /= n
    return [reconstruction, kl, reconstruction + kl]


# ---- dense undercomplete autoencoder -------------------------------------------------------

def flatten(images):
    """[n, 28, 28, 1] (or any n x 784 values) as [n, 784]."""
    return nn.reshape(images, [images.size(0), 784])


class DenseAeOutput:
    def __init__(self, code, reconstruction):
        self.code, self.reconstruction = code, reconstruction


def undercomplete_autoencoder(p, x, features=784, latent=32):
    """"encoder" features -> latent ReLU, "decoder" latent -> features sigmoid."""
    require(features > 1 and 0 < latent < features, "strictly undercomplete positive bottleneck")
    code = nn.dense(p, "encoder", x, latent, nn.Activation.Relu)
    return DenseAeOutput(code, nn.dense(p, "decoder", code, features, nn.Activation.Sigmoid))


def reconstruction_mse(prediction, original):
    """tflow's ReconstructionMse: mean over every element."""
    return nn.mean(nn.square(nn.sub(prediction, original)))


def corrupt_for_denoising(clean, standard_noise, sigma):
    """clamp(float(x + sigma z), 0, 1) with z one float64 bank [1, shape...]."""
    sigma = struct.unpack("f", struct.pack("f", sigma))[0]
    require(clean.dtype() == nn.DType.Float32 and clean.dim() >= 2 and standard_noise.dtype() == nn.DType.Float64
            and list(standard_noise.shape()) == [1] + list(clean.shape()) and math.isfinite(sigma) and sigma >= 0,
            "aligned single Gaussian bank and nonnegative sigma")
    x, z = clean.contiguous().values(), standard_noise.contiguous().values()
    out = []
    for a, b in zip(x, z):
        require(math.isfinite(a) and 0 <= a <= 1 and math.isfinite(b), "finite unit-range clean pixels/noise")
        out.append(a + sigma * b)
    # float32 rounding first (from_vector), then the clamp in float32.
    return nn.clip(nn.from_vector(out, list(clean.shape())), 0.0, 1.0)


def save_data(path, pools, extra=()):
    """tflow's data.npz record: the three pools (+ any extra (name, tensor) pairs)."""
    items = [("fit_images", pools[0].images), ("validation_images", pools[1].images),
             ("test_images", pools[2].images), ("fit_source_ids", pools[0].source),
             ("validation_source_ids", pools[1].source), ("test_source_ids", pools[2].source),
             ("test_labels", pools[2].labels)]
    nn.save_npz(str(path), items + list(extra))


def first_of_digit(labels, digit):
    for b, label in enumerate(labels.contiguous().values()):
        if int(label) == digit:
            return b
    raise RuntimeError(f"no record of digit {digit}")


def digit_gallery(root, test, panels, html):
    """Writes digit<k>.png (row b of every panel) and appends the html sections to the list `html`."""
    source = test.source.contiguous().values()
    for digit in range(10):
        b = first_of_digit(test.labels, digit)
        filename = f"digit{digit}.png"
        panel_row(Path(root) / filename, [row(t, b) for t in panels])
        html.append(f"<section><h2>Digit{digit},source{int(source[b])}</h2><img src='{filename}'></section>")


def same_arrays(a, b):
    """Byte-identical parameter stores (matched-initialisation checks)."""
    if list(a.names()) != list(b.names()):
        return False
    return all(tensor_bytes(a.at(name)) == tensor_bytes(b.at(name)) for name in a.names())


# ---- 2.3 applications: frozen checkpoints and case figures ---------------------------------

class UndercompleteModel:
    """tflow's ml::UndercompleteModel (784->32 ReLU->784 sigmoid, seed 1471) for frozen applications."""

    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1471)
        with nn.no_grad():
            undercomplete_autoencoder(self.parameters, nn.zeros([1, 784]))
        self.optimizer = make_adam(self.parameters) if training else None

    def evaluate(self, images):
        """[code, reconstruction, loss]"""
        with nn.no_grad():
            x = flatten(images)
            n = undercomplete_autoencoder(self.parameters, x)
            return [n.code, n.reconstruction, reconstruction_mse(n.reconstruction, x)]

    def decode(self, codes):
        with nn.no_grad():
            return nn.dense(self.parameters, "decoder", codes, 784, nn.Activation.Sigmoid)


def load_frozen(store, directory, what):
    """Restores best.npz (a training port) or model_arrays.npz (tflow); returns the file used."""
    for name in ("best.npz", "model_arrays.npz"):
        path = Path(directory) / name
        if path.exists():
            store.load(str(path))
            return path
    require(False, f"{what}: no best.npz/model_arrays.npz in {directory} (run the training port first)")


def case_strip3(path, panels, b):
    """tflow's case_<k>.png: row b of every panel, 3x nearest, 4 px gray 220 gaps, RGB."""
    tiles = [nn.reshape(row(p, b), [28, 28, 1]) for p in panels]
    nn.save_image_grid(nn.tile(nn.stack(tiles, 0), [1, 1, 1, 3]), str(path), len(tiles), 0.0, 1.0, 4,
                       220.0 / 255.0, 3)


def mosaic3(path, values, columns):
    """[n, 784] rows, `columns` per row, 3x nearest, 4 px gray 220 gaps, RGB."""
    n = values.size(0)
    nn.save_image_grid(nn.tile(nn.reshape(values, [n, 28, 28, 1]), [1, 1, 1, 3]), str(path), columns, 0.0, 1.0,
                       4, 220.0 / 255.0, 3)
