// Port of tflow's common/nn/vector_quantized_autoencoder.h and
// common/ml/vector_quantized_autoencoder.h: VQ-VAE with nearest learned vectors,
// exact-forward straight-through routing and separate codebook/commitment terms;
// a compact spatial model with code-index-only decoding; and the fitting-only
// smoothed raster first-order Markov prior over code grids.
#pragma once

#include <random>
#include <utility>

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

// ---- vector quantizer ---------------------------------------------------------------------

struct VectorQuantizedOutput {
  nn::Tensor indices;  // int32 [N]
  nn::Tensor assignment, quantized, straight_through, codebook_loss, commitment_loss;
};

/// encoded [N, d], book [K, d]: nearest code (squared distance |e|^2 + |b|^2 - 2 e.b,
/// lowest index on ties), q = onehot(argmin) book, ST = stop(q) + (e - stop(e)) (exactly q
/// forward, identity derivative to e), dictionary = mean (stop(e) - q)^2 (codebook only),
/// commitment = mean (e - stop(q))^2 (encoder only). `override_indices` (int32 [N]) replaces
/// the nearest assignment, as feeding tflow's indices tensor does.
inline VectorQuantizedOutput VectorQuantize(const nn::Tensor& encoded, const nn::Tensor& book, std::int64_t entries = 16,
                                            std::int64_t dimensions = 4,
                                            const nn::Tensor* override_indices = nullptr) {
  check::Require(entries > 0 && dimensions > 0, "positive VQ vocabulary and dimensions");
  VectorQuantizedOutput o;
  auto distance = nn::sum(nn::square(encoded), {1}, true) + nn::expand_dims(nn::sum(nn::square(book), {1}), 0) -
                  2.0 * nn::matmul(encoded, book, false, true);
  o.indices = override_indices ? nn::reshape(*override_indices, {encoded.size(0)}).contiguous()
                               : nn::cast(nn::argmin(nn::stop_gradient(distance), 1), nn::DType::Int32).contiguous();
  o.assignment       = nn::one_hot(o.indices, entries, 1.0, 0.0);
  o.quantized        = nn::matmul(o.assignment, book);
  o.straight_through = nn::stop_gradient(o.quantized) + (encoded - nn::stop_gradient(encoded));
  o.codebook_loss    = nn::mean(nn::square(nn::stop_gradient(encoded) - o.quantized), {0, 1});
  o.commitment_loss  = nn::mean(nn::square(encoded - nn::stop_gradient(o.quantized)), {0, 1});
  return o;
}

/// Codebook lookup of int32 ids [B, T] -> [B T, d].
inline nn::Tensor VqLookup(const nn::Tensor& ids, const nn::Tensor& book, std::int64_t entries = 16) {
  return nn::matmul(nn::reshape(nn::one_hot(ids, entries, 1.0, 0.0), {-1, entries}), book);
}

inline nn::ConvLayerOptions VqRelu() {
  nn::ConvLayerOptions relu;
  relu.init       = nn::init::he_uniform();
  relu.activation = nn::Activation::Relu;
  return relu;
}

/// codes [B 49, 4] -> 7x7x4 -> repeat2 -> Conv3 4->8 ReLU -> repeat2 -> Conv3 8->1 -> [B, 784] logits.
inline nn::Tensor VqDecoder(nn::ParameterStore& p, const nn::Tensor& codes) {
  const std::int64_t n = codes.size(0) / 49;
  auto h = nn::conv2d(p, "decoder1", nn::upsample_nearest(nn::reshape(codes, {n, 7, 7, 4}), 2), 8, 3, VqRelu());
  return nn::reshape(nn::conv2d(p, "output", nn::upsample_nearest(h, 2), 1, 3, nn::ConvLayerOptions{}), {n, 784});
}

struct VectorQuantizedAutoencoderOutput {
  nn::Tensor            encoded, book, probability, nll, reconstruction_loss, loss;
  VectorQuantizedOutput vq;
};

/// image [N, 28, 28, 1]: Conv3 1->8 ReLU/avg2, Conv3 8->4 ReLU/avg2 -> 7x7 grid of 4-vectors,
/// 16x4 codebook U(-.5,.5), loss = mean NLL + dictionary + .25 commitment.
inline VectorQuantizedAutoencoderOutput VectorQuantizedAutoencoder(nn::ParameterStore& p, const nn::Tensor& image) {
  VectorQuantizedAutoencoderOutput o;
  const std::int64_t               n = image.size(0);
  auto h    = nn::avg_pool2d(nn::conv2d(p, "encoder1", image, 8, 3, VqRelu()), 2, 2);
  o.encoded = nn::reshape(nn::avg_pool2d(nn::conv2d(p, "encoder2", h, 4, 3, VqRelu()), 2, 2), {n * 49, 4});
  o.book    = p.get("codebook", {16, 4}, nn::init::uniform(.5));
  o.vq      = VectorQuantize(o.encoded, o.book);
  auto logits           = VqDecoder(p, o.vq.straight_through);
  o.probability         = nn::sigmoid(logits);
  o.nll                 = VaePixelNll(logits, nn::reshape(image, {n, 784}));
  o.reconstruction_loss = nn::mean(o.nll, {0});
  o.loss                = o.reconstruction_loss + (o.vq.codebook_loss + .25 * o.vq.commitment_loss);
  return o;
}

/// Contiguous copies; every float value must be finite.
inline nn::NamedTensors VqFinite(nn::NamedTensors fields) {
  for (auto& [name, t] : fields) {
    t = t.contiguous();
    if (t.dtype() == nn::DType::Float32)
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all VQ outputs finite");
  }
  return fields;
}

/// Bx49 int32 ids, every one in 0..15.
inline void VqRequireIds(const nn::Tensor& ids, std::int64_t tokens, const std::string& what) {
  check::Require(ids.dtype() == nn::DType::Int32 && ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) == tokens, what);
  for (int v : ids.contiguous().to_vector<int>()) check::Require(v >= 0 && v < 16, "every VQ code index lies in0..15");
}

class VectorQuantizedModel {
 public:
  nn::ParameterStore             parameters{2707};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit VectorQuantizedModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      VectorQuantizedAutoencoder(parameters, nn::zeros({1, 28, 28, 1}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// tflow's Evaluate fields (indices int32 [B 49]).
  nn::NamedTensors Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = VectorQuantizedAutoencoder(parameters, images);
    return VqFinite({{"encoded", n.encoded},
                     {"indices", n.vq.indices},
                     {"quantized", n.vq.quantized},
                     {"straight_through", n.vq.straight_through},
                     {"probability", n.probability},
                     {"nll", n.nll},
                     {"reconstruction_loss", n.reconstruction_loss},
                     {"codebook_loss", n.vq.codebook_loss},
                     {"commitment_loss", n.vq.commitment_loss},
                     {"loss", n.loss}});
  }
  /// Target-free decoder of int32 code ids [B, 49].
  nn::Tensor Decode(const nn::Tensor& indices) {
    VqRequireIds(indices, 49, "VQ decoder indices have Bx49 shape");
    nn::NoGradGuard guard;
    return nn::sigmoid(VqDecoder(parameters, VqLookup(indices, parameters.at("codebook")))).contiguous();
  }
  static nn::Tensor IndexGrid(const nn::Tensor& indices, std::int64_t batch) {
    return nn::reshape(indices, {batch, 49}).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images) {
    return nn::grad(VectorQuantizedAutoencoder(parameters, images).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images) {
    auto         loss  = VectorQuantizedAutoencoder(parameters, images).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite VQ objective");
    optimizer->minimize(loss);
    return value;
  }
};

// ---- fitting-only raster Markov prior over code grids --------------------------------------

/// Codes per context (BOS = 16) with Dirichlet pseudocount .5: p = (count + .5) / (sum + 8).
struct VqMarkovPrior {
  std::vector<int>    count = std::vector<int>(17 * 16, 0);
  std::vector<double> probability = std::vector<double>(17 * 16, 0);

  explicit VqMarkovPrior(const nn::Tensor& fitting) {
    const auto ids = fitting.contiguous().to_vector<int>();
    const auto n = fitting.size(0), t_max = fitting.size(1);
    for (std::int64_t b = 0; b < n; ++b) {
      int previous = 16;
      for (std::int64_t t = 0; t < t_max; ++t) {
        const int code = ids[b * t_max + t];
        check::Require(code >= 0 && code < 16, "fitting code inrange");
        ++count[previous * 16 + code];
        previous = code;
      }
    }
    for (int context = 0; context < 17; ++context) {
      double sum = 8;
      for (int code = 0; code < 16; ++code) sum += count[context * 16 + code];
      for (int code = 0; code < 16; ++code)
        probability[context * 16 + code] = (count[context * 16 + code] + .5) / sum;
    }
  }
  nn::Tensor counts() const { return nn::from_vector(count, {17, 16}); }
  nn::Tensor probabilities() const { return nn::from_vector(probability, {17, 16}); }
  /// Sequential raster samples (int32 ids [n, tokens], float64 uniforms) from mt19937(seed).
  std::pair<nn::Tensor, nn::Tensor> Sample(int n, std::uint32_t seed, int tokens = 49) const {
    std::mt19937        random(seed);
    std::vector<int>    ids(std::size_t(n) * tokens);
    std::vector<double> uniform(ids.size());
    for (int b = 0; b < n; ++b) {
      int context = 16;
      for (int t = 0; t < tokens; ++t) {
        const double u = (double(random()) + .5) / 4294967296.;
        uniform[b * tokens + t] = u;
        double cumulative = 0;
        int    selected   = 15;
        for (int k = 0; k < 16; ++k) {
          cumulative += probability[context * 16 + k];
          if (u < cumulative) {
            selected = k;
            break;
          }
        }
        ids[b * tokens + t] = selected;
        context             = selected;
      }
    }
    return {nn::from_vector(ids, {n, tokens}), nn::from_vector(uniform, {n, tokens})};
  }
  /// Mean nats per grid.
  double Nll(const nn::Tensor& grid) const {
    const auto ids = grid.contiguous().to_vector<int>();
    const auto n = grid.size(0), t_max = grid.size(1);
    double     out = 0;
    for (std::int64_t b = 0; b < n; ++b) {
      int context = 16;
      for (std::int64_t t = 0; t < t_max; ++t) {
        const int code = ids[b * t_max + t];
        out -= std::log(probability[context * 16 + code]);
        context = code;
      }
    }
    return out / double(n);
  }
};

/// Code counts (int32 [16]) and perplexity exp(entropy) of an id grid.
inline std::pair<nn::Tensor, double> VqUsage(const nn::Tensor& ids) {
  std::vector<int> counts(16, 0);
  const auto       v = ids.contiguous().to_vector<int>();
  for (int code : v) {
    check::Require(code >= 0 && code < 16, "valid VQ assignment");
    ++counts[code];
  }
  double entropy = 0;
  for (int k = 0; k < 16; ++k) {
    const double p = double(counts[k]) / double(v.size());
    if (p > 0) entropy -= p * std::log(p);
  }
  return {nn::from_vector(counts, {16}), std::exp(entropy)};
}

/// Max absolute difference of two equally shaped float tensors.
inline double VqDifference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "same tensor shape");
  return MaxAbsDiff(a, b, a.numel());
}

}  // namespace ae
