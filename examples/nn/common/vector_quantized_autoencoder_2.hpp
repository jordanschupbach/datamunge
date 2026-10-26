// Port of tflow's common/nn/vector_quantized_autoencoder_2.h and
// common/ml/vector_quantized_autoencoder_2.h: two-scale VQ-VAE-2 (top 7x7 and
// top-conditioned bottom 14x14 codebooks, discrete-only decoding) and the fitting-only
// bottom|top conditional raster Markov prior.
#pragma once

#include "common/vector_quantized_autoencoder.hpp"

namespace ae {

/// top codes [B 49, 4] -> 7x7x4 -> repeat2 -> Conv3 4->4 ReLU -> [B, 14, 14, 4].
inline nn::Tensor Vq2TopDecoder(nn::ParameterStore& p, const nn::Tensor& top) {
  const std::int64_t n = top.size(0) / 49;
  return nn::conv2d(p, "top_decoder", nn::upsample_nearest(nn::reshape(top, {n, 7, 7, 4}), 2), 4, 3, VqRelu());
}

/// concat(bottom 14x14x4, top feature) -> Conv3 8->8 ReLU -> repeat2 -> Conv3 8->1 -> [B, 784] logits.
inline nn::Tensor Vq2Decoder(nn::ParameterStore& p, const nn::Tensor& bottom, const nn::Tensor& top_feature) {
  const std::int64_t n        = bottom.size(0) / 196;
  auto               combined = nn::concat({nn::reshape(bottom, {n, 14, 14, 4}), top_feature}, 3);
  auto               h        = nn::conv2d(p, "decoder", combined, 8, 3, VqRelu());
  return nn::reshape(nn::conv2d(p, "output", nn::upsample_nearest(h, 2), 1, 3, nn::ConvLayerOptions{}), {n, 784});
}

struct VectorQuantizedAutoencoder2Output {
  nn::Tensor encoded_top, encoded_bottom, top_book, bottom_book, top_feature, probability, nll, reconstruction_loss, loss;
  VectorQuantizedOutput top, bottom;
};

/// Conv3 1->4 ReLU/avg2 (h, 14x14), Conv3 4->4 ReLU/avg2 -> top 7x7 codes; top ST code -> top
/// feature; 1x1 projection of concat(h, top feature) -> bottom 14x14 codes. `top_indices`
/// (int32, B 49 values) overrides the top assignment (tflow's EncodeWithTop feed).
inline VectorQuantizedAutoencoder2Output VectorQuantizedAutoencoder2(nn::ParameterStore& p, const nn::Tensor& image,
                                                                     const nn::Tensor* top_indices = nullptr) {
  VectorQuantizedAutoencoder2Output o;
  const std::int64_t                n = image.size(0);
  auto h        = nn::avg_pool2d(nn::conv2d(p, "encoder1", image, 4, 3, VqRelu()), 2, 2);
  o.encoded_top = nn::reshape(nn::avg_pool2d(nn::conv2d(p, "encoder2", h, 4, 3, VqRelu()), 2, 2), {n * 49, 4});
  o.top_book    = p.get("top_codebook", {16, 4}, nn::init::uniform(.5));
  o.top         = VectorQuantize(o.encoded_top, o.top_book, 16, 4, top_indices);
  o.top_feature = Vq2TopDecoder(p, o.top.straight_through);
  o.encoded_bottom = nn::reshape(
      nn::conv2d(p, "bottom_projection", nn::concat({h, o.top_feature}, 3), 4, 1, nn::ConvLayerOptions{}), {n * 196, 4});
  o.bottom_book         = p.get("bottom_codebook", {16, 4}, nn::init::uniform(.5));
  o.bottom              = VectorQuantize(o.encoded_bottom, o.bottom_book);
  auto logits           = Vq2Decoder(p, o.bottom.straight_through, o.top_feature);
  o.probability         = nn::sigmoid(logits);
  o.nll                 = VaePixelNll(logits, nn::reshape(image, {n, 784}));
  o.reconstruction_loss = nn::mean(o.nll, {0});
  o.loss = o.reconstruction_loss + ((o.top.codebook_loss + o.bottom.codebook_loss) +
                                    .25 * (o.top.commitment_loss + o.bottom.commitment_loss));
  return o;
}

class VectorQuantizedModel2 {
 public:
  nn::ParameterStore             parameters{2801};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit VectorQuantizedModel2(bool training = true) {
    {
      nn::NoGradGuard guard;
      VectorQuantizedAutoencoder2(parameters, nn::zeros({1, 28, 28, 1}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  nn::NamedTensors Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = VectorQuantizedAutoencoder2(parameters, images);
    return VqFinite({{"encoded_top", n.encoded_top},
                     {"encoded_bottom", n.encoded_bottom},
                     {"top_indices", n.top.indices},
                     {"bottom_indices", n.bottom.indices},
                     {"top_quantized", n.top.quantized},
                     {"bottom_quantized", n.bottom.quantized},
                     {"top_straight_through", n.top.straight_through},
                     {"bottom_straight_through", n.bottom.straight_through},
                     {"top_feature", n.top_feature},
                     {"probability", n.probability},
                     {"nll", n.nll},
                     {"reconstruction_loss", n.reconstruction_loss},
                     {"top_codebook_loss", n.top.codebook_loss},
                     {"bottom_codebook_loss", n.bottom.codebook_loss},
                     {"top_commitment_loss", n.top.commitment_loss},
                     {"bottom_commitment_loss", n.bottom.commitment_loss},
                     {"loss", n.loss}});
  }
  /// Two-grid target-free decoder: top int32 [B, 49], bottom int32 [B, 196].
  nn::Tensor Decode(const nn::Tensor& top, const nn::Tensor& bottom) {
    check::Require(top.dim() == 2 && bottom.dim() == 2 && top.size(0) == bottom.size(0),
                   "topBx49 and bottomBx196 discrete grids");
    VqRequireIds(top, 49, "topBx49 and bottomBx196 discrete grids");
    VqRequireIds(bottom, 196, "topBx49 and bottomBx196 discrete grids");
    nn::NoGradGuard guard;
    auto            top_feature = Vq2TopDecoder(parameters, VqLookup(top, parameters.at("top_codebook")));
    return nn::sigmoid(Vq2Decoder(parameters, VqLookup(bottom, parameters.at("bottom_codebook")), top_feature))
        .contiguous();
  }
  static nn::Tensor IndexGrid(const nn::Tensor& ids, std::int64_t batch, std::int64_t tokens) {
    return nn::reshape(ids, {batch, tokens}).contiguous();
  }
  /// d mean(encoded_bottom^2)/d encoded_top and the (zero) reconstruction derivatives of both codebooks.
  nn::NamedTensors HierarchyGradients(const nn::Tensor& images) {
    auto n        = VectorQuantizedAutoencoder2(parameters, images);
    auto coupling = nn::grad(nn::mean(nn::square(n.encoded_bottom), {0, 1}), {n.encoded_top}, {true, false, false});
    auto probe    = n.reconstruction_loss + 0.0 * (nn::sum(n.top_book, {0, 1}) + nn::sum(n.bottom_book, {0, 1}));
    auto books    = nn::grad(probe, {n.top_book, n.bottom_book});
    return {{"bottom_encoded_square_dtop_encoded", coupling[0]},
            {"reconstruction_dtop_book", books[0]},
            {"reconstruction_dbottom_book", books[1]}};
  }
  /// Bottom encoding with a supplied discrete top grid.
  nn::NamedTensors EncodeWithTop(const nn::Tensor& images, const nn::Tensor& top) {
    nn::NoGradGuard guard;
    auto            n = VectorQuantizedAutoencoder2(parameters, images, &top);
    return {{"encoded_bottom", n.encoded_bottom.contiguous()}, {"bottom_indices", n.bottom.indices}};
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images) {
    return nn::grad(VectorQuantizedAutoencoder2(parameters, images).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images) {
    auto         loss  = VectorQuantizedAutoencoder2(parameters, images).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite VQ objective");
    optimizer->minimize(loss);
    return value;
  }
};

/// p(bottom code | parent-aligned top code, previous bottom code (BOS 16)), pseudocount .5.
struct Vq2ConditionalPrior {
  std::vector<int>    count = std::vector<int>(16 * 17 * 16, 0);
  std::vector<double> probability = std::vector<double>(16 * 17 * 16, 0);

  static int Parent(int position) { return ((position / 14) / 2) * 7 + (position % 14) / 2; }
  static int At(int context, int previous, int code) { return (context * 17 + previous) * 16 + code; }

  Vq2ConditionalPrior(const nn::Tensor& top, const nn::Tensor& bottom) {
    check::Require(top.size(0) == bottom.size(0) && top.size(1) == 49 && bottom.size(1) == 196,
                   "paired coarse/fine grids");
    const auto t = top.contiguous().to_vector<int>(), w = bottom.contiguous().to_vector<int>();
    for (std::int64_t b = 0; b < bottom.size(0); ++b) {
      int previous = 16;
      for (int i = 0; i < 196; ++i) {
        const int context = t[b * 49 + Parent(i)], code = w[b * 196 + i];
        ++count[At(context, previous, code)];
        previous = code;
      }
    }
    for (int context = 0; context < 16; ++context)
      for (int previous = 0; previous < 17; ++previous) {
        double sum = 8;
        for (int code = 0; code < 16; ++code) sum += count[At(context, previous, code)];
        for (int code = 0; code < 16; ++code)
          probability[At(context, previous, code)] = (count[At(context, previous, code)] + .5) / sum;
      }
  }
  nn::Tensor counts() const { return nn::from_vector(count, {16, 17, 16}); }
  nn::Tensor probabilities() const { return nn::from_vector(probability, {16, 17, 16}); }
  std::pair<nn::Tensor, nn::Tensor> Sample(const nn::Tensor& top, std::uint32_t seed) const {
    const int           n = int(top.size(0));
    const auto          t = top.contiguous().to_vector<int>();
    std::mt19937        random(seed);
    std::vector<int>    ids(std::size_t(n) * 196);
    std::vector<double> uniform(ids.size());
    for (int b = 0; b < n; ++b) {
      int previous = 16;
      for (int i = 0; i < 196; ++i) {
        const double u       = (double(random()) + .5) / 4294967296.;
        uniform[b * 196 + i] = u;
        const int context    = t[b * 49 + Parent(i)];
        int       selected   = 15;
        double    cumulative = 0;
        for (int k = 0; k < 16; ++k) {
          cumulative += probability[At(context, previous, k)];
          if (u < cumulative) {
            selected = k;
            break;
          }
        }
        ids[b * 196 + i] = selected;
        previous         = selected;
      }
    }
    return {nn::from_vector(ids, {n, 196}), nn::from_vector(uniform, {n, 196})};
  }
  double Nll(const nn::Tensor& top, const nn::Tensor& bottom) const {
    const auto t = top.contiguous().to_vector<int>(), w = bottom.contiguous().to_vector<int>();
    double     sum = 0;
    for (std::int64_t b = 0; b < bottom.size(0); ++b) {
      int previous = 16;
      for (int i = 0; i < 196; ++i) {
        const int context = t[b * 49 + Parent(i)], code = w[b * 196 + i];
        sum -= std::log(probability[At(context, previous, code)]);
        previous = code;
      }
    }
    return sum / double(bottom.size(0));
  }
};

}  // namespace ae
