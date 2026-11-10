#pragma once
/// @file vector_quantized_autoencoder_2.hpp
/// @brief Two-scale discrete VQ inference and an explicit-uniform top-conditioned raster prior.
#include "datamunge/nn/vector_quantized_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Internal two-scale shape, option and code-grid checks.
namespace vq2_detail {
/// @brief Validate image, convolution and two-codebook geometry before leaf creation.
/// @param batch Positive image count.
/// @param height Positive image height divisible by four.
/// @param width Positive image width divisible by four.
/// @param entries Positive Int32 vocabulary size for each dictionary.
/// @param dimensions Positive shared embedding width.
/// @param channels Positive final decoder hidden channels.
/// @return Number of coarse tokens per image; fine tokens are four times this.
/// @throws std::invalid_argument For invalid dimensions or overflowing products.
inline std::int64_t Geometry(
    std::int64_t batch,
    std::int64_t height,
    std::int64_t width,
    std::int64_t entries,
    std::int64_t dimensions,
    std::int64_t channels) {
  const auto top = vq_detail::Geometry(batch, height, width, entries, dimensions, dimensions);
  vq_detail::Geometry(batch, height, width, entries, dimensions, channels);
  vq_detail::Product(vq_detail::Product(9, vq_detail::Product(2, dimensions)), channels);
  vq_detail::Product(vq_detail::Product(2, dimensions), dimensions);
  vq_detail::Product(vq_detail::Product(batch, vq_detail::Product(4, top)), entries);
  return top;
}
/// @brief Require a finite nonnegative representable commitment multiplier.
/// @param beta Finite nonnegative multiplier.
/// @param type Float32 or Float64 model computation dtype.
/// @throws std::invalid_argument For invalid dtype, overflow or positive underflow.
inline void Beta(double beta, DType type) {
  if (!std::isfinite(beta) || beta < 0 || (type != DType::Float32 && type != DType::Float64) ||
      (type == DType::Float32 && (!std::isfinite(float(beta)) || (beta > 0 && float(beta) == 0))))
    throw std::invalid_argument("VQ2 beta must be nonnegative, finite and representable");
}
} // namespace vq2_detail

/// @brief Decode flattened coarse vectors into the shared fine-resolution conditioning feature.
///
/// Reshapes top[B*T,D] to [B,H/4,W/4,D], repeats spatially by two and applies
/// same Conv3 D->D/ReLU with top_decoder/W,b (He uniform/zero bias). T=H*W/16.
/// Defaults create two leaves/148 scalars and return[B,14,14,4]. Codes alone
/// enter; no image, dictionary, target, optimizer or random bank is required.
/// Finite matching Float32/64 codes accept strides and retain continuous first/
/// higher paths. Geometry/input checks precede leaves; incompatible stored
/// leaves can fail later. Inputs unchanged; owning output, no mode change.
/// Work O(B*H*W*9*D*D); synchronize shared-store mutations.
/// @param parameters Mutable matching floating store with shared top_decoder leaves.
/// @param top Finite nonempty flattened coarse vectors[B*T,dimensions].
/// @param height Positive output image height divisible by four, default28.
/// @param width Positive output image width divisible by four, default28.
/// @param dimensions Positive embedding width, default4.
/// @return Owning differentiable coarse-conditioned feature[B,H/2,W/2,D].
/// @throws std::invalid_argument For invalid shapes/dtypes/values/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2801);
/// const auto feature=n::models::Vq2TopDecoder(p,n::zeros({49,4}));
/// // feature is[1,14,14,4]; only two top_decoder leaves are created.
/// @endcode
/// @par References
/// - Razavi, van den Oord and Vinyals (2019), Generating Diverse High-Fidelity
///   Images with VQ-VAE-2, https://arxiv.org/abs/1906.00446 . Compact repeat/Conv
///   teaching variant; the coarse feature conditions both fine encoding and decoding.
/// @see Vq2Decoder(), VectorQuantizedAutoencoder2()
inline Tensor Vq2TopDecoder(
    ParameterStore& parameters,
    const Tensor&   top,
    std::int64_t    height     = 28,
    std::int64_t    width      = 28,
    std::int64_t    dimensions = 4) {
  const auto tokens = vq_detail::Geometry(1, height, width, 1, dimensions, dimensions);
  aae_detail::Matrix(top, dimensions, parameters.dtype(), false);
  if (top.size(0) % tokens)
    throw std::invalid_argument("VQ2 top decoder requires whole coarse grids");
  const auto n = top.size(0) / tokens;
  vq_detail::Geometry(n, height, width, 1, dimensions, dimensions);
  return conv2d(
      parameters,
      "top_decoder",
      upsample_nearest(reshape(top, {n, height / 4, width / 4, dimensions}), 2),
      dimensions,
      3,
      conv_vae_detail::Relu());
}

/// @brief Decode fine code vectors and the coarse conditioning feature to Bernoulli logits.
///
/// Concatenates bottom[B*h*w,D] reshaped to[B,h,w,D] with top_feature[B,h,w,D],
/// then same Conv3 2D->C/ReLU, repeat2, same Conv3 C->1/linear. h,w must be
/// positive even dimensions. Uses decoder/W,b (He/zero) and output/W,b
/// (ordinary Glorot/zero); default four leaves contain657 scalars. Both inputs
/// are code-derived; no encoder/image/target enters. No sigmoid, fitting, random
/// bank, detach or mode change is hidden. Finite matching Float32/64 inputs
/// accept strides and retain all continuous code/feature/leaf paths. Input and
/// product checks precede leaves; incompatible leaves can fail later. Owning
/// output; inputs unchanged. Work O(B*h*w*9*C*(2D+4)); synchronize shared stores.
/// @param parameters Mutable matching floating store containing/creating decoder/output leaves.
/// @param bottom Finite flattened fine vectors[B*h*w,D].
/// @param top_feature Finite matching coarse-derived features[B,h,w,D], h,w even.
/// @param channels Positive decoder hidden channels, default8.
/// @return Owning differentiable logits[B,4*h*w], without probability clipping.
/// @throws std::invalid_argument For invalid shape/dtype/values/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2801);
/// const auto feature=n::models::Vq2TopDecoder(p,n::zeros({49,4}));
/// const auto probability=n::sigmoid(n::models::Vq2Decoder(p,n::zeros({196,4}),feature));
/// // Code-only probability[1,784]; six shared decoder leaves in total.
/// @endcode
/// @par References
/// - Razavi et al. (2019), https://arxiv.org/abs/1906.00446 . Uses a compact
///   two-scale feed-forward decoder, not the paper's full architecture.
/// @see Vq2TopDecoder(), VqLookup(), VectorQuantizedAutoencoder2()
inline Tensor Vq2Decoder(
    ParameterStore& parameters,
    const Tensor&   bottom,
    const Tensor&   top_feature,
    std::int64_t    channels = 8) {
  if (!top_feature.defined() || top_feature.dim() != 4 || top_feature.size(0) <= 0 ||
      top_feature.size(1) <= 0 || top_feature.size(2) <= 0 || top_feature.size(3) <= 0 ||
      top_feature.size(1) % 2 || top_feature.size(2) % 2)
    throw std::invalid_argument("VQ2 top feature needs nonempty [B,even h,even w,D]");
  const auto n = top_feature.size(0), h = top_feature.size(1), w = top_feature.size(2),
             d = top_feature.size(3);
  vq2_detail::Geometry(n, vq_detail::Product(2, h), vq_detail::Product(2, w), 1, d, channels);
  aae_detail::Matrix(bottom, d, parameters.dtype(), false);
  aae_detail::Matrix(
      reshape(top_feature, {n, vq_detail::Product(vq_detail::Product(h, w), d)}),
      h * w * d,
      parameters.dtype(),
      false);
  if (bottom.size(0) != n * h * w)
    throw std::invalid_argument(
        "VQ2 bottom vectors must match coarse-feature batch and fine lattice");
  auto combined = concat({reshape(bottom, {n, h, w, d}), top_feature}, 3);
  auto hidden   = conv2d(parameters, "decoder", combined, channels, 3, conv_vae_detail::Relu());
  return reshape(
      conv2d(parameters, "output", upsample_nearest(hidden, 2), 1, 3, ConvLayerOptions{}),
      {n, 4 * h * w});
}

/// @brief Own both spatial encodings, dictionaries, quantizer records and Bernoulli endpoints.
///
/// All eleven top-level fields own lifetime; top/bottom each own the six native
/// quantizer fields. Intended continuous and surrogate paths remain live. No
/// fitting, prior state, random bank or optimizer is stored in this record.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2801);
/// const auto out=n::models::VectorQuantizedAutoencoder2(p,n::zeros({1,28,28,1}));
/// // top.indices has49 elements; bottom.indices has196; fourteen leaves/1157 scalars.
/// @endcode
/// @see VectorQuantizedAutoencoder2(), VectorQuantizedOutput
struct VectorQuantizedAutoencoder2Output {
  Tensor encoded_top;         ///< Flattened coarse encoder vectors[B*H*W/16,D].
  Tensor encoded_bottom;      ///< Flattened top-conditioned fine vectors[B*H*W/4,D].
  Tensor top_book;            ///< Live top dictionary[K,D], shared with its parameter leaf.
  Tensor bottom_book;         ///< Live independent fine dictionary[K,D].
  Tensor top_feature;         ///< Coarse-derived conditioning[B,H/2,W/2,D].
  Tensor probability;         ///< Sigmoid pixel probabilities[B,H*W], not binary draws.
  Tensor nll;                 ///< Pixel-summed Bernoulli NLL[B].
  Tensor reconstruction_loss; ///< Image-mean pixel NLL scalar.
  Tensor loss; ///< Reconstruction plus two dictionary and beta-weighted commitment penalties.
  VectorQuantizedOutput
      top; ///< Owning coarse quantizer, detached IDs and explicit gradient routes.
  VectorQuantizedOutput bottom; ///< Owning fine quantizer, conditioned on the coarse feature.
};

/// @brief Evaluate the source-preserving two-scale VQ-VAE with an optional discrete top override.
///
/// Same Conv3 1->D/ReLU, avg2 gives fine encoder h; same Conv3 D->D/ReLU,
/// avg2 gives coarse vectors. Top dictionary lookup feeds Vq2TopDecoder().
/// A linear Conv1 projection of concat(h,top_feature) gives fine vectors and
/// an independent fine dictionary. Vq2Decoder() sees only fine quantized
/// vectors and coarse code-derived features. There is no continuous image bypass
/// to decoding; fine ENCODING is deliberately conditioned on top codes.
/// Creation order is encoder1,encoder2,top_codebook,top_decoder,bottom_projection,
/// bottom_codebook,decoder,output (each layer W then b). Dictionaries are U(-.5,.5),
/// ReLU kernels He-uniform, linear kernels ordinary Glorot, biases zero. Default
/// K16/D4/C8 yields14 leaves/1157 scalars. Loss=mean(pixel-summed NLL)+
/// ((top_dictionary+bottom_dictionary)+beta*(top_commitment+bottom_commitment));
/// each dictionary/commitment penalty averages ALL vectors/coordinates at its own
/// resolution. Reconstruction skips both books and copies surrogate gradients
/// to encoders; fine losses can reach the coarse encoder through top_feature.
/// No EMA, resets, Gaussian KL, fitting loop, mode change or random bank is hidden.
/// Parameter creation uses only the caller's initializer stream. Optional valid
/// Int32 top IDs replace assignments without disconnecting their encoder surrogate.
/// Finite matching Float32/64 unit-range grayscale NHWC accepts strides. Width,
/// override and input checks precede leaves; incompatible leaves or computed
/// overflow can fail later. Inputs unchanged, owning outputs. Dominant conv work
/// O(BHW*9*(D^2+D*C+C)); distance work O(BHW*K*D), with O(BHW*K) assignment storage.
/// Synchronize callers mutating a shared store. Surrogate derivatives are not
/// derivatives of hard nearest assignment; higher paths follow the surrogate.
/// @param parameters Mutable matching floating store creating/reusing all fourteen leaves.
/// @param image Finite unit-range nonempty Float32/64[B,H,W,1], H,W divisible by four.
/// @param top_indices Optional valid Int32 tensor with B*H*W/16 elements, any shape; undefined
/// selects nearest.
/// @param entries Positive vocabulary size for each book, default16.
/// @param dimensions Positive embedding width, default4.
/// @param channels Positive final decoder hidden width, default8.
/// @param commitment_beta Finite nonnegative representable multiplier, default0.25.
/// @return Owning eleven-field model record, including two six-field quantizers.
/// @throws std::invalid_argument For invalid input/override/options/products or incompatible
/// leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2801);
/// const auto out=n::models::VectorQuantizedAutoencoder2(p,n::full({2,28,28,1},.3));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // The caller applies all fourteen derivatives; no optimizer step is hidden.
/// @endcode
/// @par References
/// - Razavi, van den Oord and Vinyals (2019), Generating Diverse High-Fidelity
///   Images with VQ-VAE-2, https://arxiv.org/abs/1906.00446 . This compact model
///   uses gradient dictionary updates and Bernoulli pixel NLL, not the paper's
///   EMA dictionary updates, squared pixel loss or full-resolution architecture.
/// @see VectorQuantize(), Vq2TopDecoder(), Vq2Decoder(), Vq2ConditionalPrior
inline VectorQuantizedAutoencoder2Output VectorQuantizedAutoencoder2(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   top_indices     = Tensor{},
    std::int64_t    entries         = 16,
    std::int64_t    dimensions      = 4,
    std::int64_t    channels        = 8,
    double          commitment_beta = .25) {
  conv_vae_detail::Image(image, parameters.dtype());
  const auto n = image.size(0), height = image.size(1), width = image.size(2);
  const auto tokens = vq2_detail::Geometry(n, height, width, entries, dimensions, channels);
  vq2_detail::Beta(commitment_beta, parameters.dtype());
  if (top_indices.defined()) {
    if (top_indices.numel() != n * tokens)
      throw std::invalid_argument("VQ2 top override must have one ID per coarse vector");
    vq_detail::Ids(top_indices, entries);
  }
  VectorQuantizedAutoencoder2Output o;
  auto                              h = avg_pool2d(
      conv2d(parameters, "encoder1", image, dimensions, 3, conv_vae_detail::Relu()), 2, 2);
  o.encoded_top = reshape(
      avg_pool2d(conv2d(parameters, "encoder2", h, dimensions, 3, conv_vae_detail::Relu()), 2, 2),
      {n * tokens, dimensions});
  o.top_book       = parameters.get("top_codebook", {entries, dimensions}, init::uniform(.5));
  o.top            = VectorQuantize(o.encoded_top, o.top_book, entries, dimensions, top_indices);
  o.top_feature    = Vq2TopDecoder(parameters, o.top.straight_through, height, width, dimensions);
  o.encoded_bottom = reshape(
      conv2d(
          parameters,
          "bottom_projection",
          concat({h, o.top_feature}, 3),
          dimensions,
          1,
          ConvLayerOptions{}),
      {n * 4 * tokens, dimensions});
  o.bottom_book = parameters.get("bottom_codebook", {entries, dimensions}, init::uniform(.5));
  o.bottom      = VectorQuantize(o.encoded_bottom, o.bottom_book, entries, dimensions);
  auto logits   = Vq2Decoder(parameters, o.bottom.straight_through, o.top_feature, channels);
  o.probability = sigmoid(logits);
  o.nll         = BernoulliPixelNll(logits, reshape(image, {n, height * width}));
  o.reconstruction_loss = mean(o.nll, {0});
  o.loss                = o.reconstruction_loss +
           ((o.top.codebook_loss + o.bottom.codebook_loss) +
            commitment_beta * (o.top.commitment_loss + o.bottom.commitment_loss));
  return o;
}

/// @brief Fit a fine raster prior conditioned on parent-aligned coarse symbols and the previous
/// fine symbol.
///
/// Coarse grids[B,Ht*Wt] pair with fine grids[B,4*Ht*Wt], interpreted as raster
/// Ht×Wt and 2Ht×2Wt. For fine position(r,c), parent=(r/2)*Wt+c/2 using integer
/// division. Each image starts previous=BOS=K; transitions are counted by
/// [parent_code,previous_fine,current_fine]. Positive Dirichlet smoothing gives
/// p=(count+s)/(context_total+K*s). Defaults K16,Ht=Wt=7,s=0.5 preserve the source.
/// The caller supplies fitting codes only and enforces provenance; no image,
/// neural prior, model update or RNG is involved. This first-order baseline
/// conditions on one aligned top symbol, not the full coarse grid or fine prefix.
/// Construction copies state; exports are independent. Read-only methods are
/// thread-safe. Work O(B*4*Ht*Wt+K^3), storage O(B*4*Ht*Wt+K^3).
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::models::Vq2ConditionalPrior
/// prior(n::zeros({1,1},n::DType::Int32),n::zeros({1,4},n::DType::Int32),1,1,2); const auto
/// ids=prior.Sample(n::zeros({1,1},n::DType::Int32),n::full({1,4},.2,n::DType::Float64));
/// // A four-symbol fine grid drawn from caller-supplied uniforms and one coarse code.
/// @endcode
/// @par References
/// - Razavi et al. (2019), https://arxiv.org/abs/1906.00446 , motivates fitting
///   a fine-code prior conditional on coarse codes. This repository's smoothed
///   parent/previous-symbol baseline is not the paper's neural PixelCNN prior.
/// @see VqMarkovPrior, VectorQuantizedAutoencoder2(), MidpointUniformNoise()
class Vq2ConditionalPrior {
  std::int64_t        top_height_;    ///< Validated positive coarse lattice height.
  std::int64_t        top_width_;     ///< Validated positive coarse lattice width.
  std::int64_t        entries_;       ///< Vocabulary size and BOS previous-symbol index.
  std::int64_t        top_tokens_;    ///< Checked coarse token count per image.
  std::int64_t        bottom_tokens_; ///< Four times the coarse token count.
  std::vector<int>    count_;         ///< Private Int32-safe counts[K,K+1,K], with BOS previous=K.
  std::vector<double> probability_;   ///< Private strictly positive normalized transitions.

  /// @brief Locate the aligned parent of a valid fine raster position.
  /// @param position Fine raster offset in[0,bottom_tokens).
  /// @return Coarse raster offset, using two-by-two parent cells.
  std::int64_t Parent(std::int64_t position) const {
    return ((position / (2 * top_width_)) / 2) * top_width_ + (position % (2 * top_width_)) / 2;
  }
  /// @brief Flatten one validated parent/previous/current transition index.
  /// @param context Coarse symbol in[0,K).
  /// @param previous Fine symbol in[0,K] including BOS=K.
  /// @param code Current fine symbol in[0,K).
  /// @return Row-major index into the checked K*(K+1)*K table.
  std::int64_t At(std::int64_t context, std::int64_t previous, std::int64_t code) const {
    return (context * (entries_ + 1) + previous) * entries_ + code;
  }
  /// @brief Validate a coarse grid's geometry and copy its symbols.
  /// @param top Nonempty valid Int32[B,Ht*Wt] coarse grid.
  /// @return Copied row-major coarse IDs.
  /// @throws std::invalid_argument For malformed shape, dtype or symbols.
  std::vector<int> Top(const Tensor& top) const {
    if (!top.defined() || top.dim() != 2 || top.size(0) <= 0 || top.size(1) != top_tokens_)
      throw std::invalid_argument("VQ2 prior coarse grid has incorrect shape");
    return vq_detail::Ids(top, entries_);
  }
  /// @brief Validate paired fine geometry and copy its symbols.
  /// @param bottom Valid Int32[B,4*Ht*Wt] fine grid.
  /// @param batch Positive matching coarse batch count.
  /// @return Copied row-major fine IDs.
  /// @throws std::invalid_argument For malformed shape, dtype or symbols.
  std::vector<int> Bottom(const Tensor& bottom, std::int64_t batch) const {
    if (!bottom.defined() || bottom.dim() != 2 || bottom.size(0) != batch ||
        bottom.size(1) != bottom_tokens_)
      throw std::invalid_argument("VQ2 prior fine grid has incorrect paired shape");
    return vq_detail::Ids(bottom, entries_);
  }

 public:
  /// @brief Copy paired fitting-code grids into a parent/previous-symbol transition table.
  /// @param top Fitting-only valid Int32[B,top_height*top_width] coarse codes.
  /// @param bottom Matching fitting-only Int32[B,4*top_height*top_width] fine codes.
  /// @param top_height Positive coarse height, default7.
  /// @param top_width Positive coarse width, default7.
  /// @param entries Positive Int32 vocabulary size for both grids, default16.
  /// @param smoothing Finite positive representable pseudocount, default0.5.
  /// @throws std::invalid_argument For invalid options, grids, IDs, counts/products or
  /// unrepresentable smoothing.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::models::Vq2ConditionalPrior
  /// prior(n::zeros({1,1},n::DType::Int32),n::zeros({1,4},n::DType::Int32),1,1,2);
  /// // Parent0/BOS probabilities [.75,.25]; parent0/previous0 probabilities [.875,.125].
  /// @endcode
  /// @see Vq2ConditionalPrior, probabilities()
  Vq2ConditionalPrior(
      const Tensor& top,
      const Tensor& bottom,
      std::int64_t  top_height = 7,
      std::int64_t  top_width  = 7,
      std::int64_t  entries    = 16,
      double        smoothing  = .5)
      : top_height_(top_height),
        top_width_(top_width),
        entries_(entries),
        top_tokens_(vq_detail::Product(top_height, top_width)),
        bottom_tokens_(vq_detail::Product(4, top_tokens_)) {
    vq_detail::Vocabulary(entries);
    const auto cells = vq_detail::Product(vq_detail::Product(entries, entries + 1), entries);
    if (!std::isfinite(smoothing) || smoothing <= 0 ||
        !std::isfinite(double(entries) * smoothing) || !bottom.defined() ||
        bottom.numel() > std::numeric_limits<std::int32_t>::max())
      throw std::invalid_argument(
          "VQ2 prior needs representable positive smoothing and Int32-safe counts");
    const auto t = Top(top), w = Bottom(bottom, top.size(0));
    count_.assign(cells, 0);
    probability_.resize(cells);
    for (std::int64_t b = 0; b < top.size(0); ++b) {
      std::int64_t previous = entries;
      for (std::int64_t i = 0; i < bottom_tokens_; ++i) {
        const auto context = t[b * top_tokens_ + Parent(i)], code = w[b * bottom_tokens_ + i];
        ++count_[At(context, previous, code)];
        previous = code;
      }
    }
    for (std::int64_t context = 0; context < entries; ++context)
      for (std::int64_t previous = 0; previous <= entries; ++previous) {
        double total = double(entries) * smoothing;
        for (std::int64_t code = 0; code < entries; ++code)
          total += count_[At(context, previous, code)];
        for (std::int64_t code = 0; code < entries; ++code) {
          const double p = (count_[At(context, previous, code)] + smoothing) / total;
          if (!std::isfinite(total) || !std::isfinite(p) || p <= 0)
            throw std::invalid_argument(
                "VQ2 prior smoothing must yield positive finite probabilities");
          probability_[At(context, previous, code)] = p;
        }
      }
  }
  /// @brief Export independent Int32[K,K+1,K] parent/previous/current counts.
  /// @return Owning table; changing it does not mutate the prior.
  Tensor counts() const { return from_vector(count_, {entries_, entries_ + 1, entries_}); }
  /// @brief Export independent Float64[K,K+1,K] normalized transition rows.
  /// @return Owning positive probabilities; previous=K is the BOS row per parent.
  Tensor probabilities() const {
    return from_vector(probability_, {entries_, entries_ + 1, entries_});
  }
  /// @brief Sample a fine grid from a supplied coarse grid and explicit uniform values.
  ///
  /// Uses the aligned top parent and previous sampled fine symbol; BOS resets
  /// per image. Selects the first cumulative probability strictly greater than u,
  /// with final-code fallback for row-sum roundoff. Uniforms must be Float64 in
  /// [0,1), retaining source precision at CDF boundaries. No RNG is consumed.
  /// Inputs/state unchanged; owning detached IDs. Read-only thread-safe calls.
  /// Work O(B*bottom_tokens*K), storage O(B*(top_tokens+bottom_tokens)).
  /// @param top Valid nonempty Int32[B,top_tokens] coarse codes, accepting strides.
  /// @param uniforms Explicit finite Float64[B,bottom_tokens] values in[0,1).
  /// @return Owning detached Int32[B,bottom_tokens] fine IDs.
  /// @throws std::invalid_argument For malformed grids, dtype, invalid symbols or uniform values.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// auto top=n::zeros({1,1},n::DType::Int32);
  /// n::models::Vq2ConditionalPrior prior(top,n::zeros({1,4},n::DType::Int32),1,1,2);
  /// const auto ids=prior.Sample(top,n::full({1,4},.9,n::DType::Float64));
  /// // ids=[1,1,1,1]: first u exceeds.75, then unseen previous1 is uniform.
  /// @endcode
  /// @see Nll(), VqMarkovPrior::Sample()
  Tensor Sample(const Tensor& top, const Tensor& uniforms) const {
    const auto t = Top(top);
    if (!uniforms.defined() || uniforms.dim() != 2 || uniforms.size(0) != top.size(0) ||
        uniforms.size(1) != bottom_tokens_ || uniforms.dtype() != DType::Float64)
      throw std::invalid_argument("VQ2 prior uniforms need matching Float64[B,fine tokens]");
    const auto u = uniforms.contiguous().to_vector<double>();
    for (double x : u)
      if (!std::isfinite(x) || x < 0 || x >= 1)
        throw std::invalid_argument("VQ2 prior uniforms must be finite in[0,1)");
    std::vector<int> ids(u.size());
    for (std::int64_t b = 0; b < top.size(0); ++b) {
      std::int64_t previous = entries_;
      for (std::int64_t i = 0; i < bottom_tokens_; ++i) {
        const auto context    = t[b * top_tokens_ + Parent(i)];
        double     cumulative = 0;
        int        selected   = int(entries_ - 1);
        for (std::int64_t code = 0; code < entries_; ++code) {
          cumulative += probability_[At(context, previous, code)];
          if (u[b * bottom_tokens_ + i] < cumulative) {
            selected = int(code);
            break;
          }
        }
        ids[b * bottom_tokens_ + i] = selected;
        previous                    = selected;
      }
    }
    return from_vector(ids, {top.size(0), bottom_tokens_});
  }
  /// @brief Score paired coarse/fine grids in mean conditional nats per complete fine grid.
  ///
  /// Resets BOS per image, sums fine-token NLLs with aligned parents and divides
  /// by B only. This is conditional code-grid NLL, not image likelihood or per-
  /// token loss. Inputs/state unchanged; no gradients. Work/storage O(B*bottom_tokens).
  /// @param top Valid nonempty Int32[B,top_tokens] coarse codes.
  /// @param bottom Matching valid Int32[B,bottom_tokens] fine codes.
  /// @return Mean natural-log conditional code NLL per image/grid.
  /// @throws std::invalid_argument For invalid paired geometry, dtypes or symbols.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// auto top=n::zeros({1,1},n::DType::Int32),bottom=n::zeros({1,4},n::DType::Int32);
  /// n::models::Vq2ConditionalPrior prior(top,bottom,1,1,2);
  /// const double nll=prior.Nll(top,bottom);
  /// // -log(.75)-3*log(.875), approximately0.688276 nats/grid.
  /// @endcode
  /// @see Sample(), probabilities()
  double Nll(const Tensor& top, const Tensor& bottom) const {
    const auto t = Top(top), w = Bottom(bottom, top.size(0));
    double     total = 0;
    for (std::int64_t b = 0; b < top.size(0); ++b) {
      std::int64_t previous = entries_;
      for (std::int64_t i = 0; i < bottom_tokens_; ++i) {
        const auto code = w[b * bottom_tokens_ + i];
        total -= std::log(probability_[At(t[b * top_tokens_ + Parent(i)], previous, code)]);
        previous = code;
      }
    }
    return total / double(top.size(0));
  }
};
} // namespace datamunge::nn::models
