#pragma once
/// @file vector_quantized_autoencoder.hpp
/// @brief Nearest-vector surrogate gradients, spatial VQ-VAE and explicit raster code priors.
#include <limits>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/convolutional_variational_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Internal checked shapes and discrete-symbol validation for VQ components.
namespace vq_detail {
/// @brief Multiply positive dimensions without signed overflow.
/// @param a Positive first dimension.
/// @param b Positive second dimension.
/// @return Representable product.
/// @throws std::invalid_argument For nonpositive dimensions or int64 overflow.
inline std::int64_t Product(std::int64_t a, std::int64_t b) {
  if (a <= 0 || b <= 0 || a > std::numeric_limits<std::int64_t>::max() / b)
    throw std::invalid_argument("VQ dimensions must have positive representable products");
  return a * b;
}
/// @brief Validate vocabulary size and its Int32 symbol representation.
/// @param entries Positive vocabulary size, at most INT32_MAX-1 (BOS uses entries).
/// @throws std::invalid_argument For invalid vocabulary size.
inline void Vocabulary(std::int64_t entries) {
  if (entries <= 0 || entries >= std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("VQ vocabulary must fit positive Int32 symbols plus BOS");
}
/// @brief Validate nonempty Int32 symbols and copy values in logical row-major order.
/// @param ids Nonempty Int32 tensor of any shape, with symbols in [0,entries).
/// @param entries Valid vocabulary size.
/// @return Owning symbol vector, accepting noncontiguous tensors.
/// @throws std::invalid_argument For invalid vocabulary, dtype, shape or symbol values.
inline std::vector<int> Ids(const Tensor& ids, std::int64_t entries) {
  Vocabulary(entries);
  if (!ids.defined() || ids.dtype() != DType::Int32 || ids.numel() <= 0)
    throw std::invalid_argument("VQ symbols require a nonempty Int32 tensor");
  auto values = ids.contiguous().to_vector<int>();
  for (int code : values)
    if (code < 0 || code >= entries)
      throw std::invalid_argument("VQ symbol lies outside the vocabulary");
  return values;
}
/// @brief Validate a nonempty Int32 image-by-token grid.
/// @param ids Int32 [B,T] grid with B,T>0 and valid symbols.
/// @param entries Valid vocabulary size.
/// @return Copied row-major symbols.
/// @throws std::invalid_argument For malformed grid or invalid symbols.
inline std::vector<int> Grid(const Tensor& ids, std::int64_t entries) {
  if (!ids.defined() || ids.dim() != 2 || ids.size(0) <= 0 || ids.size(1) <= 0)
    throw std::invalid_argument("VQ prior needs a nonempty [images,tokens] grid");
  return Ids(ids, entries);
}
/// @brief Check spatial model widths, image/activation/kernel products and codebook size.
/// @param batch Positive image count.
/// @param height Positive height divisible by four.
/// @param width Positive width divisible by four.
/// @param entries Positive Int32 vocabulary size.
/// @param dimensions Positive embedding width.
/// @param channels Positive encoder/decoder hidden channels.
/// @return Quarter-resolution tokens per image.
/// @throws std::invalid_argument For invalid geometry or overflowing products.
inline std::int64_t Geometry(
    std::int64_t batch,
    std::int64_t height,
    std::int64_t width,
    std::int64_t entries,
    std::int64_t dimensions,
    std::int64_t channels) {
  Vocabulary(entries);
  if (height <= 0 || width <= 0 || height % 4 || width % 4)
    throw std::invalid_argument("VQ image height and width must be positive multiples of four");
  Product(Product(batch, Product(height, width)), std::max(channels, dimensions));
  Product(Product(9, channels), dimensions);
  Product(entries, dimensions);
  const auto tokens = Product(height / 4, width / 4);
  Product(Product(batch, tokens), entries);
  return tokens;
}
} // namespace vq_detail

/// @brief Own nearest indices, lookup values and two differently detached dictionary penalties.
///
/// All Tensor handles own lifetime. indices are detached Int32[N]; assignment is
/// detached floating[N,K]; quantized[N,D] differentiates only to book. The
/// straight_through[N,D] path differentiates only to encoded, with identity
/// surrogate Jacobian. codebook_loss/commitment_loss are scalar means over N*D.
/// No fitting, RNG, EMA, resets or optimizer state is stored.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto q=n::models::VectorQuantize(n::tensor({{.2,.1}}),n::tensor({{0.,0.},{1.,0.}}),2,2);
/// // q.indices is [0]; q.quantized is [[0,0]]; both penalties are0.025.
/// @endcode
/// @see VectorQuantize(), VectorQuantizedAutoencoder()
struct VectorQuantizedOutput {
  Tensor indices;          ///< Detached Int32[N] selected code IDs, lowest index on ties.
  Tensor assignment;       ///< Detached one-hot[N,K] in the embedding computation dtype.
  Tensor quantized;        ///< Lookup vectors[N,D], with live book-only paths.
  Tensor straight_through; ///< Selected vectors with identity encoded-only surrogate paths.
  Tensor codebook_loss;    ///< Mean(stop(encoded)-quantized)^2, book-only gradients.
  Tensor commitment_loss;  ///< Mean(encoded-stop(quantized))^2, encoded-only gradients.
};

/// @brief Select nearest vectors and expose independent straight-through/dictionary/commitment
/// paths.
///
/// Distance is sum(encoded^2)+sum(book^2)-2*encoded*book^T in the input dtype.
/// Detached argmin chooses the lowest index on ties, including duplicate vectors.
/// An optional Int32 override with N elements replaces assignments; its shape
/// may be [N] or a grid. All IDs are checked before lookup. The forward surrogate
/// is stop(quantized)+(encoded-stop(encoded)), avoiding cancellation against
/// encoded; signed zero may normalize. Its identity encoded Jacobian is a
/// surrogate, not the derivative of nearest-neighbour selection. Reconstruction
/// through it gives no book gradient. Dictionary and commitment terms average
/// ALL N*D elements, with opposite targets detached. Available continuous higher
/// paths remain live; no derivative is defined through indices or tie changes.
/// Finite nonempty matching Float32/64 encoded[N,D],book[K,D] accept strides.
/// Overflowed computed distances are rejected; other computed overflow propagates.
/// Inputs unchanged; owning fields. Work O(N*K*D), storage O(N*K+N*D). No RNG/state.
/// @param encoded Finite nonempty floating matrix[N,dimensions].
/// @param book Finite matching matrix[entries,dimensions].
/// @param entries Positive Int32 vocabulary size, default16.
/// @param dimensions Positive embedding width, default4.
/// @param override_indices Optional valid Int32 tensor with N elements; undefined selects nearest.
/// @return Owning six-field quantization record with explicit gradient routing.
/// @throws std::invalid_argument For invalid shapes/dtypes/values/IDs, distance overflow or
/// products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto e=n::tensor({{.2,.1}}).requires_grad_(),b=n::tensor({{0.,0.},{1.,0.}}).requires_grad_();
/// auto q=n::models::VectorQuantize(e,b,2,2);
/// const auto g=n::grad(n::sum(q.straight_through),{e,b},{.allow_unused=true});
/// // g[0] is [[1,1]], while g[1] is undefined: no reconstruction path to book.
/// @endcode
/// @par References
/// - van den Oord, Vinyals and Kavukcuoglu (2017), Neural Discrete Representation
///   Learning, https://arxiv.org/abs/1711.00937 . Uses gradient dictionary updates,
///   element-mean penalties and a detached identity surrogate, not EMA updates.
/// @see VqLookup(), VectorQuantizedOutput, straight_through()
inline VectorQuantizedOutput VectorQuantize(
    const Tensor& encoded,
    const Tensor& book,
    std::int64_t  entries          = 16,
    std::int64_t  dimensions       = 4,
    const Tensor& override_indices = Tensor{}) {
  vq_detail::Vocabulary(entries);
  if (!encoded.defined() || !book.defined())
    throw std::invalid_argument("VQ needs defined encoded and book matrices");
  aae_detail::Matrix(encoded, dimensions, encoded.dtype(), false);
  aae_detail::Matrix(book, dimensions, encoded.dtype(), false);
  if (book.size(0) != entries)
    throw std::invalid_argument("VQ book rows must equal entries");
  vq_detail::Product(encoded.size(0), entries);
  if (override_indices.defined()) {
    vq_detail::Ids(override_indices, entries);
    if (override_indices.numel() != encoded.size(0))
      throw std::invalid_argument("VQ override must contain one symbol per encoded vector");
  }
  VectorQuantizedOutput o;
  if (override_indices.defined()) {
    o.indices = reshape(override_indices, {encoded.size(0)}).contiguous();
  } else {
    auto distance = sum(square(encoded), {1}, true) + expand_dims(sum(square(book), {1}), 0) -
                    2.0 * matmul(encoded, book, false, true);
    for (double value : distance.contiguous().to_vector<double>())
      if (!std::isfinite(value))
        throw std::invalid_argument("VQ computed distance overflow");
    o.indices = cast(argmin(stop_gradient(distance), 1), DType::Int32).contiguous();
  }
  o.assignment       = one_hot(o.indices, entries, 1., 0., book.dtype());
  o.quantized        = matmul(o.assignment, book);
  o.straight_through = stop_gradient(o.quantized) + (encoded - stop_gradient(encoded));
  o.codebook_loss    = mean(square(stop_gradient(encoded) - o.quantized), {0, 1});
  o.commitment_loss  = mean(square(encoded - stop_gradient(o.quantized)), {0, 1});
  return o;
}

/// @brief Look up a discrete image-by-token code grid without an encoder or targets.
///
/// Int32 IDs[B,T] must be nonempty in [0,K); book[K,D] must be finite nonempty
/// Float32/64. Returns flattened [B*T,D] in row-major order with live book paths,
/// accepting strides. IDs have no gradient. Inputs unchanged; owning result.
/// Work O(B*T*K*D), storage O(B*T*K+B*T*D), using the same one-hot matmul as VQ.
/// @param ids Valid nonempty Int32[B,T] grid.
/// @param book Finite floating[K,D] dictionary.
/// @param entries Positive vocabulary size, default16, equal to book rows.
/// @return Owning lookup vectors[B*T,D] differentiable to book.
/// @throws std::invalid_argument For invalid grid/book/dtype/values or overflowing products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// codes=n::models::VqLookup(n::from_vector(std::vector<int>{1,0},{1,2}),n::tensor({{0.,0.},{1.,2.}}),2);
/// // codes is [[1,2],[0,0]], flattened in image/token order.
/// @endcode
/// @par References
/// - van den Oord et al. (2017), https://arxiv.org/abs/1711.00937 , discrete dictionary lookup.
/// @see VectorQuantize(), VqDecoder()
inline Tensor VqLookup(const Tensor& ids, const Tensor& book, std::int64_t entries = 16) {
  vq_detail::Grid(ids, entries);
  if (!book.defined() || book.dim() != 2 || book.size(0) != entries)
    throw std::invalid_argument("VQ lookup book must have K rows");
  aae_detail::Matrix(book, book.size(1), book.dtype(), false);
  vq_detail::Product(ids.numel(), entries);
  return matmul(reshape(one_hot(ids, entries, 1., 0., book.dtype()), {-1, entries}), book);
}

/// @brief Decode a flattened quarter-resolution code lattice to Bernoulli logits.
///
/// Reshapes codes[B*T,D] to [B,H/4,W/4,D], repeat2, same Conv3 D->C/ReLU,
/// repeat2, same Conv3 C->1/linear, then flattens pixels to [B,H*W]. T=H*W/16.
/// Uses decoder1/W,b (He uniform/zero bias) and output/W,b (ordinary Glorot/zero
/// bias), exactly shared with VectorQuantizedAutoencoder(). Default four leaves
/// contain369 scalars. Codes alone enter: no image, book, RNG draws or fit.
/// Finite Float32/64 codes must match store dtype; strides and all continuous
/// code/parameter paths remain. Geometry/input checks precede leaf creation;
/// incompatible existing leaves can fail later. Inputs unchanged; output owns
/// lifetime. Work O(B*H*W*9*C*(D+1)); synchronize mutations of a shared store.
/// @param parameters Mutable floating decoder parameter store.
/// @param codes Finite nonempty flattened vectors[B*T,dimensions].
/// @param height Positive output height divisible by four, default28.
/// @param width Positive output width divisible by four, default28.
/// @param dimensions Positive embedding width, default4.
/// @param channels Positive hidden channels, default8.
/// @return Owning differentiable logits[B,H*W], without sigmoid clipping.
/// @throws std::invalid_argument For invalid geometry/input/dtype/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2707);
/// const auto probability=n::sigmoid(n::models::VqDecoder(p,n::zeros({49,4})));
/// // One target-free probability image[1,784]; four decoder leaves.
/// @endcode
/// @par References
/// - van den Oord et al. (2017), https://arxiv.org/abs/1711.00937 . Compact
///   repeat/convolution teaching architecture, not the paper's full model.
/// @see VqLookup(), VectorQuantizedAutoencoder()
inline Tensor VqDecoder(
    ParameterStore& parameters,
    const Tensor&   codes,
    std::int64_t    height     = 28,
    std::int64_t    width      = 28,
    std::int64_t    dimensions = 4,
    std::int64_t    channels   = 8) {
  const auto tokens = vq_detail::Geometry(1, height, width, 1, dimensions, channels);
  aae_detail::Matrix(codes, dimensions, parameters.dtype(), false);
  if (codes.size(0) % tokens)
    throw std::invalid_argument("VQ decoder needs a whole number of spatial code grids");
  const auto n = codes.size(0) / tokens;
  vq_detail::Geometry(n, height, width, 1, dimensions, channels);
  auto h = conv2d(
      parameters,
      "decoder1",
      upsample_nearest(reshape(codes, {n, height / 4, width / 4, dimensions}), 2),
      channels,
      3,
      conv_vae_detail::Relu());
  return reshape(
      conv2d(parameters, "output", upsample_nearest(h, 2), 1, 3, ConvLayerOptions{}),
      {n, height * width});
}

/// @brief Own spatial encoder, learned dictionary, quantization and Bernoulli objective endpoints.
///
/// All fields own lifetime and retain their intended autograd paths. vq contains
/// the explicit stopped/surrogate routes. No fitting loop, mode or RNG is stored.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2707);
/// const auto out=n::models::VectorQuantizedAutoencoder(p,n::zeros({1,28,28,1}));
/// // Nine leaves/805 scalars; loss=mean(pixel NLL)+dictionary+.25*commitment.
/// @endcode
/// @see VectorQuantizedAutoencoder(), VectorQuantizedOutput
struct VectorQuantizedAutoencoderOutput {
  Tensor                encoded; ///< Flattened encoder vectors[B*T,D], T=H*W/16.
  Tensor                book;    ///< Live learned dictionary[K,D], same handle as the store leaf.
  Tensor                probability; ///< Sigmoid decoder probabilities[B,H*W], not binary draws.
  Tensor                nll;         ///< Pixel-summed Bernoulli NLL[B], without Gaussian KL.
  Tensor                reconstruction_loss; ///< Image-mean NLL scalar.
  Tensor                loss; ///< Reconstruction plus codebook plus beta*commitment scalar.
  VectorQuantizedOutput vq;   ///< Owning six-field quantizer record, including both penalties.
};

/// @brief Evaluate a compact spatial VQ-VAE with explicit surrogate gradient routes.
///
/// Unit-range grayscale NHWC[B,H,W,1], B>0, H/W divisible by four: same Conv3
/// 1->C/ReLU, avg2, same Conv3 C->D/ReLU, avg2, nearest K-vector dictionary,
/// then VqDecoder(). He-uniform encoder/hidden decoder, ordinary Glorot output,
/// zero biases, codebook U(-.5,.5). Creation order is encoder1/W,b, encoder2/W,b,
/// codebook, decoder1/W,b, output/W,b; default9 leaves/805 scalars. Loss averages
/// pixel-summed stable Bernoulli NLL over B, adds element-mean dictionary square
/// and beta times element-mean commitment square. Reconstruction follows the
/// identity surrogate to encoded and skips book; neither an exact argmin
/// derivative nor a Gaussian/pixel likelihood ELBO is claimed. No EMA, resets,
/// prior fitting, sampling, RNG noise or optimizer is hidden. Parameter creation
/// uses the caller's store initializer stream. Finite matching Float32/64 inputs
/// accept strides and preserve intended gradients. Checks precede leaf creation;
/// incompatible stored leaves/computed overflow can fail later. Owning outputs;
/// inputs unchanged. Work O(B*H*W*9*C*(D+1)+B*H*W*K*D/16); synchronize shared stores.
/// @param parameters Mutable floating store creating/reusing all nine leaves.
/// @param image Finite unit-range nonempty Float32/64 grayscale NHWC[B,H,W,1].
/// @param entries Positive Int32 vocabulary size, default16.
/// @param dimensions Positive embedding width, default4.
/// @param channels Positive hidden channel width, default8.
/// @param commitment_beta Finite nonnegative commitment multiplier, default0.25, representable in
/// dtype.
/// @return Owning spatial/quantization/probability/objective record.
/// @throws std::invalid_argument For invalid inputs/options/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2707);
/// auto out=n::models::VectorQuantizedAutoencoder(p,n::full({2,28,28,1},.3));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // All nine leaves are available to the caller's optimizer; no update occurs here.
/// @endcode
/// @par References
/// - van den Oord, Vinyals and Kavukcuoglu (2017), Neural Discrete Representation
///   Learning, https://arxiv.org/abs/1711.00937 . Compact Bernoulli experiment,
///   gradient-updated codebook and element-mean penalties; no PixelCNN prior.
/// @see VectorQuantize(), VqDecoder(), VqMarkovPrior
inline VectorQuantizedAutoencoderOutput VectorQuantizedAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    std::int64_t    entries         = 16,
    std::int64_t    dimensions      = 4,
    std::int64_t    channels        = 8,
    double          commitment_beta = .25) {
  conv_vae_detail::Image(image, parameters.dtype());
  const auto n = image.size(0), height = image.size(1), width = image.size(2);
  const auto tokens = vq_detail::Geometry(n, height, width, entries, dimensions, channels);
  if (!std::isfinite(commitment_beta) || commitment_beta < 0 ||
      (parameters.dtype() == DType::Float32 &&
       (!std::isfinite(float(commitment_beta)) ||
        (commitment_beta > 0 && float(commitment_beta) == 0))))
    throw std::invalid_argument("VQ commitment beta must be finite nonnegative and representable");
  VectorQuantizedAutoencoderOutput o;
  auto                             h =
      avg_pool2d(conv2d(parameters, "encoder1", image, channels, 3, conv_vae_detail::Relu()), 2, 2);
  o.encoded = reshape(
      avg_pool2d(conv2d(parameters, "encoder2", h, dimensions, 3, conv_vae_detail::Relu()), 2, 2),
      {n * tokens, dimensions});
  o.book        = parameters.get("codebook", {entries, dimensions}, init::uniform(.5));
  o.vq          = VectorQuantize(o.encoded, o.book, entries, dimensions);
  auto logits   = VqDecoder(parameters, o.vq.straight_through, height, width, dimensions, channels);
  o.probability = sigmoid(logits);
  o.nll         = BernoulliPixelNll(logits, reshape(image, {n, height * width}));
  o.reconstruction_loss = mean(o.nll, {0});
  o.loss = o.reconstruction_loss + (o.vq.codebook_loss + commitment_beta * o.vq.commitment_loss);
  return o;
}

/// @brief Own empirical code counts and exp(entropy) perplexity of their pooled distribution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// usage=dm::nn::models::VqUsage(dm::nn::from_vector(std::vector<int>{0,1,0,1},{2,2}),2);
/// // counts=[2,2], perplexity=2; this measures code use, not likelihood or semantics.
/// @endcode
/// @see VqUsage()
struct VqCodeUsage {
  Tensor counts;     ///< Owning Int32[K] counts, including unused zero-count symbols.
  double perplexity; ///< exp(-sum p*log p), natural entropy of pooled empirical code use.
};

/// @brief Count valid code symbols and return pooled empirical usage perplexity.
///
/// Any nonempty Int32 shape is flattened; counts fit Int32 by limiting numel to
/// INT32_MAX. Zero-probability terms are omitted. A single used symbol gives1,
/// uniform use of K symbols gives K. Unused codes remain visible. Inputs unchanged;
/// owning counts, no gradients or shared state. Work O(N+K), storage O(N+K).
/// @param ids Nonempty valid Int32 code tensor, any shape/strides.
/// @param entries Positive Int32 vocabulary size, default16.
/// @return Owning counts[K] and pooled usage perplexity, not a compression/likelihood score.
/// @throws std::invalid_argument For invalid input, symbols, vocabulary or count overflow.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto usage=dm::nn::models::VqUsage(dm::nn::zeros({2,3},dm::nn::DType::Int32));
/// // counts[0]=6, the other fifteen counts are0; perplexity=1.
/// @endcode
/// @par References
/// - van den Oord et al. (2017), https://arxiv.org/abs/1711.00937 , code-use diagnostics.
/// @see VqCodeUsage, VqMarkovPrior
inline VqCodeUsage VqUsage(const Tensor& ids, std::int64_t entries = 16) {
  const auto values = vq_detail::Ids(ids, entries);
  if (ids.numel() > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("VQ pooled counts exceed Int32");
  std::vector<int> counts(entries, 0);
  for (int value : values)
    ++counts[value];
  double entropy = 0;
  for (int count : counts) {
    const double p = double(count) / double(values.size());
    if (p > 0)
      entropy -= p * std::log(p);
  }
  return {from_vector(counts, {entries}), std::exp(entropy)};
}

/// @brief Fit a smoothed first-order raster Markov prior to caller-supplied discrete grids.
///
/// Each input row begins with BOS=K; counts[previous,current] include K ordinary
/// contexts plus BOS. p=(count+smoothing)/(context_total+K*smoothing), with
/// default K16/half-count0.5. There is no EOS or cross-row transition. This cheap
/// separate code prior is not the paper's learned PixelCNN or an image likelihood.
/// The caller supplies ONLY fitting codes and chooses model/checkpoint separately;
/// this class cannot check split provenance. Construction copies all counts and
/// probabilities; the input may be destroyed. Exports return independent tensors.
/// Read-only scoring/sampling is thread-safe, no shared RNG or network updates.
/// Work O(B*T+K*K), storage O(B*T+K*K); sampling takes explicit Float64 uniforms.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::models::VqMarkovPrior prior(n::from_vector(std::vector<int>{0,1,1,0},{2,2}),2);
/// const auto ids=prior.Sample(n::full({1,3},.25,n::DType::Float64));
/// // An explicit-uniform three-symbol sample; prior never reads images or updates a model.
/// @endcode
/// @par References
/// - van den Oord et al. (2017), https://arxiv.org/abs/1711.00937 , motivates a
///   separately fitted prior. This first-order half-count baseline is a compact
///   repository variant, not the paper's neural autoregressive prior.
/// @see VectorQuantizedAutoencoder(), VqUsage(), MidpointUniformNoise()
class VqMarkovPrior {
  std::int64_t        entries_;     ///< Vocabulary size and BOS context index.
  std::vector<int>    count_;       ///< Private Int32-safe context/symbol counts.
  std::vector<double> probability_; ///< Private positive normalized transition probabilities.
 public:
  /// @brief Copy discrete fitting grids into a smoothed raster transition table.
  /// @param fitting Nonempty valid Int32[B,T] fitting-only codes, accepting strides.
  /// @param entries Positive vocabulary size, default16.
  /// @param smoothing Finite positive Dirichlet pseudocount, default0.5.
  /// @throws std::invalid_argument For malformed symbols/grid/options, count/product overflow or
  /// unrepresentable smoothing.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// dm::nn::models::VqMarkovPrior prior(dm::nn::zeros({1,2},dm::nn::DType::Int32),2);
  /// // BOS probabilities are[.75,.25]; context0 also has[.75,.25].
  /// @endcode
  /// @see VqMarkovPrior, probabilities()
  explicit VqMarkovPrior(const Tensor& fitting, std::int64_t entries = 16, double smoothing = .5)
      : entries_(entries) {
    const auto ids = vq_detail::Grid(fitting, entries);
    if (fitting.numel() > std::numeric_limits<std::int32_t>::max() || !std::isfinite(smoothing) ||
        smoothing <= 0 || !std::isfinite(double(entries) * smoothing))
      throw std::invalid_argument("VQ prior count overflow or invalid smoothing");
    const auto cells = vq_detail::Product(entries + 1, entries);
    count_.assign(cells, 0);
    probability_.resize(cells);
    const auto n = fitting.size(0), tokens = fitting.size(1);
    for (std::int64_t b = 0; b < n; ++b) {
      std::int64_t previous = entries;
      for (std::int64_t t = 0; t < tokens; ++t) {
        const auto code = ids[b * tokens + t];
        ++count_[previous * entries + code];
        previous = code;
      }
    }
    for (std::int64_t context = 0; context <= entries; ++context) {
      double total = double(entries) * smoothing;
      for (std::int64_t code = 0; code < entries; ++code)
        total += count_[context * entries + code];
      for (std::int64_t code = 0; code < entries; ++code) {
        const double p = (count_[context * entries + code] + smoothing) / total;
        if (!std::isfinite(total) || !std::isfinite(p) || p <= 0)
          throw std::invalid_argument(
              "VQ prior smoothing must yield positive finite probabilities");
        probability_[context * entries + code] = p;
      }
    }
  }
  /// @brief Export independent Int32[K+1,K] counts, including the final BOS row.
  /// @return Owning count tensor with no alias to internal state.
  Tensor counts() const { return from_vector(count_, {entries_ + 1, entries_}); }
  /// @brief Export independent Float64[K+1,K] smoothed probabilities.
  /// @return Owning positive normalized transition rows; final row is BOS.
  Tensor probabilities() const { return from_vector(probability_, {entries_ + 1, entries_}); }
  /// @brief Sample raster codes from explicit uniforms without any RNG or image input.
  ///
  /// Uniforms must be finite Float64[B,T] in[0,1), B,T>0. Each row resets BOS;
  /// choose the first cumulative probability strictly greater than u, matching
  /// source inverse-CDF ties. Final-code fallback absorbs floating row-sum error.
  /// Codes own lifetime and have no gradient; input unchanged, state unchanged.
  /// Work O(B*T*K), storage O(B*T), safe for concurrent read-only calls.
  /// @param uniforms Explicit nonempty Float64[B,T] values in[0,1), accepting strides.
  /// @return Owning Int32[B,T] sampled IDs; no random state is consumed.
  /// @throws std::invalid_argument For malformed shape/dtype or invalid uniform values.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::models::VqMarkovPrior prior(n::zeros({1,2},n::DType::Int32),2);
  /// const auto ids=prior.Sample(n::full({1,2},.75,n::DType::Float64));
  /// // ids=[1,1]: u at a cumulative boundary selects the NEXT interval.
  /// @endcode
  /// @see Nll(), MidpointUniformNoise()
  Tensor Sample(const Tensor& uniforms) const {
    if (!uniforms.defined() || uniforms.dim() != 2 || uniforms.size(0) <= 0 ||
        uniforms.size(1) <= 0 || uniforms.dtype() != DType::Float64)
      throw std::invalid_argument("VQ prior sampling needs nonempty Float64[B,T] uniforms");
    const auto u = uniforms.contiguous().to_vector<double>();
    for (double value : u)
      if (!std::isfinite(value) || value < 0 || value >= 1)
        throw std::invalid_argument("VQ prior uniforms must be finite in[0,1)");
    std::vector<int> ids(u.size());
    const auto       n = uniforms.size(0), tokens = uniforms.size(1);
    for (std::int64_t b = 0; b < n; ++b) {
      std::int64_t previous = entries_;
      for (std::int64_t t = 0; t < tokens; ++t) {
        double cumulative = 0;
        int    selected   = int(entries_ - 1);
        for (std::int64_t code = 0; code < entries_; ++code) {
          cumulative += probability_[previous * entries_ + code];
          if (u[b * tokens + t] < cumulative) {
            selected = int(code);
            break;
          }
        }
        ids[b * tokens + t] = selected;
        previous            = selected;
      }
    }
    return from_vector(ids, uniforms.shape());
  }
  /// @brief Score valid code grids in mean natural-log nats per complete row.
  ///
  /// Resets BOS per image and sums all token NLLs, then divides by B only.
  /// Grid token count may differ from fitting. State/input unchanged; no autograd.
  /// Work/storage O(B*T) for validated symbols; concurrent read-only calls are safe.
  /// @param grid Nonempty valid Int32[B,T] code grid, accepting strides.
  /// @return Mean nats per grid, not per token, image log likelihood or a codec bitrate.
  /// @throws std::invalid_argument For invalid grid/dtype/symbols.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::models::VqMarkovPrior prior(n::zeros({1,2},n::DType::Int32),2);
  /// const double nll=prior.Nll(n::zeros({1,2},n::DType::Int32));
  /// // nll=-2*log(.75), approximately0.575364 nats/grid.
  /// @endcode
  /// @see Sample(), probabilities()
  double Nll(const Tensor& grid) const {
    const auto ids    = vq_detail::Grid(grid, entries_);
    double     result = 0;
    const auto n = grid.size(0), tokens = grid.size(1);
    for (std::int64_t b = 0; b < n; ++b) {
      std::int64_t previous = entries_;
      for (std::int64_t t = 0; t < tokens; ++t) {
        const auto code = ids[b * tokens + t];
        result -= std::log(probability_[previous * entries_ + code]);
        previous = code;
      }
    }
    return result / double(n);
  }
};
} // namespace datamunge::nn::models
