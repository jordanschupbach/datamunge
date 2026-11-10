#pragma once
/// @file autoencoder_models.hpp
/// @brief Autoencoder forward models and reproducible augmentation without fitting or dataset orchestration.

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/image_ops.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <vector>

namespace datamunge::nn::models {

/// @brief Generate owning Float64 open-interval uniforms from a fresh MT19937 midpoint stream.
///
/// Each raw 32-bit engine word r produces exactly (double(r)+.5)/2^32, in logical
/// row-major order. There is one engine draw per element, without a distribution
/// adapter, cached spare, clipping or shared generator. Values lie in
/// [2^-33,1-2^-33], with 32 bits of discrete resolution, and retain that strict
/// open interval in Float64. Casting to Float32 can round the upper end to1;
/// keep Float64 for APIs requiring open-interval categorical uniforms. This is
/// the source sentence-generation stream, distinct from nn::uniform()'s 53-bit
/// generator. Input shape is unchanged, output detached and independently owned.
/// Empty axes produce empty tensors; an empty shape produces one scalar.
/// Separate calls are thread-independent. Work/storage O(product(shape)); shape
/// and representable element/byte counts are checked before allocation.
/// @param shape Nonnegative output dimensions; empty selects a scalar.
/// @param seed Full unsigned32 local MT19937 seed, with no default.
/// @return Contiguous detached Float64 tensor of exactly shape.
/// @throws std::invalid_argument For negative dimensions.
/// @throws std::overflow_error For unrepresentable element, stride or byte counts.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto u=dm::nn::models::MidpointUniformNoise({20,7},3217);
/// // Explicit Float64[B,7] token uniforms; repeating the call repeats every bit.
/// @endcode
/// @par References
/// - Matsumoto and Nishimura (1998), Mersenne Twister: A 623-dimensionally
///   equidistributed uniform pseudorandom number generator,
///   https://doi.org/10.1145/272991.272995 . The midpoint conversion follows the
///   migrated teaching experiment, rather than a standard-library distribution.
/// @see GaussianNoise(), uniform(), SequenceGenerate()
inline Tensor MidpointUniformNoise(const Shape& shape, std::uint32_t seed) {
    const auto count=shape_numel(shape);
    if(std::uint64_t(count)>std::vector<double>().max_size())
        throw std::overflow_error("midpoint uniform byte count exceeds host vector capacity");
    std::vector<double> values(static_cast<std::size_t>(count));
    std::mt19937 engine(seed);
    for(auto& value:values)value=(double(engine())+.5)/4294967296.;
    // from_vector's omitted-shape convention uses [1]; preserve an explicit scalar.
    auto result=from_vector(values,shape);
    return shape.empty()?reshape(result,shape):result;
}

/// @brief Generate explicit Float64 Gaussian banks with the original MT19937/Box-Muller stream.
///
/// A fresh std::mt19937(seed) supplies pairs u=(raw+.5)/2^32, v=(raw+.5)/2^32.
/// Successive values are sqrt(-2*log(u))*cos(2*pi*v), then the sine partner.
/// An odd final count discards the second partner. No standard-library normal
/// distribution, shared generator or cached spare is used. This preserves the
/// autoencoder examples' stream, distinct from nn::normal() and Mt19937::normal64().
/// Values are detached and newly owned, with a leading sample axis. Empty
/// dimensions/counts are supported; an empty sample shape produces [samples].
/// Work/storage O(samples*product(shape)); separate calls are thread-independent.
/// Transcendental rounding can vary between platforms/libm implementations.
///
/// @param shape Shape of each bank entry; every dimension must be nonnegative.
/// @param samples Nonnegative leading bank count.
/// @param seed Raw 32-bit Mersenne Twister seed, with no default.
/// @return Contiguous detached Float64 tensor [samples,shape...].
/// @throws std::invalid_argument For negative dimensions or sample count.
/// @throws std::overflow_error If the nonzero element count exceeds int64.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto bank = dm::nn::models::GaussianNoise({2,3}, 1, 1711);
/// // bank is Float64 [1,2,3]; repeating the call repeats its values.
/// @endcode
///
/// @par References
/// - Box and Muller (1958), A Note on the Generation of Random Normal Deviates,
///   https://doi.org/10.1214/aoms/1177706645 .
/// @see CorruptForDenoising(), normal()
inline Tensor GaussianNoise(const Shape& shape, std::int64_t samples, std::uint32_t seed) {
    Shape full{samples};
    full.insert(full.end(), shape.begin(), shape.end());
    std::vector<double> bank(static_cast<std::size_t>(shape_numel(full)));
    std::mt19937 rng(seed);
    constexpr double tau = 6.283185307179586476925286766559;
    for (std::size_t i = 0; i < bank.size(); i += 2) {
        const double u = (double(rng()) + .5) / 4294967296.;
        const double v = (double(rng()) + .5) / 4294967296.;
        const double radius = std::sqrt(-2 * std::log(u));
        bank[i] = radius * std::cos(tau * v);
        if (i + 1 < bank.size()) bank[i + 1] = radius * std::sin(tau * v);
    }
    return from_vector(bank, full);
}

/// @brief Add a supplied Gaussian bank to clean unit-range pixels, round to Float32 and clamp.
///
/// Computes clamp(float(double(clean)+double(sigma)*noise),0,1), preserving the
/// source examples' Float32 sigma and rounding order, including signed zero.
/// Values outside Float32's finite range saturate before narrowing to avoid an
/// out-of-range floating conversion. Clipping changes the Gaussian corruption
/// law, including point masses at zero and one. Input noise is explicit: this
/// function never samples or changes a generator. Strided inputs are accepted;
/// values are copied, never mutated. The result is detached augmentation data,
/// with no input/noise autograd history. Noisy observations are inputs to a
/// separately evaluated model; the unchanged clean image remains the target.
/// Work/storage O(clean.numel()); no mutable shared state is used.
///
/// @param clean Float32 unit-range finite tensor of rank at least two; empty axes allowed.
/// @param standard_noise Finite Float64 tensor with shape [1,clean.shape()...].
/// @param sigma Finite nonnegative Float32 noise deviation in original pixel units.
/// @return Newly owned contiguous Float32 tensor with clean's shape, in [0,1].
/// @throws std::invalid_argument For undefined/wrong-shape/wrong-dtype tensors,
///         invalid pixel/noise values or a negative/nonfinite sigma.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// const auto clean = n::tensor({0.,.2,.8,1.},{2,2});
/// const auto z = n::tensor({-2.,1.,-1.,2.},{1,2,2},n::DType::Float64);
/// const auto noisy = n::models::CorruptForDenoising(clean,z,.3f);
/// // noisy is approximately [[0,.5],[.5,1]]. Clean is unchanged.
/// @endcode
///
/// @par References
/// - Vincent, Larochelle, Bengio and Manzagol (2008), Extracting and Composing
///   Robust Features with Denoising Autoencoders, https://doi.org/10.1145/1390156.1390294 .
///   This implementation uses additive clipped Gaussian corruption; it is not
///   the paper's masking-noise benchmark or a claim of universal denoising.
/// @see GaussianNoise(), UndercompleteAutoencoder(), mse()
inline Tensor CorruptForDenoising(const Tensor& clean, const Tensor& standard_noise, float sigma) {
    if (!clean.defined() || !standard_noise.defined() || clean.dtype() != DType::Float32 ||
        clean.dim() < 2 || standard_noise.dtype() != DType::Float64 || !std::isfinite(sigma) || sigma < 0)
        throw std::invalid_argument("denoising corruption requires finite nonnegative sigma, Float32 pixels and Float64 noise");
    Shape shape{1};
    shape.insert(shape.end(), clean.shape().begin(), clean.shape().end());
    if (standard_noise.shape() != shape)
        throw std::invalid_argument("denoising corruption requires one aligned [1,clean.shape...] noise bank");
    const auto x = clean.contiguous().to_vector<float>();
    const auto z = standard_noise.contiguous().to_vector<double>();
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || x[i] < 0 || x[i] > 1 || !std::isfinite(z[i]))
            throw std::invalid_argument("denoising corruption requires finite unit-range pixels and finite noise");
        const double value = double(x[i]) + double(sigma) * z[i];
        const double bound = std::numeric_limits<float>::max();
        out[i] = value < -bound ? 0.f : value > bound ? 1.f : std::clamp(float(value), 0.f, 1.f);
    }
    return from_vector(out, clean.shape());
}

/// @brief Own the hidden map and reconstructed image from a spatial decoder.
///
/// Both handles retain autograd history; the caller owns parameters and fitting.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(1801);
/// const auto decoded = dm::nn::models::ConvolutionalDecoder(p, dm::nn::zeros({2, 7, 7, 4}));
/// // decoded.hidden is [2,14,14,8]; reconstruction is [2,28,28,1].
/// @endcode
/// @see ConvolutionalDecoder()
struct ConvolutionalDecoderOutput {
    Tensor hidden;         ///< First decoder's ReLU map [N,2H,2W,8].
    Tensor reconstruction; ///< Sigmoid image [N,4H,4W,1].
};

/// @brief Decode four-channel spatial codes by nearest repetition and two convolutions.
///
/// Repeats each spatial value into a 2-by-2 block, applies a same-padded 3-by-3
/// 4-to-8 ReLU convolution, repeats again, then applies a same-padded 3-by-3
/// 8-to-1 sigmoid convolution. Creates/reuses decoder1/W,b (He uniform/zeros)
/// and decoder2/W,b (Glorot uniform/zeros): 369 independent scalars. No tied
/// filters, transposed convolution, fitting, clipping or input mutation.
/// Strided Float32/64 inputs retain gradients. Shared stores need external
/// synchronization; allocation/backend errors propagate from tensor operations.
///
/// @param parameters Mutable Float32/64 store matching code's dtype.
/// @param code Nonempty floating NHWC tensor [N,H,W,4], with positive H,W and
///             representable fourfold spatial dimensions.
/// @return Differentiable intermediate map and reconstructed image.
/// @throws std::invalid_argument For wrong shape/dtype or incompatible stored leaves.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore p(1801);
/// const auto result = n::models::ConvolutionalDecoder(p, n::ones({1, 2, 3, 4}));
/// // result.reconstruction is [1,8,12,1]; p.count() is 369.
/// @endcode
///
/// @par References
/// - Masci, Meier, Ciresan and Schmidhuber (2011), Stacked Convolutional
///   Auto-Encoders for Hierarchical Feature Extraction,
///   https://people.idsia.ch/~ciresan/data/icann2011.pdf . This repository's
///   variant uses untied filters, average pooling and nearest repetition;
///   it does not reproduce the paper's stacked pretraining experiments.
/// @see ConvolutionalAutoencoder(), upsample_nearest(), conv2d()
inline ConvolutionalDecoderOutput ConvolutionalDecoder(ParameterStore& parameters, const Tensor& code) {
    if (!code.defined() || code.dim() != 4 || code.size(0) <= 0 ||
        code.size(1) <= 0 || code.size(2) <= 0 || code.size(3) != 4 ||
        code.size(1) > std::numeric_limits<std::int64_t>::max() / 4 ||
        code.size(2) > std::numeric_limits<std::int64_t>::max() / 4 ||
        (code.dtype() != DType::Float32 && code.dtype() != DType::Float64) ||
        parameters.dtype() != code.dtype())
        throw std::invalid_argument("convolutional decoder requires matching floating nonempty [N,H,W,4]");
    ConvLayerOptions relu;
    relu.init = init::he_uniform();
    relu.activation = Activation::Relu;
    const auto hidden = conv2d(parameters, "decoder1", upsample_nearest(code, 2), 8, 3, relu);
    ConvLayerOptions output;
    output.activation = Activation::Sigmoid;
    return {hidden, conv2d(parameters, "decoder2", upsample_nearest(hidden, 2), 1, 3, output)};
}

/// @brief Own all six activation endpoints of a convolutional autoencoder.
///
/// No endpoint is detached. The value owns tensor handles, not an optimizer.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(1801);
/// const auto net = dm::nn::models::ConvolutionalAutoencoder(p, dm::nn::zeros({1, 28, 28, 1}));
/// // net.code is [1,7,7,4]; reconstruction is [1,28,28,1].
/// @endcode
/// @see ConvolutionalAutoencoder()
struct ConvolutionalAeOutput {
    Tensor conv1;          ///< First ReLU feature map [N,H,W,8].
    Tensor pool1;          ///< First nonoverlapping mean pool [N,H/2,W/2,8].
    Tensor conv2;          ///< Second ReLU feature map [N,H/2,W/2,4].
    Tensor code;           ///< Spatial bottleneck [N,H/4,W/4,4].
    Tensor decoder1;       ///< First decoder ReLU feature map [N,H/2,W/2,8].
    Tensor reconstruction; ///< Sigmoid reconstruction [N,H,W,1].
};

/// @brief Encode grayscale images into spatial codes and reconstruct with untied convolutions.
///
/// Uses same-padded 3-by-3 convolutions 1-to-8 and 8-to-4, each with ReLU and
/// nonoverlapping 2-by-2 average pooling, then ConvolutionalDecoder(). The eight
/// parameter tensors encoder1/W,b, encoder2/W,b, decoder1/W,b, decoder2/W,b
/// contain 741 scalars, independent of image dimensions. ReLU layers initialize
/// with He uniform and zero bias; the output uses Glorot uniform and zero bias.
/// Input/code coordinates differ fourfold in count, without quantization or a
/// bit-rate claim. No skip path, fitting, loss, clipping or input mutation is
/// hidden here. The caller chooses the target and training loop. Finite pixel
/// values in [0,1] are conventional, but values are not range-checked. Supports
/// strided Float32/64 tensors and input/parameter gradients. Invalid input is
/// rejected before any parameter creation; incompatible existing leaves can
/// throw after earlier layers have created their parameters. Shared stores need
/// external synchronization. Arithmetic and activation storage scale as O(NHW).
///
/// @param parameters Mutable floating store matching image's dtype.
/// @param image Nonempty NHWC [N,H,W,1], H and W positive multiples of four.
/// @return All differentiable encoder/decoder endpoints, including the spatial code.
/// @throws std::invalid_argument For wrong shape/dtype or incompatible stored leaves.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore p(1801);
/// const auto x = n::full({2, 8, 12, 1}, .25);
/// const auto net = n::models::ConvolutionalAutoencoder(p, x);
/// const auto loss = n::mse(net.reconstruction, x);
/// // net.code is [2,2,3,4]; p.count() is 741. The caller owns Adam.
/// @endcode
///
/// @par References
/// - Masci, Meier, Ciresan and Schmidhuber (2011), Stacked Convolutional
///   Auto-Encoders for Hierarchical Feature Extraction,
///   https://people.idsia.ch/~ciresan/data/icann2011.pdf . This small untied
///   average-pooling/MSE variant is not the paper's architecture or benchmark.
/// @see ConvolutionalAeOutput, ConvolutionalDecoder(), mse()
inline ConvolutionalAeOutput ConvolutionalAutoencoder(ParameterStore& parameters, const Tensor& image) {
    if (!image.defined() || image.dim() != 4 || image.size(0) <= 0 ||
        image.size(1) <= 0 || image.size(2) <= 0 || image.size(1) % 4 != 0 ||
        image.size(2) % 4 != 0 || image.size(3) != 1 ||
        (image.dtype() != DType::Float32 && image.dtype() != DType::Float64) ||
        parameters.dtype() != image.dtype())
        throw std::invalid_argument("convolutional autoencoder requires matching floating [N,H,W,1], H,W positive multiples of 4");
    ConvLayerOptions relu;
    relu.init = init::he_uniform();
    relu.activation = Activation::Relu;
    ConvolutionalAeOutput out;
    out.conv1 = conv2d(parameters, "encoder1", image, 8, 3, relu);
    out.pool1 = avg_pool2d(out.conv1, 2, 2);
    out.conv2 = conv2d(parameters, "encoder2", out.pool1, 4, 3, relu);
    out.code = avg_pool2d(out.conv2, 2, 2);
    const auto decoded = ConvolutionalDecoder(parameters, out.code);
    out.decoder1 = decoded.hidden;
    out.reconstruction = decoded.reconstruction;
    return out;
}

/// @brief Own the code and reconstruction handles of a dense autoencoder.
///
/// Both tensors retain their autograd history. The parameter store owns the
/// learned leaves; this value neither fits the model nor changes its parameters.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore parameters(17);
/// const auto result = dm::nn::models::UndercompleteAutoencoder(
///     parameters, dm::nn::ones({3, 4}), 4, 2);
/// // result.code is [3,2]; result.reconstruction is [3,4].
/// @endcode
/// @see UndercompleteAutoencoder()
struct DenseAeOutput {
    Tensor code;           ///< ReLU bottleneck [N,latent].
    Tensor reconstruction; ///< Sigmoid reconstruction [N,features], in [0,1] for finite logits.
};

/// @brief Encode through a strict ReLU bottleneck and decode with a sigmoid layer.
///
/// Computes ReLU(input*W_e+b_e), then sigmoid(code*W_d+b_d), using independent
/// encoder/decoder weights. Creates or reuses encoder/W, encoder/b, decoder/W
/// and decoder/b with Glorot-uniform weights and zero biases. There are
/// 2*features*latent+latent+features scalars. Input is never clipped or mutated;
/// finite normalized pixels are the usual input, but no value range is imposed.
/// Work is O(N*features*latent). Empty batches and nonfloating/mismatched dtypes
/// are rejected before creating parameters. A shared store needs external
/// synchronization. This forward function performs no optimization.
///
/// @param parameters Mutable Float32/64 store, with the same dtype as input.
/// @param input Floating nonempty matrix [N,features].
/// @param features Input/output width greater than one; defaults to 784.
/// @param latent Positive bottleneck width strictly below features; defaults to 32.
/// @return The differentiable code and reconstruction, retaining input/store gradients.
/// @throws std::invalid_argument For invalid shape, widths, dtype or incompatible stored leaves.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore parameters(17);
/// const auto input = n::full({3, 4}, .25);
/// const auto result = n::models::UndercompleteAutoencoder(parameters, input, 4, 2);
/// const auto objective = n::mse(result.reconstruction, input);
/// // The caller chooses the optimizer and owns the training loop.
/// @endcode
///
/// @par References
/// - Goodfellow, Bengio and Courville (2016), Deep Learning, section 14.1,
///   https://www.deeplearningbook.org/contents/autoencoders.html .
///   This variant uses a ReLU encoder and sigmoid decoder rather than a linear PCA model.
/// @see DenseAeOutput, nn::dense(), nn::mse()
inline DenseAeOutput UndercompleteAutoencoder(ParameterStore& parameters, const Tensor& input,
                                              std::int64_t features = 784,
                                              std::int64_t latent = 32) {
    if (features <= 1 || latent <= 0 || latent >= features || !input.defined() || input.dim() != 2 ||
        input.size(0) <= 0 || input.size(1) != features ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) ||
        parameters.dtype() != input.dtype())
        throw std::invalid_argument("undercomplete autoencoder requires floating [N,features] and 0 < latent < features");
    const auto code = dense(parameters, "encoder", input, latent, Activation::Relu);
    return {code, dense(parameters, "decoder", code, features, Activation::Sigmoid)};
}

/// @brief Decode signed codes through the untied undercomplete autoencoder's sigmoid layer.
///
/// Creates/reuses decoder/W[latent,features] and decoder/b[features] only, using
/// ordinary Glorot-uniform/zero initialization. Default widths give two leaves /
/// 25,872 scalars. Existing full-model stores keep their encoder leaves unchanged.
/// Code-only inference needs no images or labels and performs no fitting,
/// clipping or optimizer update. Caller restores learned arrays before deployed
/// inference; a decoder initialized alone consumes a different RNG sequence from
/// the full autoencoder. Finite signed strided matching Float32/64 CPU codes
/// retain code/decoder first and higher derivatives; outputs own lifetime. All
/// input checks precede leaf creation; incompatible existing leaves may fail
/// later. Computed overflow is not repaired. Shared stores require external
/// synchronization. Work O(B*latent*features), activation storage O(B*features).
/// @param parameters Mutable matching floating store, creating/reusing decoder leaves.
/// @param code Finite signed code matrix[B,latent], B positive.
/// @param features Output width>1, default784.
/// @param latent Positive code width<features, default32.
/// @return Owning sigmoid reconstructions[B,features], with code/decoder paths.
/// @throws std::invalid_argument For invalid geometry/dtype/values/products or incompatible leaves.
/// @throws std::logic_error For non-CPU code storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore parameters(1471);
/// const auto full=n::models::UndercompleteAutoencoder(parameters,n::ones({2,4}),4,2);
/// const auto decoded=n::models::UndercompleteDecoder(parameters,full.code,4,2);
/// // decoded matches full.reconstruction; only codes enter this second call.
/// @endcode
/// @par References
/// - Goodfellow, Bengio and Courville (2016), Deep Learning, section14.1,
///   https://www.deeplearningbook.org/contents/autoencoders.html . The migrated
///   teaching variant uses a ReLU encoder and an untied sigmoid decoder.
/// @see UndercompleteAutoencoder(), DenseAeOutput, LatentFlatten()
inline Tensor UndercompleteDecoder(ParameterStore& parameters,const Tensor& code,
                                   std::int64_t features=784,std::int64_t latent=32) {
    const auto limit=std::numeric_limits<std::int64_t>::max();
    if(features<=1 || latent<=0 || latent>=features || !code.defined() || code.dim()!=2 ||
       code.size(0)<=0 || code.size(1)!=latent || parameters.dtype()!=code.dtype() ||
       (code.dtype()!=DType::Float32 && code.dtype()!=DType::Float64) || features>limit/latent || code.size(0)>limit/features)
        throw std::invalid_argument("undercomplete decoder requires finite matching signed code rows and strict bottleneck widths");
    for(double value:code.contiguous().to_vector<double>())
        if(!std::isfinite(value))throw std::invalid_argument("undercomplete decoder requires finite codes");
    return dense(parameters,"decoder",code,features,Activation::Sigmoid);
}

/// @brief Own the exact squared sigmoid-encoder Jacobian norms and their batch mean.
///
/// Handles retain differentiation through the activations and encoder weights;
/// no Jacobian tensor is allocated and no parameters are changed.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// const auto w = n::ones({3, 2});
/// const auto h = n::full({4, 2}, .5);
/// const auto penalty = n::models::SigmoidEncoderContraction(h, w);
/// // penalty.mean is 0.375; every per_example entry is 0.375.
/// @endcode
/// @see SigmoidEncoderContraction(), ContractiveAutoencoder()
struct ContractivePenalty {
    Tensor per_example; ///< [N] sums of all K*D squared encoder derivatives, not coordinate means.
    Tensor mean;        ///< Scalar mean of per_example over the nonempty batch.
};

/// @brief Compute the exact squared Frobenius norm of a sigmoid encoder's input Jacobian.
///
/// For code=sigmoid(input*encoder_weight+bias), computes
/// sum_j(code_j*(1-code_j))^2*sum_i(encoder_weight_ij)^2 for each row.
/// Both paths through code and weights remain differentiable, including higher
/// derivatives. Inputs are not detached, clipped or mutated. Supports strided
/// tensors. Work O(D*K+N*K), auxiliary output/intermediate storage O(N*K+K);
/// the full [N,K,D] Jacobian is not materialized. Values are not range/finiteness
/// checked: callers must supply actual sigmoid activations for this interpretation;
/// nonfinite values propagate through tensor arithmetic. No shared mutable state.
/// @param code Float32/64 nonempty matrix [N,K] of sigmoid encoder activations.
/// @param encoder_weight Same-dtype nonempty matrix [D,K], with input features on rows.
/// @return Per-observation squared norms and their batch mean, retaining both gradient paths.
/// @throws std::invalid_argument For undefined/nonmatrix/empty inputs, mismatched K or dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// auto w = n::ones({3, 2}).requires_grad_();
/// const auto code = n::sigmoid(n::matmul(n::zeros({4, 3}), w));
/// const auto penalty = n::models::SigmoidEncoderContraction(code, w);
/// const auto derivative = n::grad(penalty.mean, {w})[0];
/// // All six entries of derivative are 0.125.
/// @endcode
/// @par References
/// - Rifai, Vincent, Muller, Glorot and Bengio (2011), Contractive Auto-Encoders:
///   Explicit Invariance During Feature Extraction, section 4,
///   https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf .
/// @see ContractivePenalty, ContractiveAutoencoder()
inline ContractivePenalty SigmoidEncoderContraction(const Tensor& code,
                                                     const Tensor& encoder_weight) {
    if (!code.defined() || !encoder_weight.defined() || code.dim() != 2 ||
        encoder_weight.dim() != 2 || code.size(0) <= 0 || code.size(1) <= 0 ||
        encoder_weight.size(0) <= 0 || encoder_weight.size(1) != code.size(1) ||
        (code.dtype() != DType::Float32 && code.dtype() != DType::Float64) ||
        encoder_weight.dtype() != code.dtype())
        throw std::invalid_argument("sigmoid encoder contraction requires matching floating [N,K] and [D,K]");
    const auto slope = code * (1. - code);
    const auto column_norms = sum(square(encoder_weight), {0});
    const auto per_example = sum(square(slope) * column_norms, {1});
    return {per_example, nn::mean(per_example, {0})};
}

/// @brief Own a sigmoid autoencoder's forward values and full contractive objective.
///
/// Tensor handles retain the autograd graph and share referenced storage.
/// Learned leaves remain owned by the caller's store; no fitting is performed.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(1753);
/// const auto result = dm::nn::models::ContractiveAutoencoder(p, dm::nn::zeros({2, 4}), .01, 4, 2);
/// // result.loss = result.mse + 0.01 * result.penalty.mean.
/// @endcode
/// @see ContractiveAutoencoder(), ContractivePenalty
struct ContractiveOutput {
    Tensor code;           ///< Sigmoid encoder activations [N,latent].
    Tensor reconstruction; ///< Sigmoid decoder predictions [N,features].
    Tensor loss;           ///< Scalar MSE plus weight times mean squared Jacobian norm.
    Tensor mse;            ///< Scalar squared reconstruction error averaged over N*features entries.
    ContractivePenalty penalty; ///< Exact encoder sensitivity in the input's coordinate units.
};

/// @brief Evaluate an untied sigmoid autoencoder with its exact encoder contraction penalty.
///
/// Creates/reuses encoder/W, encoder/b, decoder/W and decoder/b with independent
/// Glorot-uniform weights and zero biases. Both layers use sigmoid; the objective
/// is mean(square(reconstruction-input)) + weight*penalty.mean. Derivatives flow
/// through code and encoder weights. Does not fit, detach or mutate the input.
/// There are 2*features*latent+latent+features learned scalars. Any positive latent
/// width is accepted, including overcomplete encoders. Float32/64 input/store
/// dtypes must agree. Undefined/empty inputs and invalid settings are rejected
/// before parameter creation. Input values are not range/finiteness checked.
/// Work O(N*features*latent), intermediate storage O(N*(features+latent)); no full
/// Jacobian allocation. A shared parameter store needs external synchronization.
/// @param parameters Mutable same-dtype learned parameter store.
/// @param input Floating nonempty matrix [N,features], conventionally pixels in [0,1].
/// @param weight Finite nonnegative penalty coefficient; zero keeps the diagnostic but removes its loss contribution.
/// @param features Positive input/output width, default 784.
/// @param latent Positive sigmoid code width, default 32.
/// @return Forward code/reconstruction, MSE, exact penalty and total objective.
/// @throws std::invalid_argument For invalid settings, shape, dtype or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore p(1753);
/// const auto result = n::models::ContractiveAutoencoder(p, n::full({2, 4}, .25), .01, 4, 2);
/// const auto derivatives = n::grad(result.loss, p.parameters());
/// // The caller owns the optimizer and the minibatch loop.
/// @endcode
/// @par References
/// - Rifai et al. (2011), https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf .
///   Retains the migrated example's untied weights, MSE and 32-unit bottleneck;
///   the paper's experiments use tied weights and cross-entropy reconstruction.
/// @see SigmoidEncoderContraction(), UndercompleteAutoencoder()
inline ContractiveOutput ContractiveAutoencoder(ParameterStore& parameters, const Tensor& input,
                                                double weight, std::int64_t features = 784,
                                                std::int64_t latent = 32) {
    if (!std::isfinite(weight) || weight < 0 || features <= 0 || latent <= 0 ||
        !input.defined() || input.dim() != 2 || input.size(0) <= 0 || input.size(1) != features ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) ||
        parameters.dtype() != input.dtype())
        throw std::invalid_argument("contractive autoencoder requires matching floating [N,features] and valid widths/weight");
    ContractiveOutput out;
    out.code = dense(parameters, "encoder", input, latent, Activation::Sigmoid);
    out.reconstruction = dense(parameters, "decoder", out.code, features, Activation::Sigmoid);
    out.mse = mean(square(out.reconstruction - input));
    out.penalty = SigmoidEncoderContraction(out.code, parameters.at("encoder/W"));
    out.loss = out.mse + weight * out.penalty.mean;
    return out;
}

} // namespace datamunge::nn::models
