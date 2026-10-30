#pragma once
/// @file inception_models.hpp
/// @brief Full-depth Inception v2/v3/v4 and Inception-ResNet-v2 tensor definitions.
#include "datamunge/nn/vision_architecture_components.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Version, geometry and batch-normalisation momentum for Inception v2/v3.
///
/// Only value metadata; default v3 uses reference 299-pixel geometry. Width
/// scaling changes channel widths but retains every block. No weights are fitted.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception23()
struct Inception23Config {
  int version = 3;       ///< Exactly 2 (BN-Inception) or 3.
  int image_size = 299;  ///< Square RGB side: v2 32..299, v3 107..299; reference v2 uses 224.
  int classes = 1000;    ///< At least two class logits.
  int width_divisor = 1; ///< One, two, four or eight; divides learned output widths.
  double momentum = .9997; ///< Finite BN old-value EMA retention in [0,1).

  /// @brief Validate version, input side, output count, width divisor and BN momentum.
  ///
  /// @throws std::invalid_argument For a setting outside its documented field domain.
  /// @par Example
  /// @code
  /// dm::nn::models::Inception23Config{}.Validate();
  /// @endcode
  /// @see Inception23Config
  void Validate() const {
    if ((version != 2 && version != 3) || classes < 2 ||
        image_size < (version == 2 ? 32 : 107) || image_size > 299 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 &&
         width_divisor != 8) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Inception v2/v3 configuration");
  }
  /// @brief Calculate the v3 feature-map side before the auxiliary pooling head.
  ///
  /// Only meaningful for version three; validates the whole configuration first.
  ///
  /// @return Positive side after the first reduction; 17 at 299 pixels, five at 107.
  /// @throws std::invalid_argument For invalid configuration or version two.
  /// @par Example
  /// @code
  /// const auto side = dm::nn::models::Inception23Config{}.AuxiliaryMapSize();
  /// @endcode
  /// @see InceptionAuxiliary()
  int AuxiliaryMapSize() const {
    Validate();
    if (version != 3)
      throw std::invalid_argument("only v3 has this auxiliary map");
    int size = (image_size - 3) / 2 + 1;
    size -= 2;
    size = (size - 3) / 2 + 1;
    size -= 2;
    size = (size - 3) / 2 + 1;
    return (size - 3) / 2 + 1;
  }
};

/// @brief Named intermediate maps, main logits and optional auxiliary logits.
///
/// Tensor handles retain their storage and autograd history. Parameters and
/// moving moments remain in their caller-owned stores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// const auto out = n::models::Inception23(p, state, n::zeros({1, 107, 107, 3}), cfg);
/// @endcode
/// @see Inception23(), Inception4()
struct Inception23Outputs {
  std::vector<std::pair<std::string, nn::Tensor>> stages; ///< Ordered named NHWC maps.
  nn::Tensor logits;                 ///< Main unnormalised [N,classes] scores.
  std::vector<nn::Tensor> auxiliary; ///< Empty unless requested; one [N,classes] head.
};

// Carry channel counts with tensors so branch construction never guesses the
// input width after concatenation. Width scaling applies only to learned outputs.
/// @brief Carry an NHWC feature map together with its positive channel-count metadata.
///
/// The default empty value/channels=0 is a placeholder; initialize before use.
/// The channels field must equal value.size(3); it is not a learned parameter.
///
/// @par Example
/// @code
/// const dm::nn::models::ConvFeature x{dm::nn::ones({1, 5, 5, 8}), 8};
/// @endcode
/// @see InceptionBuilder
struct ConvFeature {
  nn::Tensor value; ///< Floating NHWC activation; storage/graph retained by its handle.
  int channels = 0; ///< Matching actual channel count, positive once initialized.
};

/// @brief Compose reference Inception branches while tracking widths and stage names.
///
/// Borrows p, state, out and cfg; all must outlive the builder, so do not pass a
/// temporary cfg. Width scaling applies only to learned convolution outputs.
/// The two stores must be distinct and have the input floating dtype. Calls
/// create/reuse leaves and optionally update BN state; they never fit parameters.
/// A shared builder/stores are not safe for concurrent mutation.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// n::models::Inception23Outputs out;
/// n::models::InceptionBuilder b(p, state, out, cfg, false);
/// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
/// const auto y = b.Conv("c", x, 32, 1, 3);
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception23(), Inception4()
class InceptionBuilder {
public:
  /// @brief Borrow stores, output recorder and validated layer configuration.
  ///
  /// Initializes beta-only BN with epsilon .001 and cfg.momentum; no fitting.
  ///
  /// @param p Mutable learnable parameter store.
  /// @param state Separate mutable BN moving-statistic store.
  /// @param out Mutable ordered stage recorder.
  /// @param cfg Configuration borrowed by reference; must outlive this builder.
  /// @param training Use batch moments and eager moving-state updates when true.
  /// @throws std::invalid_argument If cfg.Validate rejects a setting.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// @endcode
  /// @see Conv(), Stage()
  InceptionBuilder(nn::ParameterStore &p, nn::ParameterStore &state, Inception23Outputs &out,
                   const Inception23Config &cfg, bool training)
      : p_(p), state_(state), out_(out), cfg_(cfg) {
    cfg.Validate();
    norm_.training = training;
    norm_.momentum = cfg.momentum;
  }
  /// @brief Borrow the learnable store used by this builder.
  ///
  /// No copy or allocation; callers are responsible for coordinating mutations.
  ///
  /// @return Mutable reference valid while the caller-owned p remains alive.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto count = b.params().count();
  /// @endcode
  /// @see InceptionBuilder
  nn::ParameterStore &params() {
    return p_;
  }
  /// @brief Apply a width-scaled rectangular convolution, beta-only BN and ReLU.
  ///
  /// Uses the builder BN training mode and eager state updates; no fitting.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @param channels Positive reference output width, exactly divisible by cfg.width_divisor.
  /// @param h Positive kernel height, default one.
  /// @param w Positive kernel width, default one.
  /// @param stride Positive common stride, default one.
  /// @param padding SAME (default) or VALID, case-sensitive.
  /// @return Feature preserving its graph, with channels/cfg.width_divisor outputs.
  /// @throws std::invalid_argument For inexact/invalid widths or invalid convolution, BN, tensor or store settings.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.Conv("c", x, 32, 1, 3);
  /// @endcode
  /// @see NormalizedConv()
  ConvFeature Conv(const std::string &name, const ConvFeature &x, int channels, int h = 1,
                   int w = 1, int stride = 1, const std::string &padding = "SAME") {
    if (channels <= 0 || channels % cfg_.width_divisor)
      throw std::invalid_argument("Inception output width must divide exactly");
    channels /= cfg_.width_divisor;
    return {NormalizedConv(p_, state_, name, x.value, channels, h, w, norm_, stride, padding),
            channels};
  }
  /// @brief Pool spatially without changing feature channel metadata.
  ///
  /// Average-pool edge handling follows the core pooling definition.
  ///
  /// @param x Initialized floating NHWC feature with matching positive channels.
  /// @param maximum Choose max-pool if true, average-pool if false (default).
  /// @param window Positive square pooling window, default three.
  /// @param stride Positive common spatial stride, default one.
  /// @param padding SAME (default) or VALID, case-sensitive.
  /// @return Pooled feature with x.channels and retained differentiation graph.
  /// @throws std::invalid_argument For invalid padding, geometry or tensor dtype.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.Pool(x, true, 3, 2);
  /// @endcode
  /// @see Conv()
  ConvFeature Pool(const ConvFeature &x, bool maximum = false, int window = 3, int stride = 1,
                   const std::string &padding = "SAME") {
    const auto pad = ParsePadding(padding);
    return {maximum ? nn::max_pool2d(x.value, window, stride, pad)
                    : nn::avg_pool2d(x.value, window, stride, pad),
            x.channels};
  }
  /// @brief Concatenate spatially matching branches and sum their channel metadata.
  ///
  /// Does not width-scale existing tensors or mutate its arguments.
  ///
  /// @param features Nonempty floating NHWC features with matching N,H,W,dtype and valid metadata.
  /// @return Channel-concatenated feature in the supplied order, with retained graph.
  /// @throws std::invalid_argument For empty or incompatible concatenation inputs.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.Cat({x, x});
  /// @endcode
  /// @see ConvFeature
  ConvFeature Cat(const std::vector<ConvFeature> &features) {
    std::vector<nn::Tensor> values;
    int channels = 0;
    for (const auto &feature : features) {
      values.push_back(feature.value);
      channels += feature.channels;
    }
    return {nn::concat(values, 3), channels};
  }
  /// @brief Append a named map to the output recorder and return the same feature.
  ///
  /// Appends to out.stages; no tensor computation or parameter creation.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @return Feature x by value; tensor storage and graph remain shared.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.Stage("example", x);
  /// @endcode
  /// @see Inception23Outputs
  ConvFeature Stage(const std::string &name, const ConvFeature &x) {
    out_.stages.emplace_back(name, x.value);
    return x;
  }
  /// @brief Apply BN-Inception branches with one, three and two stacked 3x3 filters.
  ///
  /// Four SAME branches concatenate; the double-3x3 path adds a second BN/ReLU.
  /// All learned output widths are divided exactly; spatial size is retained.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @param w Seven reference widths: one, three_reduce, three, double_reduce, double1, double2, pool.
  /// @param max_pool Choose max-pool on the projected pooling branch when true; average by default.
  /// @return Width-scaled NHWC feature, also appended to the stage recorder.
  /// @throws std::invalid_argument For invalid widths, geometry, tensor/store settings or incompatible branches.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.V2Module("m", x, {64, 64, 64, 64, 96, 96, 32});
  /// @endcode
  /// @par References
  /// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
  /// @see InceptionBuilder
  ConvFeature V2Module(const std::string &name, const ConvFeature &x,
                       const std::array<int, 7> &w, bool max_pool = false) {
    const auto a = Conv(name + "/one", x, w[0]);
    auto b = Conv(name + "/three_reduce", x, w[1]);
    b = Conv(name + "/three", b, w[2], 3, 3);
    auto c = Conv(name + "/double_reduce", x, w[3]);
    c = Conv(name + "/double1", c, w[4], 3, 3);
    c = Conv(name + "/double2", c, w[5], 3, 3);
    const auto d = Conv(name + "/pool", Pool(x, max_pool), w[6]);
    return Stage(name, Cat({a, b, c, d}));
  }
  /// @brief Downsample through two convolution paths and a max-pool path.
  ///
  /// SAME stride-two convolution and pooling paths give ceil(H/2),ceil(W/2).
  /// Output channels include the unchanged input-channel max-pool branch.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @param w Five reference widths: three_reduce, three, double_reduce, double1, double2.
  /// @return Width-scaled NHWC feature, also appended to the stage recorder.
  /// @throws std::invalid_argument For invalid widths, geometry, tensor/store settings or incompatible branches.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.V2Reduction("r", x, {128, 160, 64, 96, 96});
  /// @endcode
  /// @par References
  /// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
  /// @see InceptionBuilder
  ConvFeature V2Reduction(const std::string &name, const ConvFeature &x,
                          const std::array<int, 5> &w) {
    auto a = Conv(name + "/three_reduce", x, w[0]);
    a = Conv(name + "/three", a, w[1], 3, 3, 2);
    auto b = Conv(name + "/double_reduce", x, w[2]);
    b = Conv(name + "/double1", b, w[3], 3, 3);
    b = Conv(name + "/double2", b, w[4], 3, 3, 2);
    return Stage(name, Cat({a, b, Pool(x, true, 3, 2)}));
  }
  /// @brief Apply the v3 A module with reduced 5x5 and stacked 3x3 branches.
  ///
  /// Four SAME-resolution branches follow TF-Slim widths and beta-only BN/ReLU.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @param pool Positive reference width of the projected average-pooling branch.
  /// @return Width-scaled NHWC feature, also appended to the stage recorder.
  /// @throws std::invalid_argument For invalid widths, geometry, tensor/store settings or incompatible branches.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.V3A("a", x, 32);
  /// @endcode
  /// @par References
  /// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
  /// @see InceptionBuilder
  ConvFeature V3A(const std::string &name, const ConvFeature &x, int pool) {
    const auto a = Conv(name + "/one", x, 64);
    auto b = Conv(name + "/five_reduce", x, 48);
    b = Conv(name + "/five", b, 64, 5, 5);
    auto c = Conv(name + "/double_reduce", x, 64);
    c = Conv(name + "/double1", c, 96, 3, 3);
    c = Conv(name + "/double2", c, 96, 3, 3);
    const auto d = Conv(name + "/pool", Pool(x), pool);
    return Stage(name, Cat({a, b, c, d}));
  }
  /// @brief Apply the v3 B module with asymmetric 1x7 and 7x1 paths.
  ///
  /// All four paths preserve spatial size. Intermediate BN/ReLU means the
  /// asymmetric pair is not a single linear factored 7x7 filter.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @param width Positive reference bottleneck width, divisible by the configured divisor.
  /// @return Width-scaled NHWC feature, also appended to the stage recorder.
  /// @throws std::invalid_argument For invalid widths, geometry, tensor/store settings or incompatible branches.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.V3B("b", x, 128);
  /// @endcode
  /// @par References
  /// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
  /// @see InceptionBuilder
  ConvFeature V3B(const std::string &name, const ConvFeature &x, int width) {
    const auto a = Conv(name + "/one", x, 192);
    auto b = Conv(name + "/seven_reduce", x, width);
    b = Conv(name + "/seven_h", b, width, 1, 7);
    b = Conv(name + "/seven_v", b, 192, 7, 1);
    auto c = Conv(name + "/double_reduce", x, width);
    c = Conv(name + "/double_v1", c, width, 7, 1);
    c = Conv(name + "/double_h1", c, width, 1, 7);
    c = Conv(name + "/double_v2", c, width, 7, 1);
    c = Conv(name + "/double_h2", c, 192, 1, 7);
    const auto d = Conv(name + "/pool", Pool(x), 192);
    return Stage(name, Cat({a, b, c, d}));
  }
  /// @brief Apply the v3 C module with split 1x3 and 3x1 branches.
  ///
  /// Two parent branches each split into horizontal/vertical convolutions, then
  /// concatenate with direct and pooling branches; output has 2048/divisor channels.
  ///
  /// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
  /// @param x Initialized NHWC feature with matching positive channel metadata.
  /// @return Width-scaled NHWC feature, also appended to the stage recorder.
  /// @throws std::invalid_argument For invalid widths, geometry, tensor/store settings or incompatible branches.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception23Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// n::ParameterStore p(17), state(17);
  /// n::models::Inception23Outputs out;
  /// n::models::InceptionBuilder b(p, state, out, cfg, false);
  /// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
  /// const auto y = b.V3C("c", x);
  /// @endcode
  /// @par References
  /// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
  /// @see InceptionBuilder
  ConvFeature V3C(const std::string &name, const ConvFeature &x) {
    const auto a = Conv(name + "/one", x, 320), b = Conv(name + "/split_reduce", x, 384);
    auto c = Conv(name + "/double_reduce", x, 448);
    c = Conv(name + "/double", c, 384, 3, 3);
    const auto d = Conv(name + "/pool", Pool(x), 192);
    return Stage(name, Cat({a, Conv(name + "/split_h", b, 384, 1, 3),
                            Conv(name + "/split_v", b, 384, 3, 1),
                            Conv(name + "/double_h", c, 384, 1, 3),
                            Conv(name + "/double_v", c, 384, 3, 1), d}));
  }

private:
  nn::ParameterStore &p_;     ///< Borrowed learnable store; outlives the builder.
  nn::ParameterStore &state_; ///< Borrowed BN moving state, distinct from p_.
  Inception23Outputs &out_;   ///< Borrowed mutable stage recorder.
  const Inception23Config
      &cfg_;            ///< Borrowed immutable validated dimensions; must outlive builder.
  ConvNormConfig norm_; ///< Builder-owned BN mode/EMA settings.
};

/// @brief Global-average a feature map, optionally drop 20%, and form linear class scores.
///
/// No softmax, fitting or BN update; inference does not consume dropout randomness.
///
/// @param p Mutable parameter store; output/W and /b are lazily created/reused.
/// @param features Floating NHWC feature tensor matching p dtype.
/// @param classes Positive output-channel width.
/// @param training Apply inverted dropout with store-owned randomness when true.
/// @return [N,classes] unnormalised logits retaining gradients.
/// @throws std::invalid_argument For invalid shapes, output count, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::InceptionClassifier(p, n::ones({2, 3, 3, 8}), 3, false);
/// @endcode
/// @see Inception23(), Inception4()
inline nn::Tensor InceptionClassifier(nn::ParameterStore &p, const nn::Tensor &features,
                                      int classes, bool training) {
  nn::Tensor pooled = nn::global_avg_pool2d(features);
  if (training)
    pooled = nn::dropout(pooled, .2, true, p.generator());
  return nn::dense(p, "output", pooled, classes);
}

/// @brief Build the reference pooled auxiliary head for v3/v4/Inception-ResNet-v2.
///
/// VALID average pooling 5x5/3, BN/ReLU 1x1 projection to 128/divisor,
/// BN/ReLU convolution covering the remaining map to 768/divisor, then linear
/// logits. No optimizer or loss weighting is hidden here.
///
/// @param b Builder borrowing the parameter and moving-state stores.
/// @param input Auxiliary source map of side cfg.AuxiliaryMapSize(), with valid channels.
/// @param cfg Version-three geometry/dimensions consistent with b; the remaining VALID kernel must be positive.
/// @return [N,cfg.classes] auxiliary logits, with gradients to its source map.
/// @throws std::invalid_argument For invalid configuration, auxiliary geometry, input or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// n::models::Inception23Outputs out;
/// n::models::InceptionBuilder b(p, state, out, cfg, false);
/// const auto y = n::models::InceptionAuxiliary(b, {n::ones({1, 5, 5, 8}), 8}, cfg);
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception23(), Inception4()
inline nn::Tensor InceptionAuxiliary(InceptionBuilder &b, const ConvFeature &input,
                                     const Inception23Config &cfg) {
  auto h = b.Pool(input, false, 5, 3, "VALID");
  h = b.Conv("aux/projection", h, 128);
  const int kernel = (cfg.AuxiliaryMapSize() - 5) / 3 + 1;
  h = b.Conv("aux/spatial", h, 768, kernel, kernel, 1, "VALID");
  return nn::dense(b.params(), "aux/output", nn::reshape(h.value, {-1, h.channels}),
                   cfg.classes);
}

/// @brief Build complete TF-Slim-style Inception v2 or v3 without fitting parameters.
///
/// v2 includes the separable 7x7 bias/ReLU stem without BN, eight ordinary
/// modules and two reductions. v3 includes 3 A, 4 B and 2 split C modules plus
/// two reductions. Main inference does not depend on auxiliary parameters.
/// Uses beta-only BN, epsilon .001, eager unbiased-variance EMA updates,
/// He initialization and caller-managed preprocessing/optimization. No weights
/// are imported; scaled widths preserve depth but are explicitly compact models.
///
/// @param p Mutable learnable store.
/// @param state Separate BN moving moments, eagerly updated only during training.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], matching both stores.
/// @param cfg Valid version, geometry, class count, exact width divisor and BN momentum.
/// @param training Use batch BN moments/state updates and .8-retention dropout when true.
/// @param auxiliary Construct the v3 auxiliary head when true; ignored for v2; default false.
/// @return Named stage maps, main logits and optional auxiliary logits; graphs retained.
/// @throws std::invalid_argument For invalid cfg, RGB geometry, dtype, widths or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// const auto out = n::models::Inception23(p, state, n::zeros({1, 107, 107, 3}), cfg);
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167; Szegedy et al. (2016), https://arxiv.org/abs/1512.00567; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see InceptionBuilder, Inception4()
inline Inception23Outputs Inception23(nn::ParameterStore &p, nn::ParameterStore &state,
                                      const nn::Tensor &image,
                                      const Inception23Config &cfg = {}, bool training = false,
                                      bool auxiliary = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match Inception configuration");
  Inception23Outputs out;
  InceptionBuilder b(p, state, out, cfg, training);
  ConvFeature x{image, 3}, aux_input{};
  if (cfg.version == 2) {
    nn::ConvLayerOptions stem;
    stem.stride = 2;
    stem.activation = nn::Activation::Relu;
    stem.init = nn::init::he_uniform();
    const int channels = 64 / cfg.width_divisor;
    const int multiplier = std::min(channels / 3, 8);
    x = b.Stage(
        "stem7",
        {nn::separable_conv2d(p, "stem7", x.value, channels, 7, stem, multiplier), channels});
    x = b.Stage("pool1", b.Pool(x, true, 3, 2));
    x = b.Stage("stem1", b.Conv("stem1", x, 64));
    x = b.Stage("stem3", b.Conv("stem3", x, 192, 3, 3));
    x = b.Stage("pool2", b.Pool(x, true, 3, 2));
    x = b.V2Module("3b", x, {64, 64, 64, 64, 96, 96, 32});
    x = b.V2Module("3c", x, {64, 64, 96, 64, 96, 96, 64});
    x = b.V2Reduction("4a", x, {128, 160, 64, 96, 96});
    x = b.V2Module("4b", x, {224, 64, 96, 96, 128, 128, 128});
    x = b.V2Module("4c", x, {192, 96, 128, 96, 128, 128, 128});
    x = b.V2Module("4d", x, {160, 128, 160, 128, 160, 160, 96});
    x = b.V2Module("4e", x, {96, 128, 192, 160, 192, 192, 96});
    x = b.V2Reduction("5a", x, {128, 192, 192, 256, 256});
    x = b.V2Module("5b", x, {352, 192, 320, 160, 224, 224, 128});
    x = b.V2Module("5c", x, {352, 192, 320, 192, 224, 224, 128}, true);
  } else {
    x = b.Stage("stem1", b.Conv("stem1", x, 32, 3, 3, 2, "VALID"));
    x = b.Stage("stem2", b.Conv("stem2", x, 32, 3, 3, 1, "VALID"));
    x = b.Stage("stem3", b.Conv("stem3", x, 64, 3, 3));
    x = b.Stage("pool1", b.Pool(x, true, 3, 2, "VALID"));
    x = b.Stage("stem4", b.Conv("stem4", x, 80));
    x = b.Stage("stem5", b.Conv("stem5", x, 192, 3, 3, 1, "VALID"));
    x = b.Stage("pool2", b.Pool(x, true, 3, 2, "VALID"));
    x = b.V3A("5b", x, 32);
    x = b.V3A("5c", x, 64);
    x = b.V3A("5d", x, 64);
    auto a = b.Conv("6a/three", x, 384, 3, 3, 2, "VALID");
    auto c = b.Conv("6a/double_reduce", x, 64);
    c = b.Conv("6a/double1", c, 96, 3, 3);
    c = b.Conv("6a/double2", c, 96, 3, 3, 2, "VALID");
    x = b.Stage("6a", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    x = b.V3B("6b", x, 128);
    x = b.V3B("6c", x, 160);
    x = b.V3B("6d", x, 160);
    x = b.V3B("6e", x, 192);
    aux_input = x;
    a = b.Conv("7a/three_reduce", x, 192);
    a = b.Conv("7a/three", a, 320, 3, 3, 2, "VALID");
    c = b.Conv("7a/seven_reduce", x, 192);
    c = b.Conv("7a/seven_h", c, 192, 1, 7);
    c = b.Conv("7a/seven_v", c, 192, 7, 1);
    c = b.Conv("7a/seven_three", c, 192, 3, 3, 2, "VALID");
    x = b.Stage("7a", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    x = b.V3C("7b", x);
    x = b.V3C("7c", x);
  }
  out.logits = InceptionClassifier(p, x.value, cfg.classes, training);
  if (cfg.version == 3 && auxiliary)
    out.auxiliary.push_back(InceptionAuxiliary(b, aux_input, cfg));
  return out;
}

/// @brief Reference/compact settings for Inception v4 or Inception-ResNet-v2.
///
/// Uses the shared v3 BN/auxiliary configuration while selecting distinct full
/// block counts. Value metadata alone does not allocate or fit parameters.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception4Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// const auto layers = cfg.Layers();
/// @endcode
/// @par References
/// - Szegedy et al. (2017), https://arxiv.org/abs/1602.07261; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception4()
struct Inception4Config {
  bool residual = false;   ///< False selects v4; true selects Inception-ResNet-v2.
  int image_size = 299;    ///< Square RGB side, 107..299; reference 299.
  int classes = 1000;      ///< At least two output logits.
  int width_divisor = 1;   ///< One, two, four or eight; divides all learned output widths.
  double momentum = .9997; ///< Finite BN old-value EMA retention in [0,1).

  /// @brief Return a validated shared layer configuration with version three.
  ///
  /// Does not change this object. Version-three geometry sizes the auxiliary head.
  ///
  /// @return Independent Inception23Config with matching image/classes/width/momentum.
  /// @throws std::invalid_argument For invalid dimensions, width divisor or BN momentum.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::models::Inception4Config cfg;
  /// cfg.image_size = 107;
  /// cfg.classes = 3;
  /// cfg.width_divisor = 8;
  /// const auto layers = cfg.Layers();
  /// @endcode
  /// @see Inception23Config
  Inception23Config Layers() const {
    Inception23Config cfg;
    cfg.image_size = image_size;
    cfg.classes = classes;
    cfg.width_divisor = width_divisor;
    cfg.momentum = momentum;
    cfg.Validate();
    return cfg;
  }
};

/// @brief Apply the v4 asymmetric B block and record its stage.
///
/// Uses the full reference branch layout with beta-only BN/ReLU; no fitting.
///
/// @param b Builder borrowing stores/configuration; output widths scale by its divisor.
/// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
/// @param x Initialized NHWC feature with matching positive channel metadata.
/// @return Same-resolution feature; channels are 1024/divisor.
/// @throws std::invalid_argument For invalid input metadata, convolution/BN geometry or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// n::models::Inception23Outputs out;
/// n::models::InceptionBuilder b(p, state, out, cfg, false);
/// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
/// const auto y = n::models::Inception4B(b, "block", x);
/// @endcode
/// @par References
/// - Szegedy et al. (2017), https://arxiv.org/abs/1602.07261; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception4()
inline ConvFeature Inception4B(InceptionBuilder &b, const std::string &name,
                               const ConvFeature &x) {
  auto a = b.Conv(name + "/one", x, 384);
  auto c = b.Conv(name + "/seven_reduce", x, 192);
  c = b.Conv(name + "/seven_h", c, 224, 1, 7);
  c = b.Conv(name + "/seven_v", c, 256, 7, 1);
  auto d = b.Conv(name + "/double_reduce", x, 192);
  d = b.Conv(name + "/double_v1", d, 192, 7, 1);
  d = b.Conv(name + "/double_h1", d, 224, 1, 7);
  d = b.Conv(name + "/double_v2", d, 224, 7, 1);
  d = b.Conv(name + "/double_h2", d, 256, 1, 7);
  return b.Stage(name, b.Cat({a, c, d, b.Conv(name + "/pool", b.Pool(x), 128)}));
}

/// @brief Apply the v4 split-branch C block and record its stage.
///
/// Uses the full reference branch layout with beta-only BN/ReLU; no fitting.
///
/// @param b Builder borrowing stores/configuration; output widths scale by its divisor.
/// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
/// @param x Initialized NHWC feature with matching positive channel metadata.
/// @return Same-resolution feature; channels are 1536/divisor.
/// @throws std::invalid_argument For invalid input metadata, convolution/BN geometry or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// n::models::Inception23Outputs out;
/// n::models::InceptionBuilder b(p, state, out, cfg, false);
/// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
/// const auto y = n::models::Inception4C(b, "block", x);
/// @endcode
/// @par References
/// - Szegedy et al. (2017), https://arxiv.org/abs/1602.07261; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception4()
inline ConvFeature Inception4C(InceptionBuilder &b, const std::string &name,
                               const ConvFeature &x) {
  auto a = b.Conv(name + "/one", x, 256), c = b.Conv(name + "/split_reduce", x, 384);
  auto d = b.Conv(name + "/double_reduce", x, 384);
  d = b.Conv(name + "/double_v", d, 448, 3, 1);
  d = b.Conv(name + "/double_h", d, 512, 1, 3);
  auto pooled = b.Conv(name + "/pool", b.Pool(x), 256);
  return b.Stage(name, b.Cat({a, b.Conv(name + "/split_h", c, 256, 1, 3),
                              b.Conv(name + "/split_v", c, 256, 3, 1),
                              b.Conv(name + "/double_split_h", d, 256, 1, 3),
                              b.Conv(name + "/double_split_v", d, 256, 3, 1), pooled}));
}

/// @brief Add a scaled linear projection of Inception branches to the identity path.
///
/// Projection is a biased linear 1x1 convolution without BN. All branch kernels
/// retain BN/ReLU. Without final ReLU, scale zero gives an exact identity;
/// parameter derivatives scale with scale and input derivatives retain identity.
/// No empirical stability guarantee is implied by this algebraic property.
///
/// @param b Builder borrowing stores/configuration; branch widths scale by its divisor.
/// @param name Prefix for named layer leaves; repeated names reuse matching store entries.
/// @param x Initialized NHWC feature with matching positive channel metadata.
/// @param kind A, B or C selects the reference residual branch layout.
/// @param scale Finite real residual multiplier, including zero; typical .17/.10/.20.
/// @param relu Apply a final ReLU if true (default); the final C block uses false.
/// @return Feature x + scale*projection(branches(x)), optionally rectified, recorded by name.
/// @throws std::invalid_argument For nonfinite scale, invalid kind, metadata, geometry or parameter entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception23Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// n::models::Inception23Outputs out;
/// n::models::InceptionBuilder b(p, state, out, cfg, false);
/// const n::models::ConvFeature x{n::ones({1, 7, 7, 8}), 8};
/// const auto y = n::models::InceptionResidual(b, "r", x, 'A', .17, false);
/// @endcode
/// @par References
/// - Szegedy et al. (2017), https://arxiv.org/abs/1602.07261; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see Inception4()
inline ConvFeature InceptionResidual(InceptionBuilder &b, const std::string &name,
                                     const ConvFeature &x, char kind, double scale,
                                     bool relu = true) {
  if (!std::isfinite(scale))
    throw std::invalid_argument("finite residual scale required");
  ConvFeature joined;
  if (kind == 'A') {
    auto a = b.Conv(name + "/one", x, 32);
    auto c = b.Conv(name + "/three_reduce", x, 32);
    c = b.Conv(name + "/three", c, 32, 3, 3);
    auto d = b.Conv(name + "/double_reduce", x, 32);
    d = b.Conv(name + "/double1", d, 48, 3, 3);
    d = b.Conv(name + "/double2", d, 64, 3, 3);
    joined = b.Cat({a, c, d});
  } else if (kind == 'B' || kind == 'C') {
    auto a = b.Conv(name + "/one", x, 192);
    auto c = b.Conv(name + "/reduce", x, kind == 'B' ? 128 : 192);
    c = b.Conv(name + "/horizontal", c, kind == 'B' ? 160 : 224, 1, kind == 'B' ? 7 : 3);
    c = b.Conv(name + "/vertical", c, kind == 'B' ? 192 : 256, kind == 'B' ? 7 : 3, 1);
    joined = b.Cat({a, c});
  } else
    throw std::invalid_argument("Inception residual kind must be A, B or C");
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  const auto projected =
      nn::conv2d(b.params(), name + "/project", joined.value, x.channels, 1, linear);
  nn::Tensor result = x.value + scale * projected;
  if (relu)
    result = nn::relu(result);
  return b.Stage(name, {result, x.channels});
}

/// @brief Build complete Inception v4 or Inception-ResNet-v2 without fitting parameters.
///
/// v4 repeats A/B/C blocks 4/7/3 times. The residual variant repeats 10/20/10,
/// scales by .17/.10/.20, and uses scale one without final ReLU in its last
/// C block. Linear residual projections have bias and no BN. Other convolutions
/// use beta-only BN, epsilon .001 and He initialization. Initialization, eager
/// unbiased-variance moving updates and compact training are documented
/// adaptations; no pretrained weights or optimizer are hidden in this builder.
///
/// @param p Mutable learnable parameter store.
/// @param state Separate BN moving state; eager training updates, unchanged in inference.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], matching both stores.
/// @param cfg Valid geometry, width and residual-variant choice.
/// @param training Use batch BN moments/updates and .8-retention main dropout when true.
/// @param auxiliary Build one auxiliary head before the second reduction if true; default false.
/// @return Named stage maps, main [N,classes] scores and optional auxiliary logits.
/// @throws std::invalid_argument For invalid cfg, RGB geometry, dtype, branch widths or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::Inception4Config cfg;
/// cfg.image_size = 107;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// n::ParameterStore p(17), state(17);
/// const auto out = n::models::Inception4(p, state, n::zeros({1, 107, 107, 3}), cfg);
/// @endcode
/// @par References
/// - Szegedy et al. (2017), https://arxiv.org/abs/1602.07261; TF-Slim reference definitions, https://github.com/tensorflow/models/tree/master/research/slim/nets
/// @see InceptionResidual(), Inception23()
inline Inception23Outputs Inception4(nn::ParameterStore &p, nn::ParameterStore &state,
                                     const nn::Tensor &image, const Inception4Config &cfg = {},
                                     bool training = false, bool auxiliary = false) {
  const auto layers = cfg.Layers();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match Inception configuration");
  Inception23Outputs out;
  InceptionBuilder b(p, state, out, layers, training);
  ConvFeature x{image, 3}, aux_input;
  x = b.Stage("stem1", b.Conv("stem1", x, 32, 3, 3, 2, "VALID"));
  x = b.Stage("stem2", b.Conv("stem2", x, 32, 3, 3, 1, "VALID"));
  x = b.Stage("stem3", b.Conv("stem3", x, 64, 3, 3));
  if (cfg.residual) {
    x = b.Stage("pool1", b.Pool(x, true, 3, 2, "VALID"));
    x = b.Stage("stem4", b.Conv("stem4", x, 80));
    x = b.Stage("stem5", b.Conv("stem5", x, 192, 3, 3, 1, "VALID"));
    x = b.Stage("pool2", b.Pool(x, true, 3, 2, "VALID"));
    auto a = b.Conv("mixed/one", x, 96);
    auto c = b.Conv("mixed/five_reduce", x, 48);
    c = b.Conv("mixed/five", c, 64, 5, 5);
    auto d = b.Conv("mixed/double_reduce", x, 64);
    d = b.Conv("mixed/double1", d, 96, 3, 3);
    d = b.Conv("mixed/double2", d, 96, 3, 3);
    x = b.Stage("mixed", b.Cat({a, c, d, b.Conv("mixed/pool", b.Pool(x), 64)}));
    for (int i = 0; i < 10; ++i)
      x = InceptionResidual(b, "A" + std::to_string(i), x, 'A', .17);
    a = b.Conv("reductionA/three", x, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionA/double_reduce", x, 256);
    c = b.Conv("reductionA/double1", c, 256, 3, 3);
    c = b.Conv("reductionA/double2", c, 384, 3, 3, 2, "VALID");
    x = b.Stage("reductionA", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 20; ++i)
      x = InceptionResidual(b, "B" + std::to_string(i), x, 'B', .1);
    aux_input = x;
    a = b.Conv("reductionB/first_reduce", x, 256);
    a = b.Conv("reductionB/first", a, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionB/second_reduce", x, 256);
    c = b.Conv("reductionB/second", c, 288, 3, 3, 2, "VALID");
    d = b.Conv("reductionB/third_reduce", x, 256);
    d = b.Conv("reductionB/third1", d, 288, 3, 3);
    d = b.Conv("reductionB/third2", d, 320, 3, 3, 2, "VALID");
    x = b.Stage("reductionB", b.Cat({a, c, d, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 9; ++i)
      x = InceptionResidual(b, "C" + std::to_string(i), x, 'C', .2);
    x = InceptionResidual(b, "C9", x, 'C', 1., false);
    x = b.Stage("final", b.Conv("final", x, 1536));
  } else {
    x = b.Stage("stem_mix1", b.Cat({b.Pool(x, true, 3, 2, "VALID"),
                                    b.Conv("stem_mix1/conv", x, 96, 3, 3, 2, "VALID")}));
    auto a = b.Conv("stem_mix2/short_reduce", x, 64);
    a = b.Conv("stem_mix2/short", a, 96, 3, 3, 1, "VALID");
    auto c = b.Conv("stem_mix2/long_reduce", x, 64);
    c = b.Conv("stem_mix2/long_h", c, 64, 1, 7);
    c = b.Conv("stem_mix2/long_v", c, 64, 7, 1);
    c = b.Conv("stem_mix2/long", c, 96, 3, 3, 1, "VALID");
    x = b.Stage("stem_mix2", b.Cat({a, c}));
    x = b.Stage("stem_mix3", b.Cat({b.Conv("stem_mix3/conv", x, 192, 3, 3, 2, "VALID"),
                                    b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 4; ++i)
      x = b.V2Module("A" + std::to_string(i), x, {96, 64, 96, 64, 96, 96, 96});
    a = b.Conv("reductionA/three", x, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionA/double_reduce", x, 192);
    c = b.Conv("reductionA/double1", c, 224, 3, 3);
    c = b.Conv("reductionA/double2", c, 256, 3, 3, 2, "VALID");
    x = b.Stage("reductionA", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 7; ++i)
      x = Inception4B(b, "B" + std::to_string(i), x);
    aux_input = x;
    a = b.Conv("reductionB/three_reduce", x, 192);
    a = b.Conv("reductionB/three", a, 192, 3, 3, 2, "VALID");
    c = b.Conv("reductionB/seven_reduce", x, 256);
    c = b.Conv("reductionB/seven_h", c, 256, 1, 7);
    c = b.Conv("reductionB/seven_v", c, 320, 7, 1);
    c = b.Conv("reductionB/seven_three", c, 320, 3, 3, 2, "VALID");
    x = b.Stage("reductionB", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 3; ++i)
      x = Inception4C(b, "C" + std::to_string(i), x);
  }
  out.logits = InceptionClassifier(p, x.value, cfg.classes, training);
  if (auxiliary)
    out.auxiliary.push_back(InceptionAuxiliary(b, aux_input, layers));
  return out;
}
} // namespace datamunge::nn::models
