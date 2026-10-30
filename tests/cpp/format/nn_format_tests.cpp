// std::format / std::print support for datamunge::nn types.
#include <datamunge/format/nn.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace nn = datamunge::nn;

// Enums.
static_assert(std::formattable<nn::DType, char>);
static_assert(std::formattable<nn::Activation, char>);
static_assert(std::formattable<nn::OptimizerKind, char>);
static_assert(std::formattable<nn::AttentionImpl, char>);
static_assert(std::formattable<nn::RotaryLayout, char>);
static_assert(std::formattable<nn::WavFormat, char>);
static_assert(std::formattable<nn::Padding, char>);
static_assert(std::formattable<nn::ResizeMethod, char>);
static_assert(std::formattable<nn::DepthToSpaceMode, char>);
static_assert(std::formattable<nn::GridSampleMode, char>);
static_assert(std::formattable<nn::GridSamplePadding, char>);
static_assert(std::formattable<nn::Reduction, char>);
static_assert(std::formattable<nn::LinalgBackend, char>);
static_assert(std::formattable<nn::PadMode, char>);
static_assert(std::formattable<nn::RnnKind, char>);
// Tensors, models, containers.
static_assert(std::formattable<nn::Tensor, char>);
static_assert(std::formattable<nn::NnTensor, char>);
static_assert(std::formattable<nn::NnModel, char>);
static_assert(std::formattable<nn::ParameterStore, char>);
static_assert(std::formattable<nn::ComplexTensor, char>);
static_assert(std::formattable<nn::ConfusionMatrix, char>);
// Autograd.
static_assert(std::formattable<nn::Edge, char>);
static_assert(std::formattable<nn::TensorMeta, char>);
static_assert(std::formattable<nn::Node, char>);
static_assert(std::formattable<nn::AccumulateGrad, char>);
static_assert(std::formattable<nn::SavedTensor, char>);
static_assert(std::formattable<nn::GradOptions, char>);
static_assert(std::formattable<nn::Function, char>);
static_assert(std::formattable<nn::CheckpointOptions, char>);
static_assert(std::formattable<nn::ReversibleBlock, char>);
// Options / results.
static_assert(std::formattable<nn::Wav, char>);
static_assert(std::formattable<nn::augment::Rect, char>);
static_assert(std::formattable<nn::augment::MixedBatch, char>);
static_assert(std::formattable<nn::AttentionOptions, char>);
static_assert(std::formattable<nn::Attended, char>);
static_assert(std::formattable<nn::KvCache, char>);
static_assert(std::formattable<nn::MultiHeadOptions, char>);
static_assert(std::formattable<nn::ImageDataset, char>);
static_assert(std::formattable<nn::ImageFile, char>);
static_assert(std::formattable<nn::ImageFolder, char>);
static_assert(std::formattable<nn::ImageSize, char>);
static_assert(std::formattable<nn::GridSampleOptions, char>);
static_assert(std::formattable<nn::DeformableOptions, char>);
static_assert(std::formattable<nn::GradcheckOptions, char>);
static_assert(std::formattable<nn::GradcheckResult, char>);
static_assert(std::formattable<nn::Conv2DOptions, char>);
static_assert(std::formattable<nn::Conv3DOptions, char>);
static_assert(std::formattable<nn::PoolWithArgmax, char>);
static_assert(std::formattable<nn::BatchNormResult, char>);
static_assert(std::formattable<nn::DenseOptions, char>);
static_assert(std::formattable<nn::ConvLayerOptions, char>);
static_assert(std::formattable<nn::BatchNormOptions, char>);
static_assert(std::formattable<nn::LstmState, char>);
static_assert(std::formattable<nn::NmsResult, char>);
static_assert(std::formattable<nn::ClassMetrics, char>);
static_assert(std::formattable<nn::Mt19937, char>);
static_assert(std::formattable<nn::Mt19937_64, char>);
static_assert(std::formattable<nn::Generator, char>);
static_assert(std::formattable<nn::PeakMemoryScope, char>);
static_assert(std::formattable<nn::OptimizerConfig, char>);
static_assert(std::formattable<nn::Optimizer, char>);
static_assert(std::formattable<nn::GradientAccumulator, char>);
static_assert(std::formattable<nn::QrResult, char>);
static_assert(std::formattable<nn::SvdResult, char>);
static_assert(std::formattable<nn::EighResult, char>);
static_assert(std::formattable<nn::SlogdetResult, char>);
static_assert(std::formattable<nn::EigResult, char>);
static_assert(std::formattable<nn::CtcOptions, char>);
static_assert(std::formattable<nn::CtcLossResult, char>);
static_assert(std::formattable<nn::CtcDecoded, char>);
static_assert(std::formattable<nn::CrfDecoded, char>);
static_assert(std::formattable<nn::BatchOrder, char>);
static_assert(std::formattable<nn::EarlyStopping, char>);
static_assert(std::formattable<nn::EarlyStopping::Decision, char>);
static_assert(std::formattable<nn::WeightedMean, char>);

TEST(NnFormat, Enums) {
  EXPECT_EQ(std::format("{}", nn::DType::Int64), "int64");
  EXPECT_EQ(std::format("{}", static_cast<nn::DType>(99)), "DType(99)");
  EXPECT_EQ(std::format("{}", nn::Activation::GeluTanh), "gelu_tanh");
  EXPECT_EQ(std::format("{}", nn::OptimizerKind::AdamW), "AdamW");
  EXPECT_EQ(std::format("{}", nn::AttentionImpl::Fused), "Fused");
  EXPECT_EQ(std::format("{}", nn::RotaryLayout::HalfSplit), "HalfSplit");
  EXPECT_EQ(std::format("{}", nn::WavFormat::Pcm24), "Pcm24");
  EXPECT_EQ(std::format("{}", nn::Padding::Valid), "Valid");
  EXPECT_EQ(std::format("{}", nn::ResizeMethod::Nearest), "Nearest");
  EXPECT_EQ(std::format("{}", nn::DepthToSpaceMode::CRD), "CRD");
  EXPECT_EQ(std::format("{}", nn::GridSampleMode::Bilinear), "Bilinear");
  EXPECT_EQ(std::format("{}", nn::GridSamplePadding::Border), "Border");
  EXPECT_EQ(std::format("{}", nn::Reduction::Sum), "Sum");
  EXPECT_EQ(std::format("{}", nn::LinalgBackend::Lapack), "Lapack");
  EXPECT_EQ(std::format("{}", nn::PadMode::Reflect), "Reflect");
  EXPECT_EQ(std::format("{}", nn::RnnKind::Gru), "Gru");
  EXPECT_EQ(std::format("{}", static_cast<nn::RnnKind>(7)), "RnnKind(7)");
}

TEST(NnFormat, Tensors) {
  const auto t = nn::tensor({{1, 2, 3}, {4, 5, 6.5}});
  EXPECT_EQ(std::format("{}", t), t.str());
  EXPECT_EQ(std::format("{}", t), "float32[2, 3] [[1, 2, 3], [4, 5, 6.5]]");
  EXPECT_EQ(std::format("{}", nn::Tensor()), "Tensor(undefined)");
  EXPECT_EQ(std::format("{}", nn::NnTensor({1, 2, 3, 4}, {2, 2})),
            "NnTensor(shape=[2, 2], dtype=float32, values=[1, 2, 3, 4])");
  const auto c = nn::make_complex(nn::tensor({1, 2}), nn::tensor({0, -1}));
  EXPECT_EQ(std::format("{}", c), "ComplexTensor float32[2]\nreal: [1, 2]\nimag: [0, -1]");
  EXPECT_EQ(std::format("{}", nn::ComplexTensor{}), "ComplexTensor(undefined)");
}

TEST(NnFormat, ModelsAndStores) {
  nn::NnModel model(1);
  model.add_dense(4, "relu");
  model.add_dense(2);
  EXPECT_EQ(std::format("{}", model), "NnModel(layers=2)\n  dense0\n  dense1\n  parameters: 0");

  nn::ParameterStore store(0);
  store.get("dense/W", {3, 2}, nn::init::zeros());
  store.get("dense/b", {2}, nn::init::zeros());
  EXPECT_EQ(std::format("{}", store),
            "ParameterStore(params=2, count=8, dtype=float32)\n  dense/W: float32[3, 2]\n  dense/b: float32[2]");

  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  EXPECT_EQ(std::format("{}", cfg), "OptimizerConfig(Adam, beta1=0.9, beta2=0.999, epsilon=1e-08)");
  EXPECT_EQ(std::format("{}", nn::OptimizerConfig{}), "OptimizerConfig(" + nn::optimizer_name(nn::OptimizerKind::Sgd) + ")");
  const nn::Optimizer opt(store, 0.01, cfg);
  EXPECT_EQ(std::format("{}", opt),
            "Optimizer(Adam, lr=0.01, iterations=0, parameters=2, beta1=0.9, beta2=0.999, epsilon=1e-08)");
  nn::GradientAccumulator acc;
  EXPECT_EQ(std::format("{}", acc), "GradientAccumulator(count=0, total_weight=0)");
}

TEST(NnFormat, Autograd) {
  auto a = nn::ones({2});
  a.requires_grad_();
  const auto b = nn::mul(a, a);
  const nn::Node& node = *b.impl()->grad_fn;
  EXPECT_EQ(std::format("{}", node), "Node(Mul, inputs=[float32[2]], outputs=[float32[2]], next=[AccumulateGrad])");
  EXPECT_EQ(std::format("{}", node.next_edges().at(0)), "Edge(node=AccumulateGrad, input_nr=0)");
  EXPECT_EQ(std::format("{}", nn::Edge{}), "Edge(node=None, input_nr=0)");
  EXPECT_EQ(std::format("{}", nn::TensorMeta{{2, 3}, nn::DType::Float64}), "TensorMeta(float64[2, 3])");
  EXPECT_EQ(std::format("{}", nn::SavedTensor(nn::zeros({1}))), "SavedTensor(float32[1])");
  EXPECT_EQ(std::format("{}", nn::GradOptions{}), "GradOptions(retain_graph=false, create_graph=false, allow_unused=false)");
  EXPECT_EQ(std::format("{}", nn::Function("Square", {}, {})), "Function(Square)");
  EXPECT_EQ(std::format("{}", nn::CheckpointOptions{}), "CheckpointOptions(preserve_rng_state=true, generators=0)");
  EXPECT_EQ(std::format("{}", nn::ReversibleBlock{[](const nn::Tensor& x) { return x; }, {}}),
            "ReversibleBlock(f=set, g=empty)");
}

TEST(NnFormat, OptionStructs) {
  EXPECT_EQ(std::format("{}", nn::augment::Rect{1, 2, 3, 4}), "Rect(top=1, left=2, height=3, width=4)");
  EXPECT_EQ(std::format("{}", nn::augment::MixedBatch{nn::zeros({2, 4, 4, 1}), nn::Tensor(), 0.5}),
            "MixedBatch(images=float32[2, 4, 4, 1], labels=undefined, weight=0.5)");
  EXPECT_EQ(std::format("{}", nn::AttentionOptions{}),
            "AttentionOptions(causal=false, scale=0, dropout=0, need_weights=false, impl=Auto, mask=undefined)");
  EXPECT_TRUE(std::format("{}", nn::MultiHeadOptions{}).starts_with("MultiHeadOptions(causal=false, kv_heads=0, bias=false"));
  EXPECT_EQ(std::format("{}", nn::ImageSize{28, 32}), "ImageSize(28, 32)");
  EXPECT_EQ(std::format("{}", nn::ImageFile{"cat/1.png", 3}), "ImageFile(cat/1.png, label=3)");
  EXPECT_EQ(std::format("{}", nn::ImageFolder{{"cat", "dog"}, {{"cat/1.png", 0}}}), "ImageFolder(classes=[cat, dog], images=1)");
  EXPECT_EQ(std::format("{}", nn::GridSampleOptions{}), "GridSampleOptions(mode=Bilinear, padding=Zeros, align_corners=false)");
  EXPECT_EQ(std::format("{}", nn::DeformableOptions{}), "DeformableOptions(kernel=3x3, stride=1, pad=1x1, dilation=1)");
  EXPECT_EQ(std::format("{}", nn::GradcheckOptions{}), "GradcheckOptions(eps=1e-06, atol=1e-05, rtol=0.001)");
  EXPECT_EQ(std::format("{}", nn::GradcheckResult{false, 0.5, "bad"}),
            "GradcheckResult(ok=false, max_abs_error=0.5, message=\"bad\")");
  EXPECT_EQ(std::format("{}", nn::Conv2DOptions{}), "Conv2DOptions(strides=[1, 1], dilations=[1, 1], padding=Same, groups=1)");
  nn::Conv3DOptions c3;
  c3.padding = nn::Padding::Explicit;
  EXPECT_EQ(std::format("{}", c3),
            "Conv3DOptions(strides=[1, 1, 1], dilations=[1, 1, 1], padding=Explicit, explicit_paddings=[0, 0, 0, 0, 0, 0], groups=1)");
  EXPECT_EQ(std::format("{}", nn::DenseOptions{}), "DenseOptions(activation=identity, bias=true, bias_value=0, alpha=0.2)");
  EXPECT_EQ(std::format("{}", nn::ConvLayerOptions{}),
            "ConvLayerOptions(stride=1, dilation=1, padding=Same, activation=identity, bias=true, alpha=0.2)");
  EXPECT_EQ(std::format("{}", nn::BatchNormOptions{}),
            "BatchNormOptions(momentum=0.99, eps=0.001, scale=true, center=true, update_stats=true)");
  EXPECT_EQ(std::format("{}", nn::CtcOptions{}), "CtcOptions(blank=-1, time_major=true, ignore_longer_outputs_than_inputs=false)");
}

TEST(NnFormat, ResultStructs) {
  EXPECT_EQ(std::format("{}", nn::Wav{nn::zeros({4, 2}), 16000}), "Wav(samples=float32[4, 2], sample_rate=16000)");
  EXPECT_EQ(std::format("{}", nn::Attended{nn::zeros({1, 2}), {}}), "Attended(output=float32[1, 2], weights=undefined)");
  EXPECT_EQ(std::format("{}", nn::KvCache()), "KvCache(length=0, keys=undefined, values=undefined)");
  EXPECT_EQ(std::format("{}", nn::ImageDataset{nn::zeros({2, 2, 2, 1}), nn::zeros({2}, nn::DType::Int64)}),
            "ImageDataset(images=float32[2, 2, 2, 1], labels=int64[2])");
  EXPECT_EQ(std::format("{}", nn::PoolWithArgmax{}), "PoolWithArgmax(output=undefined, argmax=undefined)");
  EXPECT_EQ(std::format("{}", nn::BatchNormResult{}), "BatchNormResult(y=undefined, mean=undefined, variance=undefined)");
  EXPECT_EQ(std::format("{}", nn::lstm_zero_state(1, 3)), "LstmState(h=float32[1, 3], c=float32[1, 3])");
  EXPECT_EQ(std::format("{}", nn::NmsResult{}), "NmsResult(valid=0, indices=undefined, scores=undefined)");
  EXPECT_EQ(std::format("{}", nn::ClassMetrics{0.5, 1, 0.75, 0.25, 4}),
            "ClassMetrics(precision=0.5, recall=1, f1=0.75, iou=0.25, support=4)");
  const auto a = nn::tensor({{2, 0}, {0, 3}});
  EXPECT_EQ(std::format("{}", nn::qr(a)), "QrResult(q=float32[2, 2], r=float32[2, 2])");
  EXPECT_EQ(std::format("{}", nn::svd(a)), "SvdResult(s=float32[2], u=float32[2, 2], v=float32[2, 2])");
  EXPECT_EQ(std::format("{}", nn::eigh(a, false)), "EighResult(e=float32[2], v=undefined)");
  EXPECT_EQ(std::format("{}", nn::slogdet(a)), "SlogdetResult(sign=1, log_abs_det=1.79176)");
  EXPECT_EQ(std::format("{}", nn::EigResult{}), "EigResult(e=undefined, v=undefined)");
  EXPECT_EQ(std::format("{}", nn::CtcLossResult{}), "CtcLossResult(loss=undefined, gradient=undefined)");
  EXPECT_EQ(std::format("{}", nn::CtcDecoded{{{1, 2}, {3}}, nn::Tensor(), {}, {}}),
            "CtcDecoded(sequences=[[1, 2], [3]], dense=undefined)");
  EXPECT_EQ(std::format("{}", nn::CrfDecoded{}), "CrfDecoded(tags=undefined, scores=undefined)");
}

TEST(NnFormat, MetricsAndTraining) {
  nn::ConfusionMatrix cm(2);
  cm.add(0, 0);
  cm.add(1, 0);
  cm.add(1, 1);
  cm.add(1, 1);
  EXPECT_EQ(std::format("{}", cm),
            "ConfusionMatrix(classes=2, total=4, accuracy=0.75)\n"
            "     [,0] [,1]\n"
            "[0,]    1    0\n"
            "[1,]    1    2");
  EXPECT_EQ(std::format("{}", nn::ConfusionMatrix(1)), "ConfusionMatrix(classes=1, total=0)\n     [,0]\n[0,]    0");

  nn::BatchOrder order(10, 4, 1);
  EXPECT_EQ(std::format("{}", order), "BatchOrder(batches_per_epoch=3, epoch=-1, cursor=0)");
  nn::EarlyStopping stop(2);
  EXPECT_EQ(std::format("{}", stop), "EarlyStopping(best_epoch=-1, misses=0)");
  const auto decision = stop.observe(0.5, 0);
  EXPECT_EQ(std::format("{}", stop), "EarlyStopping(best=0.5, best_epoch=0, misses=0)");
  EXPECT_EQ(std::format("{}", decision), "Decision(improved=true, stop=false)");
  nn::WeightedMean wm;
  EXPECT_EQ(std::format("{}", wm), "WeightedMean(count=0)");
  wm.add(2.0, 3);
  EXPECT_EQ(std::format("{}", wm), "WeightedMean(value=2, count=3)");
  EXPECT_TRUE(std::format("{}", nn::PeakMemoryScope()).starts_with("PeakMemoryScope(baseline="));
}

TEST(NnFormat, GeneratorsAreNotAdvanced) {
  nn::Generator g(7), h(7);
  EXPECT_EQ(std::format("{}", g), "Generator");
  EXPECT_EQ(g.next_u64(), h.next_u64());
  EXPECT_EQ(std::format("{}", nn::Mt19937(1)), "Mt19937");
  EXPECT_EQ(std::format("{}", nn::Mt19937_64(1)), "Mt19937_64");
}
