#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <regex>
#include <stdexcept>

namespace n = datamunge::nn;
namespace {
std::size_t count_kind(const n::ModelDiagram& diagram, const std::string& prefix) {
  std::size_t result = 0;
  for (const auto& node : diagram.nodes())
    result += node.kind.starts_with(prefix);
  return result;
}
void check_order(const n::ModelDiagram& diagram) {
  for (std::size_t i = 0; i < diagram.nodes().size(); ++i)
    for (const auto j : diagram.nodes()[i].inputs)
      EXPECT_LT(j, i);
}
} // namespace

TEST(NnModelPlot, LinearAndDeepShapesCountsAndOrdinaryForward) {
  n::ParameterStore p(17);
  n::Module         linear([&](const auto& x) { return n::dense(p, "linear", x, 3); }, "Linear");
  const auto        input    = n::ones({2, 4});
  const auto        baseline = linear(input);
  const auto        before   = p.flatten();
  const auto        plot     = linear.plot(input);
  ASSERT_EQ(plot.nodes().size(), 3u);
  EXPECT_EQ(plot.nodes()[0].shape, (n::Shape{2, 4}));
  EXPECT_EQ(plot.nodes()[1].shape, (n::Shape{2, 3}));
  EXPECT_EQ(plot.nodes()[1].label, "linear");
  EXPECT_EQ(plot.parameter_count(), 15);
  EXPECT_EQ(p.flatten(), before);
  EXPECT_FALSE(input.requires_grad());
  EXPECT_FALSE(input.grad().defined());
  EXPECT_EQ(n::max(n::abs(linear(input) - baseline)).item(), 0.);
  EXPECT_NE(plot.to_svg().find("<title id=\"model-title\">Linear</title>"), std::string::npos);
  n::ParameterStore deep(19);
  n::Module         model(
      [&](const auto& x) { return n::dense(deep, "head", n::mlp(deep, "hidden", x, {5, 4}), 2); });
  const auto diagram = model.plot(n::zeros({1, 3}));
  EXPECT_EQ(diagram.nodes().size(), 5u);
  EXPECT_EQ(diagram.parameter_count(), 3 * 5 + 5 + 5 * 4 + 4 + 4 * 2 + 2);
  check_order(diagram);
}

TEST(NnModelPlot, SharedLayersResidualEdgesAndUnusedWeights) {
  n::ParameterStore p(23);
  p.get("unused", {99}, n::init::zeros());
  const auto diagram = n::plot_model(
      [&](const auto& x) { return n::dense(p, "shared", x, 3) + n::dense(p, "shared", x * 2., 3); },
      n::ones({1, 2}));
  EXPECT_EQ(count_kind(diagram, "Dense"), 2u);
  EXPECT_EQ(diagram.parameter_count(), 9);
  std::size_t add = 0;
  for (const auto& node : diagram.nodes())
    if (node.label == "Add") {
      ++add;
      EXPECT_EQ(node.inputs.size(), 2u);
      EXPECT_NE(node.inputs[0], node.inputs[1]);
    }
  EXPECT_EQ(add, 1u);
  check_order(diagram);
}

TEST(NnModelPlot, CnnDepthwiseTransposeAndBatchNormState) {
  n::ParameterStore p(31), state(31);
  const auto        forward = [&](const n::Tensor& x) {
    auto h = n::conv2d(p, "conv", x, 4, 3, {.activation = n::Activation::Relu});
    h      = n::batch_norm(p, state, "norm", h, false);
    h      = n::depthwise_conv2d(p, "dw", h, 2, 3);
    h      = n::conv2d_transpose(p, "up", h, 3, 3, {.stride = 2});
    return n::global_avg_pool2d(h);
  };
  forward(n::ones({1, 4, 4, 2}));
  const auto before  = state.flatten();
  const auto diagram = n::plot_model(forward, n::ones({1, 4, 4, 2}));
  EXPECT_EQ(diagram.parameter_count(), p.count());
  EXPECT_EQ(count_kind(diagram, "Conv2D k"), 1u);
  EXPECT_EQ(count_kind(diagram, "DepthwiseConv2D"), 1u);
  EXPECT_EQ(count_kind(diagram, "Conv2DTranspose"), 1u);
  EXPECT_EQ(count_kind(diagram, "BatchNorm"), 1u);
  EXPECT_EQ(diagram.nodes().back().shape, (n::Shape{1, 3}));
  EXPECT_EQ(state.flatten(), before);
  check_order(diagram);
}

TEST(NnModelPlot, CompositeStagesAndExpansionHaveSameUniqueParameters) {
  n::ParameterStore p(41);
  const auto        forward = [&](const n::Tensor& x) {
    return n::model_stage("Hidden block", x, n::mlp(p, "block", x, {4, 4}));
  };
  const auto grouped  = n::plot_model(forward, n::ones({1, 2}));
  const auto expanded = n::plot_model(forward, n::ones({1, 2}), {.expand_layers = true});
  EXPECT_EQ(grouped.nodes().size(), 3u);
  EXPECT_GT(expanded.nodes().size(), grouped.nodes().size());
  EXPECT_EQ(grouped.parameter_count(), 32);
  EXPECT_EQ(expanded.parameter_count(), grouped.parameter_count());
  EXPECT_EQ(n::model_stage("identity", n::ones({1}), n::zeros({1})).item(), 0.);
}

TEST(NnModelPlot, GradientsAndGradientModeArePreserved) {
  auto x = n::ones({1, 3}, n::DType::Float64).requires_grad_();
  n::sum(x * x).backward();
  const auto     previous = x.grad().clone();
  n::NoGradGuard guard;
  const auto     plot = n::plot_model([](const auto& v) { return n::square(v); }, x);
  EXPECT_FALSE(n::is_grad_enabled());
  EXPECT_TRUE(x.requires_grad());
  EXPECT_EQ(n::max(n::abs(x.grad() - previous)).item(), 0.);
  EXPECT_THROW(
      n::plot_model(
          [](const n::Tensor&) -> n::Tensor {
            throw std::runtime_error("deliberate callback failure");
          },
          x),
      std::runtime_error);
  EXPECT_FALSE(n::is_grad_enabled());
  EXPECT_EQ(n::plot_model([](const auto& v) { return v; }, x).nodes().size(), 2u);
}

TEST(NnModelPlot, NestedAndConcurrentTracesAreIsolated) {
  const auto outer = n::plot_model(
      [](const n::Tensor& x) {
        const auto inner = n::plot_model([](const auto& v) { return n::tanh(v); }, x);
        EXPECT_EQ(inner.nodes().back().shape, x.shape());
        return n::relu(x);
      },
      n::ones({1, 2}));
  EXPECT_EQ(outer.nodes().size(), 3u);
  const auto run = [](int features) {
    n::ParameterStore p(17);
    return n::plot_model(
        [&](const auto& x) { return n::dense(p, "head", x, 2); }, n::ones({1, features}));
  };
  auto a = std::async(std::launch::async, run, 3), b = std::async(std::launch::async, run, 7);
  EXPECT_EQ(a.get().parameter_count(), 8);
  EXPECT_EQ(b.get().parameter_count(), 16);
}

TEST(NnModelPlot, SplitOutputSlotMetadataIsExact) {
  const auto graph = n::plot_model(
      [](const auto& x) { return n::split(x, std::vector<std::int64_t>{1, 3}, 1)[1]; },
      n::ones({2, 4}));
  EXPECT_EQ(graph.nodes().back().shape, (n::Shape{2, 3}));
  EXPECT_EQ(graph.nodes()[graph.nodes().size() - 2].shape, (n::Shape{2, 3}));
}

TEST(NnModelPlot, SvgEscapingDeterminismAndFileErrors) {
  const auto graph = n::plot_model(
      [](const auto& x) { return x + 1.; },
      n::ones({1, 2}),
      {.title = "A < B & \"quoted\" 'title'"});
  const auto svg = graph.to_svg();
  EXPECT_NE(svg.find("A &lt; B &amp; &quot;quoted&quot; &apos;title&apos;"), std::string::npos);
  EXPECT_EQ(svg, graph.to_svg());
  const auto root = std::filesystem::temp_directory_path() / "datamunge-nn-model-plot-tests";
  std::filesystem::create_directories(root);
  const auto file = root / "model.svg";
  graph.save_svg(file.string());
  std::ifstream     input(file);
  const std::string actual((std::istreambuf_iterator<char>(input)), {});
  EXPECT_EQ(actual, svg);
  input.close();
  EXPECT_THROW(graph.save_svg((root / "missing" / "model.svg").string()), std::runtime_error);
  std::filesystem::remove(file);
  std::filesystem::remove(root);
}

TEST(NnModelPlot, RejectsInvalidInputsAndUnrecordedOutputs) {
  const n::ModelForward identity = [](const auto& x) { return x; };
  EXPECT_THROW(n::Module({}), std::invalid_argument);
  EXPECT_THROW(n::plot_model({}, n::ones({1})), std::invalid_argument);
  EXPECT_THROW(n::plot_model(identity, n::Tensor{}), std::invalid_argument);
  EXPECT_THROW(n::plot_model(identity, n::zeros({1}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(n::plot_model(identity, n::ones({1}), {.width = 10}), std::invalid_argument);
  EXPECT_THROW(
      n::plot_model([](const auto& x) { return x.detach(); }, n::ones({1})), std::invalid_argument);
  EXPECT_THROW(
      n::plot_model([](const auto&) { return n::Tensor{}; }, n::ones({1})), std::invalid_argument);
}

TEST(NnModelPlot, AllStylesFlowLeftToRightAndRotateLabelsCounterclockwise) {
  n::ParameterStore p(67);
  const auto        forward = [&](const n::Tensor& x) {
    auto h = x;
    for (int i = 0; i < 12; ++i)
      h = n::dense(p, "layer" + std::to_string(i), h, 3, n::Activation::Relu);
    return h + n::dense(p, "skip", x, 3);
  };
  for (auto style : {n::ModelPlotStyle::Flow, n::ModelPlotStyle::Graph, n::ModelPlotStyle::LeNet}) {
    const auto graph = n::plot_model(forward, n::ones({1, 2}), {.style = style});
    const auto svg   = graph.to_svg();
    std::map<std::size_t, std::pair<double, double>> boxes;
    const std::regex                                 glyph(
        R"svg(id="model-glyph-(\d+)" data-left="([^"]+)" data-right="([^"]+)")svg");
    for (auto it = std::sregex_iterator(svg.begin(), svg.end(), glyph);
         it != std::sregex_iterator();
         ++it)
      boxes[std::stoull((*it)[1])] = {std::stod((*it)[2]), std::stod((*it)[3])};
    ASSERT_EQ(boxes.size(), graph.nodes().size());
    for (std::size_t i = 0; i < graph.nodes().size(); ++i)
      for (auto parent : graph.nodes()[i].inputs)
        EXPECT_LT(boxes.at(parent).second, boxes.at(i).first);
    const std::regex rotation(R"(rotate\(-90\))");
    EXPECT_EQ(
        std::distance(
            std::sregex_iterator(svg.begin(), svg.end(), rotation), std::sregex_iterator()),
        graph.nodes().size());
    EXPECT_NE(svg.find("data-model-layout=\"left-to-right\""), std::string::npos);
  }
}

TEST(NnModelPlot, StyleChangesGlyphsAndPreservesShapesAndParameters) {
  n::ParameterStore p(73);
  const auto        forward = [&](const n::Tensor& x) { return n::conv2d(p, "conv", x, 16, 3); };
  std::vector<n::ModelDiagram> diagrams;
  for (auto style : {n::ModelPlotStyle::Flow, n::ModelPlotStyle::Graph, n::ModelPlotStyle::LeNet})
    diagrams.push_back(n::plot_model(forward, n::ones({1, 8, 8, 3}), {.style = style}));
  EXPECT_NE(diagrams[0].to_svg().find("volume-front"), std::string::npos);
  EXPECT_EQ(diagrams[1].to_svg().find("volume-front"), std::string::npos);
  EXPECT_NE(diagrams[2].to_svg().find("feature-plane"), std::string::npos);
  for (const auto& g : diagrams) {
    EXPECT_EQ(g.parameter_count(), p.count());
    ASSERT_EQ(g.nodes().size(), diagrams[0].nodes().size());
    for (std::size_t i = 0; i < g.nodes().size(); ++i) {
      EXPECT_EQ(g.nodes()[i].shape, diagrams[0].nodes()[i].shape);
      EXPECT_EQ(g.nodes()[i].inputs, diagrams[0].nodes()[i].inputs);
    }
  }
  n::ModelPlotOptions options;
  options.colors.convolution = "#123456";
  EXPECT_NE(
      n::plot_model(forward, n::ones({1, 8, 8, 3}), options).to_svg().find("#123456"),
      std::string::npos);
}

TEST(NnModelPlot, BlocksLayersAndOperationsKeepIndependentBoundaries) {
  n::ParameterStore p(79);
  const auto        forward = [&](const n::Tensor& x) {
    return n::model_stage("Two-layer block", x, n::mlp(p, "hidden", x, {4, 4}));
  };
  const auto blocks = n::plot_model(forward, n::ones({1, 2}));
  const auto layers =
      n::plot_model(forward, n::ones({1, 2}), {.detail = n::ModelPlotDetail::Layers});
  const auto ops =
      n::plot_model(forward, n::ones({1, 2}), {.detail = n::ModelPlotDetail::Operations});
  EXPECT_EQ(blocks.nodes().size(), 3u);
  EXPECT_EQ(layers.nodes().size(), 4u);
  EXPECT_EQ(layers.nodes()[1].label, "hidden/dense0");
  EXPECT_EQ(layers.nodes()[2].label, "hidden/dense1");
  EXPECT_GT(ops.nodes().size(), layers.nodes().size());
  EXPECT_EQ(blocks.parameter_count(), layers.parameter_count());
  EXPECT_EQ(layers.parameter_count(), ops.parameter_count());
}

TEST(NnModelPlot, GraphSamplesOnlyRealDenseUnitsAndCanHideNeurons) {
  n::ParameterStore   p(83);
  const auto          forward = [&](const n::Tensor& x) { return n::dense(p, "dense", x, 12); };
  n::ModelPlotOptions options;
  options.style        = n::ModelPlotStyle::Graph;
  options.max_neurons  = 4;
  const auto       svg = n::plot_model(forward, n::ones({1, 4}), options).to_svg();
  const std::regex unit(R"svg(data-unit="(\d+)")svg");
  EXPECT_EQ(
      std::distance(std::sregex_iterator(svg.begin(), svg.end(), unit), std::sregex_iterator()), 8);
  EXPECT_NE(svg.find("data-unit=\"11\""), std::string::npos);
  EXPECT_NE(svg.find("neuron-connection"), std::string::npos);
  options.show_neurons = false;
  const auto hidden    = n::plot_model(forward, n::ones({1, 4}), options).to_svg();
  EXPECT_EQ(hidden.find("data-unit="), std::string::npos);
  EXPECT_EQ(hidden.find("neuron-connection"), std::string::npos);
  const auto conv = n::plot_model(
                        [&](const auto& x) { return n::conv2d(p, "conv", x, 3, 1); },
                        n::ones({1, 4, 4, 2}),
                        {.style = n::ModelPlotStyle::Graph})
                        .to_svg();
  EXPECT_EQ(conv.find("neuron-connection"), std::string::npos);
}

TEST(NnModelPlot, InvalidStylesPalettesAndSamplingAreRejectedBeforeForward) {
  int        calls   = 0;
  const auto forward = [&](const n::Tensor& x) {
    ++calls;
    return x;
  };
  n::ModelPlotOptions options;
  options.style = static_cast<n::ModelPlotStyle>(999);
  EXPECT_THROW(n::plot_model(forward, n::ones({1}), options), std::invalid_argument);
  options        = {};
  options.detail = static_cast<n::ModelPlotDetail>(999);
  EXPECT_THROW(n::plot_model(forward, n::ones({1}), options), std::invalid_argument);
  options              = {};
  options.colors.dense = "red\"/><script>";
  EXPECT_THROW(n::plot_model(forward, n::ones({1}), options), std::invalid_argument);
  options             = {};
  options.max_neurons = 1000;
  EXPECT_THROW(n::plot_model(forward, n::ones({1}), options), std::invalid_argument);
  EXPECT_EQ(calls, 0);
}

namespace {
// Read exposed drawing geometry independently of renderer helpers.
std::map<std::size_t, std::array<double, 3>> face_dimensions(const std::string& svg) {
  const std::regex pattern(
      R"svg(id="model-glyph-(\d+)"[^>]*data-face-width="([^"]+)" data-face-height="([^"]+)" data-depth="([^"]+)")svg");
  std::map<std::size_t, std::array<double, 3>> result;
  for (auto i = std::sregex_iterator(svg.begin(), svg.end(), pattern); i != std::sregex_iterator();
       ++i)
    result[std::stoull((*i)[1])] = {std::stod((*i)[2]), std::stod((*i)[3]), std::stod((*i)[4])};
  return result;
}
} // namespace

TEST(NnModelPlot, LinearGeometryPreservesSpatialChannelAndVolumeRatiosAcrossStyles) {
  n::ParameterStore p(91);
  const auto        forward = [&](const auto& x) {
    return n::avg_pool2d(n::conv2d(p, "maps", x, 4, 1), 2, 2);
  };
  for (auto style : {n::ModelPlotStyle::Flow, n::ModelPlotStyle::LeNet, n::ModelPlotStyle::Graph}) {
    n::ModelPlotOptions options;
    options.style                 = style;
    options.sizing.minimum_extent = .01;
    const auto diagram            = n::plot_model(forward, n::ones({1, 8, 16, 2}), options);
    const auto dims               = face_dimensions(diagram.to_svg());
    ASSERT_EQ(dims.size(), diagram.nodes().size());
    std::size_t conv = 0, pool = 0;
    for (std::size_t i = 0; i < diagram.nodes().size(); ++i) {
      if (diagram.nodes()[i].kind.starts_with("Conv2D"))
        conv = i;
      if (diagram.nodes()[i].shape == n::Shape{1, 4, 8, 4})
        pool = i;
    }
    ASSERT_GT(conv, 0u);
    ASSERT_GT(pool, conv);
    const auto a = dims.at(0), b = dims.at(conv), c = dims.at(pool);
    EXPECT_NEAR(a[0], 144., 1e-6);
    EXPECT_NEAR(a[1], 72., 1e-6);
    EXPECT_NEAR(b[2] / a[2], 2., 1e-6); // Twice the channel count.
    EXPECT_NEAR(c[0] / b[0], .5, 1e-6);
    EXPECT_NEAR(c[1] / b[1], .5, 1e-6);
    EXPECT_NEAR(c[0] * c[1] * c[2] / (b[0] * b[1] * b[2]), .25, 1e-6);
    // Batch size never changes glyph dimensions.
    EXPECT_EQ(
        dims, face_dimensions(n::plot_model(forward, n::ones({3, 8, 16, 2}), options).to_svg()));
  }
}

TEST(NnModelPlot, ExplicitCompressionFloorAndVectorCountsControlGeometry) {
  n::ModelPlotOptions options;
  options.sizing.minimum_extent = .01;
  n::ParameterStore p(97);
  const auto        forward = [&](const auto& x) { return n::dense(p, "wide", x, 16); };
  const auto linear = face_dimensions(n::plot_model(forward, n::ones({1, 4}), options).to_svg());
  EXPECT_NEAR(linear.at(1)[1] / linear.at(0)[1], 4., 1e-6);
  options.sizing.channel_scale = n::ModelPlotScale::SquareRoot;
  const auto root = face_dimensions(n::plot_model(forward, n::ones({1, 4}), options).to_svg());
  EXPECT_NEAR(root.at(1)[1] / root.at(0)[1], 2., 1e-6);
  options.sizing.channel_scale = n::ModelPlotScale::Logarithmic;
  const auto log = face_dimensions(n::plot_model(forward, n::ones({1, 4}), options).to_svg());
  EXPECT_NEAR(log.at(0)[1], 144. * std::log1p(4.) / std::log1p(16.), 1e-3);
  options.sizing.minimum_extent = 16.;
  options.sizing.spatial_size   = 16.;
  options.sizing.channel_scale  = n::ModelPlotScale::Linear;
  EXPECT_DOUBLE_EQ(
      face_dimensions(n::plot_model(forward, n::ones({1, 4}), options).to_svg()).at(0)[1], 16.);
}

TEST(NnModelPlot, RankThreeAndFiveSpatialAxesHaveDocumentedGeometry) {
  const auto          identity = [](const auto& x) { return x; };
  n::ModelPlotOptions options;
  options.sizing.minimum_extent = .01;
  const auto sequence =
      face_dimensions(n::plot_model(identity, n::ones({2, 12, 3}), options).to_svg()).at(0);
  EXPECT_NEAR(sequence[0], 144., 1e-6);
  EXPECT_NEAR(sequence[1], 12., 1e-6);
  const auto volume =
      face_dimensions(n::plot_model(identity, n::ones({2, 2, 3, 12, 3}), options).to_svg()).at(0);
  EXPECT_NEAR(volume[0], 144., 1e-6);
  EXPECT_NEAR(volume[1], 72., 1e-6); // Unfolded D*H = 6, width = 12.
}

TEST(NnModelPlot, LabelContentOpacityPaletteSpacingAndPlaneLimitAreEffective) {
  const auto          forward = [](const auto& x) { return n::relu(x); };
  n::ModelPlotOptions options;
  options.opacity           = .25;
  options.edge_opacity      = .4;
  options.funnel_opacity    = .05;
  options.line_width        = 2.;
  options.colors.background = "#112233";
  options.colors.text       = "#DDEEFF";
  options.colors.edge       = "#223344";
  options.colors.outline    = "#AABBCC";
  options.labels.size       = 18.;
  options.labels.type       = n::ModelPlotLabel::Shape;
  auto svg                  = n::plot_model(forward, n::ones({1, 8, 8, 16}), options).to_svg();
  EXPECT_NE(svg.find("fill-opacity=\"0.25\""), std::string::npos);
  EXPECT_NE(svg.find("stroke-opacity=\"0.4\""), std::string::npos);
  EXPECT_NE(svg.find("stroke-width=\"2\""), std::string::npos);
  for (auto color : {"#112233", "#DDEEFF", "#223344", "#AABBCC"})
    EXPECT_NE(svg.find(color), std::string::npos);
  EXPECT_NE(svg.find("font-size=\"18\""), std::string::npos);
  EXPECT_EQ(svg.find(">ReLU</text>"), std::string::npos);
  EXPECT_NE(svg.find("<title>"), std::string::npos);
  const auto original   = face_dimensions(svg);
  options.spacing.layer = 240;
  options.spacing.lane  = 160;
  options.labels.type   = n::ModelPlotLabel::None;
  svg                   = n::plot_model(forward, n::ones({1, 8, 8, 16}), options).to_svg();
  EXPECT_EQ(svg.find("data-node-label="), std::string::npos);
  EXPECT_EQ(face_dimensions(svg), original);
  options.style             = n::ModelPlotStyle::LeNet;
  options.sizing.max_planes = 3;
  svg                       = n::plot_model(forward, n::ones({1, 8, 8, 16}), options).to_svg();
  const std::regex plane(R"(class="feature-plane")");
  EXPECT_EQ(
      std::distance(std::sregex_iterator(svg.begin(), svg.end(), plane), std::sregex_iterator()),
      9);
}

TEST(NnModelPlot, InvalidPresentationSettingsFailBeforeForward) {
  int        calls   = 0;
  const auto forward = [&](const auto& x) {
    ++calls;
    return x;
  };
  const auto invalid = [&](const auto& change) {
    n::ModelPlotOptions o;
    change(o);
    EXPECT_THROW(n::plot_model(forward, n::ones({1}), o), std::invalid_argument);
  };
  invalid([](auto& o) { o.opacity = std::numeric_limits<double>::quiet_NaN(); });
  invalid([](auto& o) { o.edge_opacity = 1.01; });
  invalid([](auto& o) { o.funnel_opacity = -.1; });
  invalid([](auto& o) { o.line_width = 0.; });
  invalid([](auto& o) { o.spacing.layer = std::numeric_limits<double>::infinity(); });
  invalid([](auto& o) { o.spacing.lane = -1; });
  invalid([](auto& o) { o.labels.size = 0; });
  invalid([](auto& o) { o.labels.wrap = 0; });
  invalid([](auto& o) { o.labels.type = static_cast<n::ModelPlotLabel>(999); });
  invalid([](auto& o) { o.sizing.minimum_extent = 0; });
  invalid([](auto& o) { o.sizing.max_planes = 33; });
  invalid([](auto& o) { o.sizing.spatial_scale = static_cast<n::ModelPlotScale>(999); });
  invalid([](auto& o) { o.sizing.channel_scale = static_cast<n::ModelPlotScale>(999); });
  invalid([](auto& o) { o.colors.background = "transparent"; });
  EXPECT_EQ(calls, 0);
}

TEST(NnModelPlot, AutomaticWidthAndPerNodeLabelPlacementAvoidFixedPadding) {
  const auto graph = n::plot_model(
      [](const auto& x) { return n::relu(x); },
      n::ones({1, 2}),
      {.legend = false, .labels = {.type = n::ModelPlotLabel::Name}});
  const auto  svg = graph.to_svg();
  std::smatch match;
  ASSERT_TRUE(
      std::regex_search(
          svg, match, std::regex(R"svg(<svg[^>]*width="([^"]+)" height="([^"]+)")svg")));
  EXPECT_LT(std::stod(match[1]), 600.);
  EXPECT_LT(std::stod(match[2]), 400.);
  EXPECT_NE(svg.find("font-family=\"monospace\""), std::string::npos);
}

TEST(NnModelPlot, CompositeMobileAndRecurrentBlocksKeepAllReachableParameters) {
  namespace m = n::models;
  {
    n::ParameterStore           p(17), state(18);
    std::vector<m::MobileLayer> metadata;
    m::MobileContext            context{p, state, metadata, false};
    m::BlockArgs                block;
    auto                        forward = [&](const auto& x) {
      return m::MBConv(context, "unit", x, 8, block, 1, true).output;
    };
    const auto grouped = n::plot_model(forward, n::ones({1, 4, 4, 8}));
    const auto expanded =
        n::plot_model(forward, n::ones({1, 4, 4, 8}), {.detail = n::ModelPlotDetail::Operations});
    EXPECT_EQ(grouped.parameter_count(), p.count());
    EXPECT_EQ(expanded.parameter_count(), p.count());
    EXPECT_LT(grouped.nodes().size(), expanded.nodes().size());
  }
  {
    n::ParameterStore p(17);
    auto              forward = [&](const auto& x) { return m::CrnnBiLstm(p, "rnn", x, 3, 5); };
    const auto        grouped = n::plot_model(forward, n::ones({1, 4, 8}));
    const auto        expanded =
        n::plot_model(forward, n::ones({1, 4, 8}), {.detail = n::ModelPlotDetail::Operations});
    EXPECT_EQ(grouped.parameter_count(), p.count());
    EXPECT_EQ(expanded.parameter_count(), p.count());
    EXPECT_EQ(grouped.nodes().back().shape, (n::Shape{1, 4, 5}));
    EXPECT_LT(grouped.nodes().size(), expanded.nodes().size());
  }
}

TEST(NnModelPlot, RawParameterLeavesPackBesideTheirConsumers) {
  std::vector<n::Tensor> weights;
  for (int i = 0; i < 16; ++i)
    weights.push_back(n::ones({1, 4}).requires_grad_());
  const auto diagram = n::plot_model(
      [&](n::Tensor x) {
        for (const auto& w : weights)
          x = n::relu(x + n::reshape(n::transpose(n::reshape(w, {2, 2}), {1, 0}), {1, 4}));
        return x;
      },
      n::ones({1, 4}),
      {.legend = false});
  const auto  svg = diagram.to_svg();
  std::smatch match;
  ASSERT_TRUE(
      std::regex_search(
          svg, match, std::regex(R"svg(<svg[^>]*width="([^"]+)" height="([^"]+)")svg")));
  EXPECT_LT(std::stod(match[2]), 1000.);
  EXPECT_EQ(diagram.parameter_count(), 64);
}

TEST(NnModelPlot, MultiInputStagesRetainBranchesSharedWeightsAndGradientBehavior) {
  n::Tensor weight  = n::ones({1, 4}).requires_grad_();
  auto      forward = [&](const n::Tensor& x) {
    const auto a = x * weight, b = x + weight;
    return n::model_stage("merge", std::vector<n::Tensor>{a, b, a}, n::relu(a + b));
  };
  const auto grouped = n::plot_model(forward, n::ones({1, 4}));
  const auto expanded =
      n::plot_model(forward, n::ones({1, 4}), {.detail = n::ModelPlotDetail::Operations});
  EXPECT_EQ(grouped.nodes()[grouped.nodes().size() - 2].inputs.size(), 2);
  EXPECT_EQ(grouped.parameter_count(), 4);
  EXPECT_EQ(expanded.parameter_count(), 4);
  EXPECT_LT(grouped.nodes().size(), expanded.nodes().size());
  auto x = n::ones({1, 4}).requires_grad_();
  n::sum(forward(x)).backward();
  EXPECT_EQ(x.grad().to_vector<float>(), (std::vector<float>{2, 2, 2, 2}));
}

TEST(NnModelPlot, PwcCostVolumeBlocksRetainBothFeatureBranches) {
  n::ParameterStore p(17);
  auto              forward = [&](const n::Tensor& x) {
    return n::models::PWCNet(p, x, x * .8, 64, 64, true).flow;
  };
  const auto grouped  = n::plot_model(forward, n::ones({1, 64, 64, 3}) * .5);
  const auto expanded = n::plot_model(
      forward, n::ones({1, 64, 64, 3}) * .5, {.detail = n::ModelPlotDetail::Operations});
  EXPECT_LT(grouped.nodes().size(), expanded.nodes().size() / 2);
  EXPECT_EQ(grouped.parameter_count(), expanded.parameter_count());
  EXPECT_EQ(grouped.parameter_count(), p.count());
  EXPECT_EQ(grouped.nodes().back().shape, (n::Shape{1, 64, 64, 2}));
  int costs = 0;
  for (const auto& node : grouped.nodes())
    if (node.label.find("cost volume") != std::string::npos) {
      ++costs;
      EXPECT_EQ(node.inputs.size(), 2);
    }
  EXPECT_EQ(costs, 5);
}
