
#include <cfenv>
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnRegNetModels, ReleasedDesignStagesAndRoundingModeRestoration) {
  const auto m = n::models::RegNetModel("y400mf");
  const auto d = n::models::RegNetDesign(m);
  EXPECT_EQ(d.widths, std::vector<int>({48, 104, 208, 440}));
  EXPECT_EQ(d.depths, std::vector<int>({1, 3, 6, 6}));
  EXPECT_EQ(d.groups, std::vector<int>({8, 8, 8, 8}));
  const auto x = n::models::RegNetDesign(n::models::RegNetModel("x400mf"));
  EXPECT_EQ(x.widths, std::vector<int>({32, 64, 160, 400}));
  EXPECT_EQ(x.depths, std::vector<int>({1, 2, 7, 12}));
  const int previous = std::fegetround();
  for (const auto *name : {"x400mf", "x800mf", "x1.6gf", "x3.2gf", "x8gf", "x16gf", "x32gf",
                           "y400mf", "y800mf", "y1.6gf", "y3.2gf", "y8gf", "y16gf", "y32gf"}) {
    std::fesetround(FE_TONEAREST);
    const auto params = n::models::RegNetModel(name), reference = n::models::RegNetModel(name);
    const auto expected = n::models::RegNetDesign(reference);
    for (int mode : {FE_DOWNWARD, FE_UPWARD}) {
      std::fesetround(mode);
      const auto other = n::models::RegNetDesign(params);
      EXPECT_EQ(std::fegetround(), mode);
      EXPECT_EQ(other.quantized, expected.quantized);
      EXPECT_EQ(other.widths, expected.widths);
      EXPECT_EQ(other.depths, expected.depths);
    }
  }
  std::fesetround(FE_DOWNWARD);
  auto overflow = m;
  overflow.depth = 1;
  overflow.w0 = std::numeric_limits<int>::max() - 7;
  EXPECT_THROW(n::models::RegNetDesign(overflow), std::invalid_argument);
  EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
  std::fesetround(previous);
}
TEST(NnRegNetModels, InputBasedSqueezeAndPostAddReluShortcutDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  {
    n::NoGradGuard guard;
    n::models::RegNetBlock(c, "width", n::ones({1, 8, 8, 16}, type), 24, 2, 8, .25);
  }
  EXPECT_EQ(p.at("width/se/reduce/W").shape(), n::Shape({1, 1, 24, 4}));
  const auto x =
      n::linspace(-1, 1, 256, type).reshape({1, 4, 4, 16}).detach().requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::RegNetBlock(c, "b", x, 16, 1, 8, .25);
  }
  p.set("b/c/bn/gamma", n::zeros({16}, type));
  p.set("b/c/bn/beta", n::zeros({16}, type));
  const auto y = n::models::RegNetBlock(c, "b", x, 16, 1, 8, .25);
  EXPECT_EQ(n::max(n::abs(y - n::relu(x))).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y), {x})[0] - n::greater(x, 0).to(type))).item(), 0);
}
TEST(NnRegNetModels, FullXAndYCountsShapesAndMeasuredMacs) {
  n::set_num_threads(2);
  n::NoGradGuard guard;
  for (auto [name, count, moving, macs] :
       std::vector<std::tuple<std::string, int64_t, int64_t, int64_t>>{
           {"x400mf", 5495976, 37856, 413812608},
           {"y400mf", 4344144, 27152, 401842848},
           {"x800mf", 7259656, 37312, 799699712}}) {
    n::ParameterStore p(17), state(17);
    n::models::RegNetConfig cfg;
    cfg.model = name;
    const auto y = n::models::RegNet(p, state, n::zeros({1, 224, 224, 3}), cfg);
    EXPECT_EQ(p.count(), count);
    EXPECT_EQ(state.count(), moving);
    EXPECT_EQ(y.Macs(), macs);
    EXPECT_EQ(y.features.size(1), 7);
    EXPECT_EQ(y.logits.shape(), n::Shape({1, 1000}));
  }
}
TEST(NnRegNetModels, CompleteCifarGradientsAndMovingStateSemantics) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator gen(29);
  n::models::RegNetConfig cfg;
  cfg.classes = 10;
  cfg.small_stem = true;
  cfg.update_stats = false;
  const auto x = n::normal({2, 32, 32, 3}, 0, .2, gen),
             y = n::models::RegNet(p, state, x, cfg, true).logits;
  EXPECT_EQ(p.count(), 3907554);
  EXPECT_EQ(state.count(), 27152);
  for (const auto &g : n::grad(n::mean(n::square(y)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::RegNet(p, state, x, cfg);
  EXPECT_EQ(saved, state.flatten());
  cfg.update_stats = true;
  n::models::RegNet(p, state, x, cfg, true);
  EXPECT_NE(saved, state.flatten());
}
TEST(NnRegNetModels, RejectsInvalidDesignGroupsAndGeometry) {
  EXPECT_THROW(n::models::RegNetModel("unknown"), std::invalid_argument);
  auto m = n::models::RegNetModel("y400mf");
  m.group = 0;
  EXPECT_THROW(n::models::RegNetDesign(m), std::invalid_argument);
  m = n::models::RegNetModel("y400mf");
  m.wm = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(n::models::RegNetDesign(m), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  EXPECT_THROW(n::models::RegNetBlock(c, "b", n::ones({1, 4, 4, 8}), 16, 1, 0, .25),
               std::invalid_argument);
  EXPECT_THROW(n::models::RegNetConvBN(c, "c", n::ones({1, 4, 4, 8}), 16, 3, 1, 3, true),
               std::invalid_argument);
  EXPECT_THROW(n::models::RegNet(p, state, n::ones({2, 3})), std::invalid_argument);
}
