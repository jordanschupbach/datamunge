
#include <datamunge/image/jpeg.hpp>
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_vqa_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/vqa_baseline.cpp"
#undef main
#include "../fixtures/nn_vqa_reference.hpp"
TEST(Vqa, IndependentEveryLstmStateAndEveryCoordinateFiniteDifference) {
  LstmChecks();
}
TEST(Vqa, StableSoftAnswerLossEveryAdjointAndOfficialCountFormula) {
  LossScoreChecks();
}
TEST(Vqa, FullCompactAllStagesAndEveryLearnedModalityGradient) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(Vqa, EveryOriginalJpegPixelAndTenHumanAnswerPreparation) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const std::filesystem::path root = "build/cnn-datasets/vqa";
  const auto data = m::LoadVqa(root);
  std::map<int, std::vector<float>> images;
  std::map<int, std::pair<int, int>> dimensions;
  std::ifstream sources(root / "source_images.tsv");
  int id;
  std::string path;
  while (sources >> id >> path) {
    dm::image::JpegDecodeOptions options;
    options.channels = 3;
    const auto original = nn::from_image(dm::image::read_jpeg(path, options));
    dimensions[id] = {int(original.size(0)), int(original.size(1))};
    images[id] = original.to_vector<float>();
  }
  std::map<std::string, m::VqaData> parts = {
      {"fit", data.fit}, {"validation", data.validation}, {"test", data.test}};
  std::map<std::string, int> offsets;
  std::set<int> checked;
  std::ifstream records(root / "records.tsv");
  std::string line;
  int total = 0;
  while (std::getline(records, line)) {
    std::istringstream row(line);
    std::string split;
    std::int64_t qid;
    int image, type, length;
    row >> split >> image >> qid >> type >> length;
    std::vector<int> tokens(24), counts(data.config.answers);
    for (auto &v : tokens)
      row >> v;
    for (int a = 0; a < 10; ++a) {
      int answer;
      row >> answer;
      ++counts[answer];
    }
    const auto &d = parts.at(split);
    const int at = offsets[split]++;
    EXPECT_EQ(d.question_ids.to_vector<std::int64_t>()[at], qid);
    EXPECT_EQ(d.image_ids.to_vector<int>()[at], image);
    EXPECT_EQ(d.lengths.to_vector<int>()[at], length);
    EXPECT_EQ(d.types.to_vector<int>()[at], type);
    const auto tv = nn::select(d.tokens, 0, at).to_vector<int>(),
               prefix = nn::select(d.prefix, 0, at).to_vector<int>();
    EXPECT_EQ(tv, tokens);
    EXPECT_EQ(prefix[0], tokens[0]);
    EXPECT_EQ(prefix[1], tokens[1]);
    const auto freq = nn::select(d.target, 0, at).to_vector<float>(),
               score = nn::select(d.score, 0, at).to_vector<float>();
    int best = 0;
    for (int c = 0; c < data.config.answers; ++c) {
      EXPECT_FLOAT_EQ(freq[c], float(counts[c]) / 10.f);
      double sum = 0;
      for (int a = 0; a < 10; ++a)
        sum += std::min(1., double(counts[c] - (a < counts[c])) / 3.);
      EXPECT_NEAR(score[c], c ? sum / 10. : 0., 3e-8);
      if (counts[c] > counts[best])
        best = c;
    }
    EXPECT_EQ(d.hard.to_vector<int>()[at], best);
    if (checked.insert(image).second) {
      const auto rgb = nn::select(d.rgb, 0, at).to_vector<float>();
      const auto [h, w] = dimensions.at(image);
      const auto &raw = images.at(image);
      for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
          for (int c = 0; c < 3; ++c) {
            float sy = float(y) * h / 64.f, sx = float(x) * w / 64.f, dy = sy - std::floor(sy),
                  dx = sx - std::floor(sx);
            int y0 = int(sy), x0 = int(sx), y1 = std::min(y0 + 1, h - 1),
                x1 = std::min(x0 + 1, w - 1);
            auto pixel = [&](int iy, int ix) {
              return raw[(iy * w + ix) * 3 + c] / 255.f;
            };
            const float a = pixel(y0, x0) + (pixel(y0, x1) - pixel(y0, x0)) * dx,
                        b = pixel(y1, x0) + (pixel(y1, x1) - pixel(y1, x0)) * dx;
            EXPECT_NEAR(rgb[(y * 64 + x) * 3 + c], a + (b - a) * dy, 2e-7);
          }
    }
    ++total;
  }
  EXPECT_EQ(total, 415);
  EXPECT_EQ(checked.size(), 85);
  EXPECT_EQ(m::VqaText(root).size(), 415);
}
TEST(Vqa, EveryRealMetricPriorAndPreviewPixelMatchesIndependentHost) {
  namespace m = dm::nn::models;
  namespace r = vqa_reference;
  const auto split = m::LoadVqa("build/cnn-datasets/vqa");
  const auto &d = split.test;
  const auto probability = nn::softmax(
      nn::sin(nn::arange(0, d.target.numel()) * .137).reshape(d.target.shape()) * 3., 1);
  const r::VqaData legacy{d.rgb,  d.tokens,    d.lengths,      d.target, d.score,
                          d.hard, d.image_ids, d.question_ids, d.types,  d.prefix};
  const r::VqaData fit{split.fit.rgb,       split.fit.tokens,       split.fit.lengths,
                       split.fit.target,    split.fit.score,        split.fit.hard,
                       split.fit.image_ids, split.fit.question_ids, split.fit.types,
                       split.fit.prefix};
  const auto a = m::ScoreVqa(probability, d);
  const auto b = r::ScoreVqa(probability, legacy);
  EXPECT_NEAR(a.nll, b.nll, 2e-14);
  EXPECT_DOUBLE_EQ(a.consensus, b.consensus);
  EXPECT_DOUBLE_EQ(a.exact, b.exact);
  EXPECT_DOUBLE_EQ(a.coverage, b.coverage);
  EXPECT_DOUBLE_EQ(a.oracle, b.oracle);
  EXPECT_EQ(a.type_count, b.type_count);
  EXPECT_EQ(a.type_score, b.type_score);
  for (bool prefix : {false, true})
    EXPECT_LE(
        nn::max(nn::abs(m::VqaPrior(split.fit, d, prefix) - r::VqaPrior(fit, legacy, prefix)))
            .item(),
        3e-8);
  const auto tiles = m::VqaPreviewTiles(d, probability).to_vector<float>(),
             p = probability.to_vector<float>(), score = d.score.to_vector<float>(),
             rgb = d.rgb.to_vector<float>();
  const int classes = split.config.answers;
  for (int n = 0; n < d.rgb.size(0); ++n) {
    int answer = 0;
    for (int c = 1; c < classes; ++c)
      if (p[n * classes + c] > p[n * classes + answer])
        answer = c;
    const float s = score[n * classes + answer];
    for (int y = 0; y < 80; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c) {
          float expected = 0;
          if (y < 64) {
            bool border = x < 2 || y < 2 || x >= 62 || y >= 62;
            bool color = s >= .6 ? c == 1 : s > 0 ? c < 2 : c == 0;
            expected = border ? float(color)
                              : float(std::clamp(
                                    std::lround(rgb[((n * 64 + y) * 64 + x) * 3 + c] * 255.f),
                                    0l, 255l)) /
                                    255.f;
          } else if (x < classes) {
            const int height = int(std::lround(14.f * p[n * classes + x]));
            expected = (y >= 80 - height && (c == (x == answer ? 1 : 2))) ? 1.f : 0.f;
          }
          EXPECT_FLOAT_EQ(tiles[((n * 80 + y) * 64 + x) * 3 + c], expected);
        }
  }
}
TEST(Vqa, FitOnlyCountsHaveVisibleSgdParityAndExactLargeQuestionIds) {
  namespace m = dm::nn::models;
  const auto d = m::LoadVqa("build/cnn-datasets/vqa");
  const auto target = d.fit.target.to(nn::DType::Float64), prefix = d.fit.prefix;
  const auto group = nn::all(nn::equal(nn::expand_dims(prefix, 1), nn::expand_dims(prefix, 0)),
                             {2})
                         .to(nn::DType::Float64),
             global_target = nn::sum(target, {0}, true) + 1.,
             prefix_target = nn::matmul(group, target);
  nn::ParameterStore p(281, nn::DType::Float64);
  const auto global = p.get("global", global_target.shape(), nn::init::zeros()),
             local = p.get("prefix", prefix_target.shape(), nn::init::zeros());
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(p, .5, c);
  sgd.minimize(nn::sum(nn::square(global - global_target)) +
               nn::sum(nn::square(local - prefix_target)));
  EXPECT_EQ(nn::max(nn::abs(global - global_target)).item(), 0.);
  EXPECT_EQ(nn::max(nn::abs(local - prefix_target)).item(), 0.);
  const auto ids =
      nn::from_vector(std::vector<std::int64_t>{9007199254740993ll, 9007199254740995ll}, {2});
  EXPECT_EQ(m::VqaRows(ids, {1}).item<std::int64_t>(), 9007199254740995ll);
}
