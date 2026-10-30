// CMD: just run-nn vqa_baseline
// Human visual question answering:CNN plus masked LSTM and multimodal fusion.
// Real pinned VQA-v2 training questions/10 answers matched to cached COCO images.
// Fit-only token/answer vocabularies and disjoint image pools;no templated QA.
// Three default epochs per joint/question-only/image-only model;soft answer CE,
// official leave-one-annotator-out score and coverage/oracle limits reported.
// Native masked-LSTM value/every-weight-gradient,padding/order,softCE/score,
// full/compact model gradients and independent observed-input restoration checks.
//
// Port of tflow's vqa_baseline. Preparation (in ~/projects/tflow_examples): just
// vqa-assets; vqa.npz, questions.tsv and answer_vocabulary.tsv are read from
// $DATAMUNGE_DATA_DIR/vqa. Data, vocabularies and batch order (nn::BatchOrder
// seed 281) are tflow's; weights come from datamunge's generator (store seed
// 277), so trained numbers match TF statistically. TF builds the full model
// (base64/emb300/hidden256/fusion512) as a graph only; here it also runs one
// forward/backward pass on the 2-image fixture. TF's "deployment GraphDef has
// only image/tokens/lengths placeholders" check becomes a structural check on
// a second model built without optimizer or target input. Checkpoints are
// best.npz; outputs go to output/nn/models/vqa_baseline/<mode>.
// Options: epochs=3 data=PATH checks_only=0|1 threads=N.

#include <datamunge/datasets/assets.hpp>
#include <cmath>
#include <iostream>

#include "common/paths.hpp"
#include "common/vqa.hpp"

// Experiment workflow from examples/nn/common/vqa.hpp.
namespace vqa {
using namespace ::vqa;
inline void TrainVqaMode(const std::filesystem::path& root, const VqaSplits& data, VqaMode mode, int epochs, const std::filesystem::path& base) {
  const auto   name = VqaModeName(mode);
  VqaModel     model(data.config, mode);
  const auto   probe     = VqaRows(data.fit, {0, 1, 2, 3, 4, 5, 6, 7});
  const double initial   = model.Loss(probe);
  const auto   untrained = VqaPredict(model, data.test);
  const auto   directory = base / name;
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_softCE\tvalidation_softCE\tvalidation_consensus\n";
  nn::BatchOrder order(std::size_t(data.fit.rgb.size(0)), 8, 281);
  double         best     = INFINITY;
  int            selected = 0;
  std::cout << "vqa " << name << ":" << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors;" << epochs
            << " epochs,batch8,Adam.001,human questions fit/val/test=" << data.fit.rgb.size(0) << "/" << data.validation.rgb.size(0) << "/"
            << data.test.rgb.size(0) << ",vocab=" << data.config.vocabulary << ",answers=" << data.config.answers << ",T=" << data.config.time
            << ",fixed fit=" << initial << std::endl;
  const auto start = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    std::size_t seen    = 0;
    double      average = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      std::vector<std::size_t> rows(ids.begin(), ids.end());
      average += double(rows.size()) * model.Loss(VqaRows(data.fit, rows), true);
      seen += rows.size();
    }
    check::Require(seen == std::size_t(data.fit.rgb.size(0)), "all fitting human questions once per epoch");
    average /= double(seen);
    const auto score = ScoreVqa(VqaPredict(model, data.validation), data.validation);
    if (score.nll < best) {
      best = score.nll, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << average << '\t' << score.nll << '\t' << score.consensus << '\n';
    history.flush();
    std::cout << " " << name << " epoch " << epoch << ":trainCE=" << average << ",validationCE=" << score.nll << ",consensus=" << score.consensus
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  model.Restore(prefix);
  check::Near(ScoreVqa(VqaPredict(model, data.validation), data.validation).nll, best, 1e-8, "restored VQA validation agreement");
  const double learned = model.Loss(probe);
  check::Require(learned < initial, "fixed soft10-answer fitting loss improves");
  // Deployment: an independent model without optimizer/target input restored from best.npz.
  VqaModel deployed(data.config, mode, false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer && deployed.parameters.names() == model.parameters.names(),
                 "only observed image/question IDs/lengths and checkpoint filename");
  const auto before      = deployed.Snapshot();
  const auto probability = VqaPredict(deployed, data.test), reference = VqaPredict(model, data.test), repeat = VqaPredict(deployed, data.test);
  const auto pv = probability.to_vector<float>(), rv = reference.to_vector<float>(), qv = repeat.to_vector<float>();
  double     parity = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    parity = std::max(parity, std::abs(double(pv[i]) - rv[i]));
    check::Near(pv[i], qv[i], 0, "deterministic VQA inference");
  }
  check::Near(parity, 0, 1e-7, "independent VQA checkpoint parity");
  const int  answers = data.config.answers;
  const auto one     = deployed.Predict(VqaRows(data.test, {3})).to_vector<float>();
  for (int c = 0; c < answers; ++c) check::Near(one[std::size_t(c)], pv[std::size_t(3 * answers + c)], 2e-6, "single/batch multimodal answer agreement");
  const auto image_ids   = data.test.image_ids.to_vector<std::int32_t>();
  auto       observed    = VqaRows(data.test, {0});
  std::size_t other_image = 1;
  while (other_image < image_ids.size() && image_ids[other_image] == image_ids[0]) ++other_image;
  check::Require(other_image < image_ids.size(), "different test image for modality control");
  const auto original = deployed.Predict(observed).to_vector<float>();
  observed.rgb        = VqaRows(data.test.rgb, {other_image});
  const auto changed_image = deployed.Predict(observed).to_vector<float>();
  observed            = VqaRows(data.test, {0});
  observed.tokens     = VqaRows(data.test.tokens, {1});
  observed.lengths    = VqaRows(data.test.lengths, {1});
  const auto changed_question = deployed.Predict(observed).to_vector<float>();
  double     image_change = 0, question_change = 0;
  for (int c = 0; c < answers; ++c) {
    image_change += std::abs(double(original[std::size_t(c)]) - changed_image[std::size_t(c)]);
    question_change += std::abs(double(original[std::size_t(c)]) - changed_question[std::size_t(c)]);
  }
  image_change /= answers, question_change /= answers;
  check::Require(mode == VqaMode::kQuestion ? image_change == 0 : image_change > 1e-7, "expected image dependence or image-free baseline invariance");
  check::Require(mode == VqaMode::kImage ? question_change == 0 : question_change > 1e-7,
                 "expected question dependence or question-free baseline invariance");
  check::Require(before == deployed.Snapshot(), "all CNN/LSTM/embedding/fusion values immutable");
  std::cout << " " << name << " selected=" << selected << ",restored valCE=" << best << ",fixed fitting " << initial << " -> " << learned
            << ",observed image/question sensitivity=" << image_change << "/" << question_change << ",train wall=" << seconds << "s" << std::endl;
  PrintVqa(name, probability, data.test);
  PrintVqa(name + " untrained", untrained, data.test);
  const auto global = VqaPrior(data.fit, data.test), prefix_prior = VqaPrior(data.fit, data.test, true);
  if (mode == VqaMode::kJoint) {
    PrintVqa("fit answer prior", global, data.test);
    PrintVqa("fit question-prefix prior", prefix_prior, data.test);
  }
  const auto    words = VqaAnswers(root, answers);
  const auto    text  = VqaText(root);
  const auto    qids  = data.test.question_ids.to_vector<std::int64_t>();
  const auto    score = data.test.score.to_vector<float>();
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "question_id\timage_id\tquestion\tmajority\tprediction\tconsensus\n";
  for (int n = 0; n < int(probability.size(0)); ++n) {
    const int   answer = VqaArgmax(pv, n, answers);
    const auto& q      = text.at(qids[std::size_t(n)]);
    predictions << qids[std::size_t(n)] << '\t' << image_ids[std::size_t(n)] << '\t' << q.first << '\t' << q.second << '\t' << words[std::size_t(answer)]
                << '\t' << score[std::size_t(n * answers + answer)] << '\n';
  }
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"rgb", data.test.rgb},
                {"tokens", data.test.tokens},
                {"lengths", data.test.lengths},
                {"target", data.test.target},
                {"consensus", data.test.score},
                {"hard", data.test.hard},
                {"image_ids", data.test.image_ids},
                {"question_ids", data.test.question_ids},
                {"types", data.test.types},
                {"probability", probability},
                {"untrained", untrained},
                {"answer_prior", global},
                {"prefix_prior", prefix_prior},
                {"fit_image_ids", data.fit.image_ids},
                {"validation_image_ids", data.validation.image_ids}});
  VqaPreview(data.test, probability, directory / "comparison.png");
  std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(predictions) && bool(history), "save VQA histories/raw question predictions");
  std::cout << " " << name << " standalone parity=" << parity << ",single/batch/immutable parameters passed;artifacts:" << directory.string() << std::endl;
}
}  // namespace vqa

// Experiment workflow from examples/nn/common/vqa.hpp.
namespace vqa {
using namespace ::vqa;
inline void TrainVqa(const std::filesystem::path& root, int epochs, const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "at least two short VQA epochs");
  const auto data = LoadVqa(root);
  for (auto mode : {VqaMode::kJoint, VqaMode::kQuestion, VqaMode::kImage}) TrainVqaMode(root, data, mode, epochs, directory);
  std::cout << "Complete CNN/LSTM VQA and independently trained unimodal baselines passed;custom image-held-out VQA-v2 subset,not official benchmark "
               "accuracy.\n";
}
}  // namespace vqa



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace vqa;

namespace {

struct HostLstm {
  std::array<double, 2>              h{}, c{};
  std::vector<std::array<double, 2>> steps;
};
HostLstm Host(const std::vector<double>& table, const std::vector<double>& wx, const std::vector<double>& wh, const std::vector<double>& bias,
              const std::array<int, 4>& tokens, int length) {
  HostLstm result;
  for (int time = 0; time < 4; ++time) {
    if (time < length) {
      std::array<double, 8> gates{};
      for (int j = 0; j < 8; ++j) {
        gates[j] = bias[j];
        for (int k = 0; k < 2; ++k) gates[j] += table[2 * tokens[time] + k] * wx[k * 8 + j] + result.h[k] * wh[k * 8 + j];
      }
      for (int k = 0; k < 2; ++k) {
        const double i = 1 / (1 + std::exp(-gates[k])), f = 1 / (1 + std::exp(-gates[2 + k])), g = std::tanh(gates[4 + k]),
                     o = 1 / (1 + std::exp(-gates[6 + k]));
        result.c[k] = f * result.c[k] + i * g;
        result.h[k] = o * std::tanh(result.c[k]);
      }
    }
    result.steps.push_back(result.h);
  }
  return result;
}

Tensor Ints(const std::vector<int>& values, nn::Shape shape) {
  return nn::from_vector(std::vector<std::int32_t>(values.begin(), values.end()), shape);
}

void LstmChecks() {
  nn::ParameterStore p(283);
  VqaConfig          config;
  config.vocabulary = 6, config.embedding = 2, config.hidden = 2, config.time = 4;
  {
    nn::NoGradGuard guard;
    VqaLstm(p, Ints({0, 0, 0, 0, 0, 0, 0, 0}, {2, 4}), Ints({1, 1}, {2}), config);
  }
  std::vector<double> table(12), wx(16), wh(16), bias = {.1, -.2, 1, .7, .05, -.1, .2, -.3};
  for (int i = 0; i < 12; ++i) table[i] = (i % 7 - 3) * .13;
  for (int i = 0; i < 16; ++i) wx[i] = (i % 5 - 2) * .11, wh[i] = (i % 7 - 3) * .07;
  const std::vector<std::string>         variable_names = {"question/embedding/table", "question/lstm/Wx", "question/lstm/Wh", "question/lstm/b"};
  const std::vector<std::vector<double>> data           = {table, wx, wh, bias};
  check::Require(p.names() == variable_names, "embedding/LSTM variable creation order");
  for (int v = 0; v < 4; ++v) {
    std::vector<float> value(data[v].begin(), data[v].end());
    p.set(variable_names[v], nn::from_vector(value, p.parameters()[std::size_t(v)].shape()));
  }
  const Tensor weights = nn::tensor({.3, -.4, .2, .1}, {2, 2});
  auto         run     = [&](const std::vector<int>& ids, const std::vector<int>& size) { return VqaLstm(p, Ints(ids, {2, 4}), Ints(size, {2}), config); };
  const auto   result  = run({2, 3, 0, 0, 0, 0, 0, 0}, {2, 0});
  const auto   gradients = nn::grad(nn::sum(result.h * weights), p.parameters());
  const auto   expected  = Host(table, wx, wh, bias, {2, 3, 0, 0}, 2);
  double       value_error = 0;
  for (int time = 0; time < 4; ++time) {
    const auto step = result.steps[std::size_t(time)].to_vector<double>();
    for (int c = 0; c < 2; ++c) {
      value_error = std::max(value_error, std::abs(expected.steps[std::size_t(time)][std::size_t(c)] - step[std::size_t(c)]));
      check::Near(step[std::size_t(2 + c)], 0, 0, "zero-length question state remainszero");
    }
  }
  const auto cell = result.c.to_vector<double>();
  for (int c = 0; c < 2; ++c) check::Near(cell[std::size_t(c)], expected.c[std::size_t(c)], 1e-7, "cell memory freezes after last valid token");
  check::Near(value_error, 0, 1e-7, "independent signed LSTM gate/state values");
  double       gradient_error = 0;
  const double epsilon        = 1e-3;
  for (int variable = 0; variable < 4; ++variable) {
    const auto g = gradients[std::size_t(variable)].to_vector<double>();
    for (int coordinate = 0; coordinate < int(data[std::size_t(variable)].size()); ++coordinate) {
      auto plus = data, minus = data;
      plus[std::size_t(variable)][std::size_t(coordinate)] += epsilon;
      minus[std::size_t(variable)][std::size_t(coordinate)] -= epsilon;
      const auto   a = Host(plus[0], plus[1], plus[2], plus[3], {2, 3, 0, 0}, 2), b = Host(minus[0], minus[1], minus[2], minus[3], {2, 3, 0, 0}, 2);
      const double derivative = (.3 * (a.h[0] - b.h[0]) - .4 * (a.h[1] - b.h[1])) / (2 * epsilon);
      gradient_error          = std::max(gradient_error, std::abs(derivative - g[std::size_t(coordinate)]));
    }
  }
  check::Near(gradient_error, 0, 2e-6, "EVERY embedding/input/recurrent/bias coordinate host finite difference");
  const auto table_gradient = gradients[0].to_vector<double>();
  for (int coordinate = 0; coordinate < 2; ++coordinate) check::Near(table_gradient[std::size_t(coordinate)], 0, 0, "padding embedding receives zero gradient");
  nn::NoGradGuard guard;
  const auto      padded = run({2, 3, 5, 4, 1, 2, 4, 5}, {2, 0}).h.to_vector<double>();
  for (int c = 0; c < 2; ++c) check::Near(padded[std::size_t(c)], expected.h[std::size_t(c)], 1e-7, "padding values cannot change question state");
  const auto reordered = run({3, 2, 0, 0, 0, 0, 0, 0}, {2, 0}).h.to_vector<double>();
  double     change    = 0;
  for (int c = 0; c < 2; ++c) change += std::abs(reordered[std::size_t(c)] - expected.h[std::size_t(c)]);
  check::Require(change > 1e-5, "question encoder is sequence-sensitive,not bag averaging");
  std::cout << " signed masked LSTM every-state error=" << value_error << ",ALL52 parameter-coordinate host FD error=" << gradient_error
            << ",zero-length/cell freeze/padding/zero-PAD-gradient/order checks passed\n";
}

void LossScoreChecks() {
  const std::vector<float> logits = {1000, 999, 998, -2, 1, .5}, target = {.6f, .3f, .1f, 0, .2f, .8f};
  Tensor                   z      = nn::from_vector(logits, {2, 3}).requires_grad_();
  const auto               loss   = VqaSoftCE(z, nn::from_vector(target, {2, 3}));
  const auto               g      = nn::grad(loss, {z})[0].to_vector<double>();
  double                   expected = 0;
  for (int n = 0; n < 2; ++n) {
    double maximum = -INFINITY, sum = 0, total = 0;
    for (int c = 0; c < 3; ++c) maximum = std::max(maximum, double(logits[n * 3 + c])), total += target[n * 3 + c];
    for (int c = 0; c < 3; ++c) sum += std::exp(logits[n * 3 + c] - maximum);
    for (int c = 0; c < 3; ++c) {
      const double probability = std::exp(logits[n * 3 + c] - maximum) / sum;
      expected += target[n * 3 + c] * (std::log(sum) + maximum - logits[n * 3 + c]) / 2;
      check::Near(g[std::size_t(n * 3 + c)], (probability * total - target[n * 3 + c]) / 2, 1e-7,
                  "EVERY soft-target CE derivative including observed targetmass");
    }
  }
  check::Near(loss.item(), expected, 2e-7, "stable ten-answer target mean CE");
  const double known[] = {0, .3, .6, .9, 1, 1, 1, 1, 1, 1, 1};
  for (int matching = 0; matching <= 10; ++matching)
    check::Near(VqaConsensusCount(matching), known[matching], 1e-12, "official10-way leave-one-out consensus is NOTmin(k/3,1)");
  VqaData data;
  data.target            = nn::tensor({0, 1, 0, 1, 0, 0}, {2, 3});
  data.score             = nn::tensor({0, .6, .3, 0, 0, 0}, {2, 3});
  data.hard              = Ints({1, 0}, {2});
  data.types             = Ints({0, 2}, {2});
  const auto probability = nn::tensor({.1, .8, .1, .8, .1, .1}, {2, 3});
  const auto score       = ScoreVqa(probability, data);
  check::Near(score.consensus, .3, 2e-8, "known score lookup and UNK-answer alwayszero");
  check::Near(score.exact, .5, 0, "unknown-majority never counted as correct unknownprediction");
  check::Near(score.coverage, .5, 0, "held-out answer-vocabulary coverage");
  check::Near(score.oracle, .3, 2e-8, "closed-vocabulary oracle score cap");
  std::cout << " every stable soft-answer CE value/gradient,official0/.3/.6/.9/1 consensus,UNK-zero/coverage/oracle/type fixtures passed\n";
}

std::int64_t Census(const VqaConfig& c) {
  std::int64_t total = 0;
  int          in    = 3;
  for (int block = 0; block < 3; ++block) {
    const int out = c.base * (1 << block);
    for (int layer = 0; layer < 2; ++layer) total += 9LL * in * out + out, in = out;
  }
  total += std::int64_t(in) * c.fusion + c.fusion;
  total += std::int64_t(c.vocabulary) * c.embedding + std::int64_t(c.embedding + c.hidden) * 4 * c.hidden + 4 * c.hidden + std::int64_t(c.hidden) * c.fusion +
           c.fusion;
  total += 3LL * c.fusion * c.fusion + c.fusion + std::int64_t(c.fusion) * c.answers + c.answers;
  return total;
}

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(277);
    VqaConfig          config;
    if (compact) config.base = 8, config.embedding = 32, config.hidden = 32, config.fusion = 64;
    std::vector<float> rgb(2 * 64 * 64 * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(.5 + .4 * std::sin(i * .13));
    std::vector<int> ids(48, 0);
    for (int t = 0; t < 4; ++t) ids[std::size_t(t)] = t + 2, ids[std::size_t(24 + t)] = 6 - t;
    Tensor     image     = nn::from_vector(rgb, {2, 64, 64, 3}).requires_grad_();
    const auto net       = VqaBaseline(p, image, Ints(ids, {2, 24}), Ints({4, 3}, {2}), config);
    const auto objective = nn::mean(nn::square(net.logits));
    auto       inputs    = p.parameters();
    inputs.push_back(image);
    inputs.push_back(net.question_h);
    const auto gradients = nn::grad(objective, inputs);
    check::Require(p.count() == Census(config) && p.size() == 24 && net.stages.size() == 14 && net.question_steps.size() == 24,
                   "independent VQA parameter/stage/shared-LSTM census");
    std::cout << " " << (compact ? "compact" : "full") << " joint VQA:" << p.count() << " parameters/24 derivatives (executed),14fusion stages/24 shared LSTM steps\n";
    if (!compact) continue;
    std::vector<nn::Shape> expected;
    int                    side = 64;
    for (int block = 0; block < 3; ++block) {
      for (int layer = 0; layer < 2; ++layer) expected.push_back({2, side, side, 8 << block});
      side /= 2;
      expected.push_back({2, side, side, 8 << block});
    }
    for (std::int64_t w : {64, 64, 192, 64, 64}) expected.push_back({2, w});
    for (int i = 0; i < 14; ++i) check::Require(net.stages[std::size_t(i)].shape() == expected[std::size_t(i)], "all exact image/fusion/head geometries");
    for (const auto& h : net.question_steps) check::Require(h.shape() == nn::Shape({2, 32}), "all24 recurrent step state shapes");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite VQA tensor/modality gradients");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "ALL24parameter tensors and BOTH modalities receive nonzero gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all24 learned and image/question gradients finite/nonzero,norm range[" << minimum << "," << maximum
              << "],14CNN/fusion and24recurrent shapes passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path root        = paths::Dataset("vqa");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        root = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    LstmChecks();
    LossScoreChecks();
    ModelChecks();
    if (checks_only) return 0;
    if (!std::filesystem::exists(root / "vqa.npz"))
      datamunge::datasets::prepare_example_data("vqa", root.string());
    TrainVqa(root, epochs, paths::OutputDir("models/vqa_baseline"));
    std::cout << "All VQA baseline checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
