// CMD: just run-nn bilinear_cnn
// Fine-grained flower classification with two independent VGG16 feature streams.
// https://arxiv.org/abs/1504.07889
// Spatial mean outer product,smoothed signed square root,L2 descriptor and classifier.
// Full two512-channel streams/102way graph;compact32-channel streams/6way training.
// Three default epochs on real Oxford Flowers102 subset,official split isolation.
// Every bilinear/root/normalization/CE gradient,spatial invariance and both-stream
// parameter checks;restored independent immutable RGB-only inference and baselines.
//
// Port of tflow's bilinear_cnn. Preparation (in ~/projects/tflow_examples):
// just fine-grained-assets; flowers.npz is read from $DATAMUNGE_DATA_DIR/flowers102.
// The data and the batch order (nn::BatchOrder seed 239) are tflow's; weights
// come from datamunge's generator, so trained numbers match TF statistically.
// TF builds the full 102-way model as a graph only; here it also runs
// forward/backward once on the 2x64x64 fixture. The checkpoint is best.npz.
// Outputs go to output/nn/models/bilinear_cnn.
// Options: epochs=3 data=PATH checks_only=0|1 threads=N seed=233 (weight seed).

#include <cmath>
#include <iostream>

#include "common/bilinear.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/bilinear.hpp.
namespace fine {
using namespace ::fine;
inline void TrainBilinear(const std::filesystem::path& root, int epochs, const std::filesystem::path& directory,
                          std::uint64_t seed = 233) {
  check::Require(epochs >= 2, "two or more short bilinear epochs");
  const auto    data = LoadFlowers(root);
  BilinearModel model(true, seed);
  const auto    probe     = FlowerRows(data.fit, {0, 8, 16, 24, 32, 40});
  const double  initial   = model.Loss(probe);
  const auto    untrained = FlowerPredict(model, data.test);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_nll\tvalidation_nll\tvalidation_accuracy\n";
  nn::BatchOrder order(48, 4, 239);
  double         best     = INFINITY;
  int            selected = 0;
  std::cout << "bilinear_cnn:" << model.parameters.count() << " parameters/" << model.parameters.size()
            << " tensors;two independent13conv streams,4x4x32 each,meanouter1024D/root/unit/6wayhead;" << epochs
            << " epochs,batch4,Adam.001\n real Flowers102 six-class subset,48fit/48validation/48test from official splits;no "
               "pretraining/augmentation/segmentation/bbox\n balanced fixed fit initial="
            << initial << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double      average = 0;
    std::size_t seen    = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      average += double(ids.size()) * model.Loss(FlowerRows(data.fit, std::vector<std::size_t>(ids.begin(), ids.end())), true);
      seen += ids.size();
    }
    check::Require(seen == 48, "every flower fitting source once per epoch");
    average /= double(seen);
    const auto score = ScoreFlowers(FlowerPredict(model, data.validation), data.validation.labels);
    if (score.nll < best) {
      best = score.nll, selected = epoch;
      model.parameters.save(prefix);
    }
    history << epoch << '\t' << average << '\t' << score.nll << '\t' << score.accuracy << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainNLL=" << average << ",validationNLL=" << score.nll << ",accuracy=" << score.accuracy << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(prefix);
  check::Near(ScoreFlowers(FlowerPredict(model, data.validation), data.validation.labels).nll, best, 1e-8, "restored flower validation agreement");
  const double learned = model.Loss(probe);
  check::Require(learned < initial, "balanced fixed fitting loss improves");
  BilinearModel deployed(false);
  deployed.parameters.load(prefix);
  check::Require(!deployed.optimizer, "only observed RGB is an inference input (no labels/optimizer)");
  const auto before      = deployed.Snapshot();
  const auto probability = FlowerPredict(deployed, data.test), reference = FlowerPredict(model, data.test),
             repeat      = FlowerPredict(deployed, data.test);
  const auto pv = probability.to_vector<float>(), rv = reference.to_vector<float>(), qv = repeat.to_vector<float>();
  double     parity = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    parity = std::max(parity, std::abs(double(pv[i]) - rv[i]));
    check::Near(pv[i], qv[i], 0, "deterministic flower inference");
  }
  check::Near(parity, 0, 1e-7, "standalone bilinear checkpoint parity");
  const auto one = deployed.Predict(FlowerRows(data.test.rgb, {3})).to_vector<float>();
  for (int c = 0; c < 6; ++c) check::Near(one[c], pv[3 * 6 + c], 2e-6, "single/batch flower agreement");
  check::Require(before == deployed.Snapshot(), "all weights unchanged during observed-image inference");
  const nn::Tensor uniform = nn::full({48, 6}, 1.0 / 6);
  const auto       color   = FlowerColorBaseline(data.fit, data.test);
  const auto       score   = ScoreFlowers(probability, data.test.labels);
  std::cout << " selected=" << selected << ",restored validationNLL=" << best << ",balanced fixed fitting " << initial << " -> " << learned << "\n";
  PrintFlowers("learned", score);
  PrintFlowers("untrained same bilinear", ScoreFlowers(untrained, data.test.labels));
  PrintFlowers("uniform", ScoreFlowers(uniform, data.test.labels));
  PrintFlowers("fit-only mean RGB prototypes", ScoreFlowers(color, data.test.labels));
  const nn::Tensor confusion = nn::from_vector(std::vector<std::int32_t>(score.confusion.begin(), score.confusion.end()), {6, 6});
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"rgb", data.test.rgb}, {"labels", data.test.labels}, {"ids", data.test.ids}, {"probability", probability},
                {"untrained", untrained}, {"uniform", uniform}, {"color", color}, {"confusion", confusion},
                {"fit_ids", data.fit.ids}, {"validation_ids", data.validation.ids}});
  FlowerPreview(data.test, probability, directory / "comparison.png");
  if (std::filesystem::exists(root / "manifest.json"))
    std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "flower history written");
  std::cout << " independent RGB-only parity=" << parity << ",all parameters immutable,single/batch agreement;artifacts:" << directory.string()
            << " (training " << seconds << " s)\nComplete short bilinear CNN demonstration passed;custom six-class Flowers subset,not full fine-grained benchmark accuracy.\n";
}
}  // namespace fine



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace fine;

namespace {

void OuterProductChecks() {
  std::vector<float> a(24), b(36), up(12);
  for (int i = 0; i < 24; ++i) a[i] = float((i % 9 - 4) * .17);
  for (int i = 0; i < 36; ++i) b[i] = float((i % 11 - 5) * .23);
  for (int i = 0; i < 12; ++i) up[i] = float((i % 7 - 3) * .19);
  Tensor     first = nn::from_vector(a, {2, 2, 3, 2}).requires_grad_(), second = nn::from_vector(b, {2, 2, 3, 3}).requires_grad_();
  const auto output   = SpatialBilinear(first, second);
  const auto gradient = nn::grad(nn::sum(output * nn::from_vector(up, {2, 2, 3})), {first, second});
  const auto out = output.to_vector<double>(), ga = gradient[0].to_vector<double>(), gb = gradient[1].to_vector<double>();
  std::vector<double> da(a.size()), db(b.size());
  double              error = 0, gradient_error = 0;
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        double expected = 0;
        for (int position = 0; position < 6; ++position) {
          const double av = a[(n * 6 + position) * 2 + i], bv = b[(n * 6 + position) * 3 + j], weight = up[(n * 2 + i) * 3 + j];
          expected += av * bv / 6;
          da[(n * 6 + position) * 2 + i] += weight * bv / 6;
          db[(n * 6 + position) * 3 + j] += weight * av / 6;
        }
        error = std::max(error, std::abs(expected - out[(n * 2 + i) * 3 + j]));
      }
  for (std::size_t i = 0; i < a.size(); ++i) gradient_error = std::max(gradient_error, std::abs(da[i] - ga[i]));
  for (std::size_t i = 0; i < b.size(); ++i) gradient_error = std::max(gradient_error, std::abs(db[i] - gb[i]));
  check::Near(error, 0, 2e-7, "all spatial mean outer-product entries");
  check::Near(gradient_error, 0, 2e-7, "all first AND second feature Jacobian entries");
  std::vector<float> pa(a.size()), pb(b.size());
  const int          permutation[] = {5, 2, 1, 4, 0, 3};
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 6; ++i) {
      for (int c = 0; c < 2; ++c) pa[(n * 6 + i) * 2 + c] = a[(n * 6 + permutation[i]) * 2 + c];
      for (int c = 0; c < 3; ++c) pb[(n * 6 + i) * 3 + c] = b[(n * 6 + permutation[i]) * 3 + c];
    }
  const auto permuted = SpatialBilinear(nn::from_vector(pa, {2, 2, 3, 2}), nn::from_vector(pb, {2, 2, 3, 3})).to_vector<double>();
  for (std::size_t i = 0; i < out.size(); ++i) check::Near(permuted[i], out[i], 2e-7, "aligned location order does not change bilinear pooling");
  const auto unpaired   = SpatialBilinear(nn::from_vector(pa, {2, 2, 3, 2}), nn::from_vector(b, {2, 2, 3, 3})).to_vector<double>();
  double     difference = 0;
  for (std::size_t i = 0; i < out.size(); ++i) difference += std::abs(unpaired[i] - out[i]);
  check::Require(difference > .1, "bilinear is not an outer product of independently averaged features");
  std::cout << " all signed multi-batch outer-product values/BOTH stream Jacobians max errors=" << error << "/" << gradient_error
            << ",joint spatial invariance/unpaired interaction sensitivity passed\n";
}

void RootLossChecks() {
  const std::vector<float> input = {-4, 0, 9, -.01f, 1e-8f, 1}, up = {.3f, -.7f, .2f, -.4f, .6f, .1f};
  Tensor     x        = nn::from_vector(input, {2, 3}).requires_grad_();
  const auto output   = BilinearUnit(BilinearRoot(x));
  const auto gradient = nn::grad(nn::sum(output * nn::from_vector(up, {2, 3})), {x})[0].to_vector<double>();
  const auto out      = output.to_vector<double>();
  double     error    = 0;
  for (int n = 0; n < 2; ++n) {
    std::array<double, 3> root{}, derivative{};
    double                norm2 = 0, dot = 0;
    for (int c = 0; c < 3; ++c) {
      const double value = input[n * 3 + c], den = std::abs(value) + double(1e-5f);
      root[c]       = value / std::sqrt(den);
      derivative[c] = (.5 * std::abs(value) + double(1e-5f)) / std::pow(den, 1.5);
      norm2 += root[c] * root[c];
      dot += root[c] * up[n * 3 + c];
    }
    const double norm = std::sqrt(norm2);
    for (int c = 0; c < 3; ++c) {
      check::Near(out[n * 3 + c], root[c] / norm, 2e-7, "every signed normalized bilinear value");
      const double expected = (up[n * 3 + c] / norm - root[c] * dot / std::pow(norm, 3)) * derivative[c];
      error                 = std::max(error, std::abs(expected - gradient[n * 3 + c]));
    }
  }
  check::Near(error, 0, 3e-5, "all smoothed signed-root and L2 Jacobian coordinates including atzero");
  Tensor     zero = nn::zeros({2, 3}).requires_grad_();
  const auto zout = BilinearUnit(BilinearRoot(zero));
  const auto zg   = nn::grad(nn::sum(zout * nn::from_vector(up, {2, 3})), {zero})[0].to_vector<double>();
  for (double v : zout.to_vector<double>()) check::Near(v, 0, 0, "zero bilinear descriptor stayszero");
  for (double v : zg) check::Require(std::isfinite(v), "zero-root/zero-norm derivative finite");
  const std::vector<float> scores = {1000, 999, 998, 997, 996, 995, -3, 2, .5f, -1, 1, 0};
  Tensor     z    = nn::from_vector(scores, {2, 6}).requires_grad_();
  const auto loss = FlowerCrossEntropy(z, nn::tensor({0, 4}).to(nn::DType::Int32), 6);
  const auto gz   = nn::grad(loss, {z})[0].to_vector<double>();
  double     expected_loss = 0;
  for (int n = 0; n < 2; ++n) {
    double maximum = -INFINITY, sum = 0;
    for (int c = 0; c < 6; ++c) maximum = std::max(maximum, double(scores[n * 6 + c]));
    for (int c = 0; c < 6; ++c) sum += std::exp(scores[n * 6 + c] - maximum);
    const int target = n ? 4 : 0;
    expected_loss += (std::log(sum) + maximum - scores[n * 6 + target]) / 2;
    for (int c = 0; c < 6; ++c)
      check::Near(gz[n * 6 + c], (std::exp(scores[n * 6 + c] - maximum) / sum - (c == target)) / 2, 1e-7, "every stable CE logit derivative");
  }
  check::Near(loss.item(), expected_loss, 1e-7, "known logsumexp NLL");
  const auto metric = ScoreFlowers(nn::full({6, 6}, 1.0 / 6), nn::arange(0, 6, 1, nn::DType::Int32));
  check::Near(metric.nll, std::log(6.), 1e-7, "uniform flower NLL");
  check::Near(metric.accuracy, 1. / 6, 1e-12, "deterministic first-class tie accuracy");
  check::Near(metric.macro_accuracy, 1. / 6, 1e-12, "balanced macro accuracy");
  check::Near(metric.brier, 5. / 6, 1e-7, "six-way Brier sum,not channel average");
  std::cout << " signed-root/unit every analytic Jacobian max error=" << error
            << ",zero safety,extreme CE/all derivatives and known NLL/accuracy/Brier fixtures passed\n";
}

std::int64_t Census(int divisor, int classes) {
  const int    widths[] = {64, 128, 256, 512, 512}, layers[] = {2, 2, 3, 3, 3};
  std::int64_t one = 0;
  int          in  = 3;
  for (int block = 0; block < 5; ++block)
    for (int layer = 0; layer < layers[block]; ++layer) {
      const int out = widths[block] / divisor;
      one += 9LL * in * out + out;
      in = out;
    }
  return 2 * one + std::int64_t(in) * in * classes + classes;
}

void ModelChecks() {
  for (int divisor : {1, 16}) {
    const int          classes = divisor == 1 ? 102 : 6;
    nn::ParameterStore p(233);
    std::vector<float> input(2 * 64 * 64 * 3), up(std::size_t(2) * classes);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(.5 + .4 * std::sin(i * .13));
    for (std::size_t i = 0; i < up.size(); ++i) up[i] = float((int(i) % 7 - 3) * .11);
    Tensor     image = nn::from_vector(input, {2, 64, 64, 3}).requires_grad_();
    const auto net   = BilinearCNN(p, image, divisor, classes);
    auto       vars  = p.parameters();
    vars.push_back(image);
    const auto gradients = nn::grad(nn::sum(net.logits * nn::from_vector(up, {2, classes})), vars);
    check::Require(p.count() == Census(divisor, classes) && p.size() == 54 && net.stages.size() == 37, "independent parameter/26conv/two-stream/head census");
    std::cout << " " << (divisor == 1 ? "full" : "compact") << " bilinear:" << p.count()
              << " parameters/54 derivatives (executed),37stages,two independent feature stores\n";
    if (divisor == 1) continue;
    std::vector<nn::Shape> expected;
    const int              widths[] = {4, 8, 16, 32, 32}, layers[] = {2, 2, 3, 3, 3};
    for (int stream = 0; stream < 2; ++stream) {
      int side = 64;
      for (int block = 0; block < 5; ++block) {
        for (int layer = 0; layer < layers[block]; ++layer) expected.push_back({2, side, side, widths[block]});
        if (block < 4) {
          side /= 2;
          expected.push_back({2, side, side, widths[block]});
        }
      }
    }
    expected.push_back({2, 32, 32});
    expected.push_back({2, 1024});
    expected.push_back({2, 6});
    for (int i = 0; i < 37; ++i) check::Require(net.stages[i].shape() == expected[i], "all exact two-stream/spatial outer/root/unit/head shapes");
    const auto d = net.stages[35].to_vector<double>();
    for (int n = 0; n < 2; ++n) {
      double norm = 0;
      for (int c = 0; c < 1024; ++c) norm += d[n * 1024 + c] * d[n * 1024 + c];
      check::Near(norm, 1, 2e-6, "unit bilinear descriptor");
    }
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite both-stream parameter/RGB derivative");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "every54 tensor in BOTH independent streams/head and RGB receives gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all54parameter and RGB gradients finite/nonzero,norm range[" << minimum << "," << maximum
              << "],all37 exact stages and unit descriptors passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::uint64_t         seed        = 233;
    std::filesystem::path root        = paths::Dataset("flowers102");
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
      else if (arg.rfind("seed=", 0) == 0)
        seed = std::stoull(arg.substr(5));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N seed=N");
    }
    OuterProductChecks();
    RootLossChecks();
    ModelChecks();
    if (checks_only) return 0;
    check::Require(std::filesystem::exists(root / "flowers.npz"), "real flowers missing;run just fine-grained-assets");
    TrainBilinear(root, epochs, paths::OutputDir("models/bilinear_cnn"), seed);
    std::cout << "All bilinear CNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
