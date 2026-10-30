// CMD: just run-nn siamese
// Siamese CNN verification and class-disjoint one-shot similarity learning.
// https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
// Shared four-VALID-convolution ReLU encoder, three max pools, sigmoid
// embedding and learned weighted absolute-difference similarity logit.
// Full105px/64-128-128-256/4096 model (38951745 parameters).
// Compact48px/8-16-16-32/64 preserves all stages; kernels5/3/3/2 instead
// of10/7/4/4. Glorot/He initialization, negative initial score weights,
// Adam.001,batch32,L2.0001 and three epochs are explicit adaptations.
// Each epoch samples1024 balanced pairs from MNIST digits0-4 (512/class).
// Validation uses only digits5-6; test uses only7-9 from official test IDX.
// Checkpoint selects128 fixed2-way validation tasks, with pair BCE tiebreak.
// Score512 balanced test pairs and2563-way tasks: one support/class,
// different query row, no fine-tuning. Compare untrained and pixel baselines.
// No Omniglot,20-way, clinical or paper-accuracy claim.
//
// Port of tflow's siamese. MNIST is read from $DATAMUNGE_DATA_DIR/mnist
// (tflow's output/datasets/mnist). Pairs, tasks and batches are drawn with
// tflow's std::mt19937 seeds (common/siamese_data.hpp), so they are TF's rows;
// weights come from datamunge's generator, so trained numbers match TF
// statistically. TF builds the full 105 px model as a graph only; here it
// also runs one forward/backward pass on a 2-pair batch. The checkpoint is
// best.npz; the episode preview is written with nn::save_png. Outputs go to
// output/nn/models/siamese_digits.
// Options: epochs=3 data=PATH checks_only=0|1 accuracy=0|1 threads=N;
// accuracy=1 adds70% unseen verification and one-shot benchmark gates.

#include "common/dataset_assets.hpp"
#include <cmath>
#include <iostream>
#include <set>

#include "common/paths.hpp"
#include "common/siamese.hpp"

// Experiment workflow from examples/nn/common/siamese.hpp.
namespace metric {
using namespace ::metric;
inline void TrainSiamese(const std::filesystem::path& data_dir, int epochs, bool accuracy, const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "two or more epochs required for learning demonstration");
  const auto train = dataset_assets::LoadMnist(data_dir.string(), "train"), test = dataset_assets::LoadMnist(data_dir.string(), "test");
  check::Require(train.images.size(0) == 60000 && test.images.size(0) == 10000, "official MNIST split sizes");
  const auto fitting = SiameseBuckets(train, {0, 1, 2, 3, 4}), validation = SiameseBuckets(train, {5, 6}),
             testing = SiameseBuckets(test, {7, 8, 9});
  const auto validation_pairs = SiamesePairs(validation, 256, 41), test_pairs = SiamesePairs(testing, 512, 43),
             probe = SiamesePairs(fitting, 256, 47);
  const auto validation_tasks = SiameseTasks(validation, 128, 53), test_tasks = SiameseTasks(testing, 256, 59);
  SiameseModel model;
  const auto   initial   = ScoreSiamesePairs(model, train, probe);
  const auto   untrained = ScoreSiameseTasks(model, test, test_tasks);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_loss\tvalidation_pair_loss\tvalidation_2way_accuracy\n";
  double              best = -1, best_loss = INFINITY;
  int                 selected = -1;
  std::vector<double> losses;
  std::cout << "== Siamese: class-disjoint MNIST fit0-4/validation5-6/test7-9;512 images/class; " << model.parameters.count()
            << " shared parameters" << std::endl;
  std::cout << " 3 epochs by default;1024 balanced resampled pairs/epoch,batch32,Adam.001,L2.0001; untrained fitting pair loss="
            << initial.loss << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const auto pairs = SiamesePairs(fitting, 1024, 100 + epoch);
    double     sum   = 0;
    for (std::size_t start = 0; start < pairs.size(); start += 32) {
      const auto   out   = model.Run(MakeSiameseBatch(train, {pairs.begin() + std::ptrdiff_t(start), pairs.begin() + std::ptrdiff_t(start + 32)}), true);
      const double value = out.first;
      check::Require(std::isfinite(value), "finite Siamese training loss");
      sum += value;
    }
    losses.push_back(sum / 32);
    const auto verification = ScoreSiamesePairs(model, train, validation_pairs);
    const auto tasks        = ScoreSiameseTasks(model, train, validation_tasks);
    if (tasks.accuracy > best || (tasks.accuracy == best && verification.loss < best_loss)) {
      best      = tasks.accuracy;
      best_loss = verification.loss;
      selected  = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << losses.back() << '\t' << verification.loss << '\t' << tasks.accuracy << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": training BCE=" << losses.back() << ", unseen-class validation pair BCE=" << verification.loss
              << ", 2-way one-shot=" << tasks.accuracy << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(ScoreSiamesePairs(model, train, validation_pairs).loss, best_loss, 1e-7, "restored validation pair loss");
  check::Near(ScoreSiameseTasks(model, train, validation_tasks).accuracy, best, 1e-12, "restored one-shot validation accuracy");
  const auto learned = ScoreSiamesePairs(model, train, probe), verification = ScoreSiamesePairs(model, test, test_pairs);
  const auto tasks   = ScoreSiameseTasks(model, test, test_tasks);
  check::Require(learned.loss < initial.loss - 1e-4 && losses.back() < losses.front() - 1e-4, "Siamese fitting and epoch losses must improve");
  std::cout << "  selected=" << selected << "; restored validation one-shot=" << best << " and BCE=" << best_loss
            << " verified; fixed fitting BCE " << initial.loss << " -> " << learned.loss << " (training " << seconds << " s)" << std::endl;
  std::cout << " 512 balanced unseen-class test pairs: BCE=" << verification.loss << ", accuracy=" << verification.accuracy
            << ", TP/FP/FN/TN=" << verification.tp << '/' << verification.fp << '/' << verification.fn << '/' << verification.tn
            << "; mean same/different probabilities=" << verification.positive_probability << '/' << verification.negative_probability
            << std::endl;
  std::cout << " 256 class-disjoint 3-way one-shot tasks: trained=" << tasks.accuracy << ", untrained=" << untrained.accuracy
            << ", pixel nearest=" << tasks.pixel_accuracy
            << ", chance=0.333333; exactly one support/class, distinct query, no updates" << std::endl;
  ExportSiameseTasks(test, test_tasks, tasks, directory);
  check::Require(bool(history), "write Siamese history");
  if (accuracy) check::Require(verification.accuracy >= .7 && tasks.accuracy >= .7, "optional unseen-class verification/one-shot accuracy benchmark");
  std::cout << "  checkpoint, history, task scores and query/support preview: " << directory.string()
            << "\n  full short Siamese learning demonstration passed" << std::endl;
}
}  // namespace metric



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace metric;

namespace {

double MaxAbs(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0;
  for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}

void HeadChecks() {
  nn::ParameterStore p(0);
  const float        av[] = {.1f, .8f, -.2f, .4f, -.5f, .7f}, bv[] = {-.4f, .2f, .9f, .4f, .6f, -.3f};
  Tensor a = nn::from_vector(std::vector<float>(av, av + 6), {2, 3}).requires_grad_(),
         b = nn::from_vector(std::vector<float>(bv, bv + 6), {2, 3}).requires_grad_();
  const Tensor t = nn::from_vector(std::vector<float>{1, 0});
  {
    nn::NoGradGuard guard;
    SiameseLogit(p, a, b, 3);
  }
  const std::vector<float> weights = {-.7f, .4f, 1.2f};
  const float              bias    = -.2f;
  p.set("similarity/W", nn::from_vector(weights, {3, 1}));
  p.set("similarity/b", nn::from_vector(std::vector<float>{bias}));
  const Tensor logit = SiameseLogit(p, a, b, 3), reverse = SiameseLogit(p, b, a, 3), identity = SiameseLogit(p, a, a, 3);
  const Tensor loss     = nn::mean(BinaryCrossEntropy(logit, t));
  const auto   gradient = nn::grad(loss, {a, b, p.at("similarity/W"), p.at("similarity/b")});
  const auto   equal    = nn::grad(nn::mean(identity), {a})[0].to_vector<double>();
  Tensor       extreme  = nn::from_vector(std::vector<float>{1000.f, -1000.f, 1000.f, -1000.f, 0.f, 0.f}).requires_grad_();
  const Tensor stable   = BinaryCrossEntropy(extreme, nn::from_vector(std::vector<float>{1.f, 0.f, 0.f, 1.f, 0.f, 1.f}));
  const auto   sg       = nn::grad(nn::sum(stable), {extreme})[0].to_vector<double>();
  const auto   lv = logit.to_vector<double>(), rv = reverse.to_vector<double>(), iv = identity.to_vector<double>(),
             ga = gradient[0].to_vector<double>(), gb = gradient[1].to_vector<double>(), gw = gradient[2].to_vector<double>(),
             gbias = gradient[3].to_vector<double>(), sv = stable.to_vector<double>();
  double host_loss = 0, error = 0, dw[3] = {}, db = 0;
  const double tv[] = {1, 0};
  for (int i = 0; i < 2; ++i) {
    double z = bias;
    for (int j = 0; j < 3; ++j) z += weights[j] * std::abs(double(av[i * 3 + j]) - bv[i * 3 + j]);
    host_loss += (std::max(z, 0.) - z * tv[i] + std::log1p(std::exp(-std::abs(z)))) / 2;
    check::Near(lv[i], z, 2e-7, "weighted L1 logit");
    check::Near(rv[i], z, 2e-7, "swap symmetry");
    check::Near(iv[i], -.2, 1e-7, "identical pair bias");
    const double delta = (1 / (1 + std::exp(-z)) - tv[i]) / 2;
    db += delta;
    for (int j = 0; j < 3; ++j) {
      const double diff = double(av[i * 3 + j]) - bv[i * 3 + j], sign = (diff > 0) - (diff < 0), da = delta * weights[j] * sign;
      error = std::max(error, std::abs(ga[i * 3 + j] - da));
      error = std::max(error, std::abs(gb[i * 3 + j] + da));
      dw[j] += delta * std::abs(diff);
    }
  }
  check::Near(loss.item(), host_loss, 1e-7, "host mean BCE");
  for (int j = 0; j < 3; ++j) error = std::max(error, std::abs(gw[j] - dw[j]));
  error = std::max(error, std::abs(gbias[0] - db));
  for (int i = 0; i < 6; ++i) check::Near(equal[i], 0, 0, "identical input subgradient");
  const double extreme_loss[] = {0, 0, 1000, 1000, std::log(2.), std::log(2.)}, extreme_gradient[] = {0, 0, 1, -1, .5, -.5};
  for (int i = 0; i < 6; ++i) {
    check::Near(sv[i], extreme_loss[i], 1e-7, "extreme BCE stable");
    check::Near(sg[i], extreme_gradient[i], 1e-7, "extreme and zero BCE derivative");
  }
  check::Near(error, 0, 2e-7, "every L1/head/BCE derivative");
  check::Require(p.count() == 4 && p.size() == 2, "same/reverse/identical head reuses exact variables");
  std::cout << "  weighted L1 logits/symmetry/identity; every input/weight/bias BCE derivative error=" << error
            << "; stable +/-1000 and zero logits passed" << std::endl;
}

void ModelChecks() {
  {
    nn::ParameterStore  p(0);
    const SiameseConfig c;
    std::vector<float>  x(2 * 105 * 105), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = .5f + .45f * std::sin(i * .173f), y[i] = .5f + .45f * std::cos(i * .097f);
    const auto a = SiameseEncoder(p, nn::from_vector(x, {2, 105, 105, 1}), c), b = SiameseEncoder(p, nn::from_vector(y, {2, 105, 105, 1}), c);
    const auto score = SiameseLogit(p, a.embedding, b.embedding, c.embedding);
    check::Require(p.count() == 38951745 && p.size() == 12, "full shared independent census");
    const auto g = nn::grad(nn::sum(score), p.parameters());
    check::Require(g.size() == 12, "all full derivatives");
    for (const auto& v : g)
      for (float f : v.to_vector<float>()) check::Require(std::isfinite(f), "finite full twin derivatives");
    std::cout << "  full105px shared Siamese:38951745 parameters/12 derivatives (executed once on a 2-pair batch)" << std::endl;
  }
  nn::ParameterStore p(17);
  const auto         c = SiameseConfig::Compact();
  std::vector<float> xv(2 * 48 * 48), yv(xv.size());
  for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = .5f + .45f * std::sin(i * .173f), yv[i] = .5f + .45f * std::cos(i * .097f);
  const Tensor input = nn::from_vector(xv, {2, 48, 48, 1}), other = nn::from_vector(yv, {2, 48, 48, 1});
  const auto   a = SiameseEncoder(p, input, c), b = SiameseEncoder(p, other, c);
  const Tensor score = SiameseLogit(p, a.embedding, b.embedding, c.embedding), reverse = SiameseLogit(p, b.embedding, a.embedding, c.embedding),
               identity    = SiameseLogit(p, a.embedding, a.embedding, c.embedding);
  const Tensor left_score  = SiameseLogit(p, a.embedding, nn::stop_gradient(b.embedding), c.embedding);
  const Tensor right_score = SiameseLogit(p, nn::stop_gradient(a.embedding), b.embedding, c.embedding);
  nn::GradOptions keep;
  keep.retain_graph = true;
  const auto g  = nn::grad(nn::sum(score), p.parameters(), keep);
  const auto ga = nn::grad(nn::sum(left_score), p.parameters(), keep);
  const auto gb = nn::grad(nn::sum(right_score), p.parameters(), keep);
  check::Require(p.count() == 24337 && p.size() == 12, "compact two twins share single encoder/head variables");
  check::Require(a.embedding.shape() == nn::Shape({2, 64}) && b.embedding.shape() == a.embedding.shape(), "shared embedding shapes");
  for (float v : a.embedding.to_vector<float>()) check::Require(v > 0 && v < 1, "sigmoid embedding finite range");
  const int sides[] = {44, 20, 8, 3}, channels[] = {8, 16, 16, 32};
  for (int i = 0; i < 4; ++i)
    check::Require(a.stages[std::size_t(i)].shape() == nn::Shape({2, sides[i], sides[i], channels[i]}), "compact convolution stage shape");
  const auto sv = score.to_vector<double>(), rv = reverse.to_vector<double>(), iv = identity.to_vector<double>();
  for (int i = 0; i < 2; ++i) {
    check::Near(sv[i], rv[i], 1e-7, "whole CNN swap symmetry");
    check::Near(iv[i], .1, 1e-7, "whole CNN identical scores");
  }
  double low = INFINITY, high = 0, error = 0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    const auto gv = g[i].to_vector<double>();
    double     n  = 0;
    for (double v : gv) n += v * v;
    n = std::sqrt(n);
    check::Require(std::isfinite(n) && n > 0, "every Siamese parameter receives gradient: " + p.names()[i]);
    low = std::min(low, n), high = std::max(high, n);
    if (p.names()[i].rfind("encoder/", 0) == 0) {
      const auto av = ga[i].to_vector<double>(), bv = gb[i].to_vector<double>();
      std::vector<double> sum(av.size());
      for (std::size_t j = 0; j < av.size(); ++j) sum[j] = av[j] + bv[j];
      error = std::max(error, MaxAbs(gv, sum));
    }
  }
  check::Near(error, 0, 2e-6, "shared encoder gradient is sum of twin contributions");
  std::cout << "  compact48px:24337 shared parameters; all stages/12 derivatives passed, norms[" << low << ',' << high
            << "], additive twin-gradient error=" << error << std::endl;
}

void DataChecks() {
  std::vector<float>        images(30 * 784);
  std::vector<std::int64_t> labels(30);
  for (int i = 0; i < 30; ++i) {
    labels[std::size_t(i)] = i / 3;
    for (int j = 0; j < 784; ++j) images[std::size_t(i * 784 + j)] = float((i * 7 + j) % 256) / 255.f;
  }
  nn::ImageDataset d{nn::from_vector(images, {30, 28, 28, 1}), nn::from_vector(labels, {30})};
  const auto fit = SiameseBuckets(d, {0, 1, 2, 3, 4}), val = SiameseBuckets(d, {5, 6}), test = SiameseBuckets(d, {7, 8, 9});
  const auto pairs = SiamesePairs(fit, 100, 11);
  check::Require(pairs == SiamesePairs(fit, 100, 11) && pairs != SiamesePairs(fit, 100, 12), "pair reproducibility and independent seeds");
  int positive = 0;
  for (const auto& p : pairs) {
    const int a = int(labels[p.left]), b = int(labels[p.right]);
    check::Require(p.left != p.right && a < 5 && b < 5 && p.same == float(a == b), "pair identity/class/label alignment");
    positive += int(p.same);
  }
  check::Require(positive == 50, "exact balanced labels");
  const auto             batch = MakeSiameseBatch(d, {pairs[7], pairs[2], pairs[0]});
  const std::vector<int> order = {7, 2, 0};
  const auto             bl = batch.labels.to_vector<float>(), left = batch.left.to_vector<float>(), right = batch.right.to_vector<float>();
  for (int i = 0; i < 3; ++i) {
    const auto& p = pairs[std::size_t(order[std::size_t(i)])];
    check::Near(bl[std::size_t(i)], p.same, 0, "pair label gather");
    for (int j = 0; j < 784; ++j) {
      check::Near(left[std::size_t(i * 784 + j)], float((int(p.left) * 7 + j) % 256) / 255.f, 0, "left pixel gather");
      check::Near(right[std::size_t(i * 784 + j)], float((int(p.right) * 7 + j) % 256) / 255.f, 0, "right pixel gather");
    }
  }
  const auto tasks = SiameseTasks(test, 100, 17);
  check::Require(tasks == SiameseTasks(test, 100, 17) && tasks != SiameseTasks(test, 100, 18), "task reproducibility and independent seeds");
  for (const auto& task : tasks) {
    check::Require(task.support.size() == 3 && task.classes.size() == 3, "exactly one support per unseen class");
    std::set<int> seen;
    for (std::size_t i = 0; i < 3; ++i) {
      const int label = int(labels[task.support[i]]);
      check::Require(task.support[i] != task.query && label == task.classes[i] && label >= 7, "support/query distinct and labels aligned");
      seen.insert(label);
    }
    check::Require(seen.size() == 3 && labels[task.query] == task.classes[std::size_t(task.correct)], "unique support labels and known correct index");
  }
  check::Require(SiameseClasses(fit) == std::vector<int>({0, 1, 2, 3, 4}) && SiameseClasses(val) == std::vector<int>({5, 6}) &&
                     SiameseClasses(test) == std::vector<int>({7, 8, 9}),
                 "strict train/validation/test class disjointness");
  check::Require(BestOneShot({-.4, .8, .1}) == 1 && BestOneShot({.8, .1, -.4}) == 0 && BestOneShot({0, 0, 0}) == 0,
                 "one-shot argmax, support permutation and deterministic tie");
  bool rejected = false;
  try {
    SiamesePairs(fit, 3, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "reject unbalanced odd pair count");
  std::cout << "  class-disjoint pair/task sampling, exact balance, nonidentity positives, reordered pixels/labels, one support/class and "
               "argmax/tie fixtures passed"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks_only = false, accuracy = false;
    std::filesystem::path dir = paths::Dataset("mnist");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        dir = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 accuracy=0|1 threads=N");
    }
    HeadChecks();
    ModelChecks();
    DataChecks();
    if (checks_only) {
      std::cout << "All Siamese fixtures passed (training not requested).\n";
      return 0;
    }
    TrainSiamese(dir, epochs, accuracy, paths::OutputDir("models/siamese_digits"));
    std::cout << "All Siamese checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
