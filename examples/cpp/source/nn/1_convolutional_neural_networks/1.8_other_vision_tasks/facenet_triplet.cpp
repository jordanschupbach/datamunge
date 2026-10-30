// CMD: just run-nn facenet_triplet
// FaceNet-style128D unit embeddings, squared triplet loss and online mining.
// https://arxiv.org/abs/1503.03832
// NN1: eleven ReLU convolutions, two LRN stages, four3x3 stride2 pools,
// two4096-unit p=2 maxout layers, linear128 and L2. Full220pxRGB graph
// has140025664 parameters/all28 derivatives; no full forward allocation.
// Compact28pxgrayscale uses1/8conv widths and128-unit maxouts (108040params)
// preserving every conv/pool/LRN/maxout stage and128D normalization.
// LRN radius2,bias1,alpha1e-4,beta.75 and SAME padding are explicit choices.
// Default3epochs x32P5K8 batches of40 distinct images, Adam.001,L2.0001.
// Fit MNIST0-4(256/class), validate5-6(128/class), test7-9(128/class).
// All ordered positive pairs, closest strict semi-hard negative inside.2
// margin; no hard/easy fallback. Recompute embeddings before each update.
// Checkpoint selects fixed validation-triplet objective. Threshold calibration
// uses only validation pairs. Report unseen verification AUC/accuracy and
// nearest-support3-way retrieval, with untrained baseline and inspected PNG.
// No real-face/LFW accuracy, pretrained or original optimizer/data claim.
//
// Port of tflow's facenet_triplet. MNIST is read from $DATAMUNGE_DATA_DIR/mnist
// (tflow's dataset cache). Buckets, P-by-K batches, pairs and tasks use
// tflow's std::mt19937 draws, so the data rows match TF's; weights come from
// datamunge's generator, so trained numbers match TF statistically.
// TF -> datamunge: TF builds the full 220 px NN1 as a graph only; here its
// census (140025664 / 28 tensors) is computed from the layer shapes, and
// full=1 also allocates and runs it once (about 1.5 GB). Checkpoints are .npz
// (best.npz). Outputs go to output/nn/models/facenet_digits.
// Options: epochs=3 data=PATH checks_only=0|1 accuracy=0|1 full=0|1 threads=N.
// Optional accuracy=1 gates unseen verification and retrieval at.7.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

#include "common/facenet.hpp"

// Experiment workflow from examples/nn/common/facenet.hpp.
namespace metric {
using namespace ::metric;
inline void TrainFaceNet(const std::filesystem::path& data_dir, int epochs, bool accuracy,
                         const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "at least two epochs for short triplet learning check");
  const auto [train, test] = LoadDigits(data_dir);
  const auto train_labels  = Labels(train);
  const auto fitting = SiameseBuckets(train, {0, 1, 2, 3, 4}, 256), probe_buckets = SiameseBuckets(train, {0, 1, 2, 3, 4}, 32),
             validation = SiameseBuckets(train, {5, 6}, 128), testing = SiameseBuckets(test, {7, 8, 9}, 128);
  FaceNetModel model;
  const auto   before              = EmbedDigitPool(model, train, probe_buckets);
  const auto   probe               = FixedMetricTriplets(before.labels, 512, 71);
  const double initial             = ScoreFixedTriplets(before.host, probe).loss;
  const auto   val_before          = EmbedDigitPool(model, train, validation);
  const auto   validation_triplets = FixedMetricTriplets(val_before.labels, 512, 73);
  const auto   validation_pairs = SiamesePairs(validation, 256, 79), test_pairs = SiamesePairs(testing, 512, 83);
  const auto   tasks     = SiameseTasks(testing, 256, 89);
  const auto   untrained = ScoreMetricTasks(EmbedDigitPool(model, test, testing), tasks);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\tmined_loss\ttriplets\tmissing_semihard\tfixed_fitting_loss\tvalidation_loss\n";
  double       best     = INFINITY;
  int          selected = -1;
  std::int64_t total_mined = 0;
  std::cout << "== FaceNet-style NN1: " << model.parameters.count()
            << " parameters,128D unit sphere; fit digits0-4(256/class), validation5-6(128/class), test7-9(128/class)" << std::endl;
  std::cout << " 3 epochs default,32 batches/epoch,P5xK8 unique images,batch40,Adam.001,L2.0001,margin.2; strict closest "
               "semi-hard negatives with no fallback; initial fixed fitting loss="
            << initial << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    double sum = 0;
    int    updates = 0, mined = 0, missing = 0;
    for (int batch = 0; batch < 32; ++batch) {
      const auto       rows = MetricBatchRows(fitting, 8, epoch * 1000 + batch);
      const nn::Tensor x    = SiameseImages(train, rows);
      std::vector<int> labels;
      for (std::size_t row : rows) labels.push_back(train_labels[row]);
      const HostEmbeddings current(model.Encode(x));
      const auto           mining = MineSemiHard(current, labels);
      check::Require(mining.positive_pairs == 280, "all ordered within-class positive pairs");
      missing += mining.without_semihard;
      for (const auto& t : mining.triplets) {
        const double ap = EmbeddingDistance(current, t[0], t[1]), an = EmbeddingDistance(current, t[0], t[2]);
        check::Require(ap < an && an < ap + .2f, "mined negative strictly inside margin and farther than positive");
      }
      if (mining.triplets.empty()) continue;
      sum += model.Train(x, mining.triplets);
      mined += int(mining.triplets.size());
      ++updates;
    }
    check::Require(updates > 0, "at least one nonempty semi-hard update each epoch");
    total_mined += mined;
    const auto   val          = EmbedDigitPool(model, train, validation);
    const double value        = ScoreFixedTriplets(val.host, validation_triplets).loss;
    const double fitting_loss = ScoreFixedTriplets(EmbedDigitPool(model, train, probe_buckets).host, probe).loss;
    if (value < best) {
      best = value, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / updates << '\t' << mined << '\t' << missing << '\t' << fitting_loss << '\t' << value << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": selected-triplet mean=" << sum / updates << ", mined=" << mined
              << ", no eligible negative=" << missing << ", fixed fitting loss=" << fitting_loss
              << ", unseen validation loss=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  const auto validation_pool = EmbedDigitPool(model, train, validation);
  check::Near(ScoreFixedTriplets(validation_pool.host, validation_triplets).loss, best, 1e-8, "restored validation triplet objective");
  const auto learned = ScoreFixedTriplets(EmbedDigitPool(model, train, probe_buckets).host, probe);
  check::Require(std::isfinite(learned.loss) && learned.loss < initial - 1e-4, "fixed fitting triplet objective must improve");
  const auto threshold    = CalibrateMetricThreshold(MetricPairDistances(validation_pool, validation_pairs), validation_pairs);
  const auto test_pool    = EmbedDigitPool(model, test, testing);
  const auto verification = ScoreMetricPairs(MetricPairDistances(test_pool, test_pairs), test_pairs, threshold);
  const auto retrieval    = ScoreMetricTasks(test_pool, tasks);
  std::cout << "  selected=" << selected << "; restored validation loss=" << best << " verified; fixed fitting triplet loss "
            << initial << " -> " << learned.loss << ", fraction satisfying margin=" << learned.margin_fraction
            << ", total mined=" << total_mined << std::endl;
  std::cout << "  threshold=" << threshold << " calibrated solely on256 validation pairs;512 balanced unseen test pairs: accuracy="
            << verification.accuracy << ", AUC=" << verification.auc << ", TP/FP/FN/TN=" << verification.tp << '/'
            << verification.fp << '/' << verification.fn << '/' << verification.tn
            << ", mean same/different squared distances=" << verification.same << '/' << verification.different << std::endl;
  std::cout << " 256 unseen3-way one-support retrieval tasks: trained=" << retrieval.accuracy
            << ", untrained=" << untrained.accuracy << ", chance=0.333333; no test updates or threshold tuning" << std::endl;
  ExportMetricTasks(test, tasks, retrieval, test_pool, directory);
  check::Require(bool(history), "write metric training history");
  if (accuracy)
    check::Require(verification.accuracy >= .7 && retrieval.accuracy >= .7, "optional class-disjoint metric accuracy benchmark");
  std::cout << "  checkpoint/history,128D embedding TSV and retrieval preview: " << directory.string()
            << " (training " << seconds << " s)\n  full short FaceNet-style triplet learning demonstration passed" << std::endl;
}
}  // namespace metric



namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void NormalizationChecks() {
  const std::vector<float> v = {3, 4, 0, -1, 2, 2, 0, 0, 0, .001f, -.002f, .001f};
  std::vector<float>       w(12);
  for (int i = 0; i < 12; ++i) w[i] = .3f + .1f * i;
  Tensor x = nn::from_vector(v, {4, 3}).requires_grad_();
  const Tensor weights = nn::from_vector(w, {4, 3});
  const Tensor normalized = metric::UnitEmbedding(x, 1e-4f);
  const Tensor gradient   = nn::grad(nn::sum(normalized * weights), {x})[0];
  const std::vector<float> m = {1, 3, 4, 2, -2, -1, 5, 5, 0, -2, -1, -3}, g = {1, -2, .5, 3, 4, -1};
  Tensor       maxin  = nn::from_vector(m, {2, 6}).requires_grad_();
  const Tensor maxout = metric::MaxoutPairs(maxin, 3);
  const Tensor maxgrad = nn::grad(nn::sum(maxout * nn::from_vector(g, {2, 3})), {maxin})[0];
  const auto   nv = normalized.to_vector<double>(), gv = gradient.to_vector<double>();
  double       error = 0;
  for (int i = 0; i < 4; ++i) {
    double squared = 0, dot = 0;
    for (int j = 0; j < 3; ++j) squared += double(v[3 * i + j]) * v[3 * i + j];
    const double norm = std::sqrt(std::max(squared, double(1e-4f)));
    for (int j = 0; j < 3; ++j) dot += (v[3 * i + j] / norm) * w[3 * i + j];
    for (int j = 0; j < 3; ++j) {
      const double y = v[3 * i + j] / norm, derivative = (w[3 * i + j] - (squared > 1e-4f ? y * dot : 0)) / norm;
      error = std::max(error, std::abs(nv[3 * i + j] - y));
      error = std::max(error, std::abs(gv[3 * i + j] - derivative));
    }
  }
  check::Near(error, 0, 2e-5, "every normalized value and derivative including zero/floored norm");
  const double expected[] = {0, 1, -2, 0, 0, .5, 1.5, 1.5, 4, 0, -1, 0};
  const auto   mg         = maxgrad.to_vector<double>();
  for (int i = 0; i < 12; ++i) check::Near(mg[i], expected[i], 0, "maxout unique/tied selected derivative");
  const float selected[] = {3, 4, -1, 5, 0, -1};
  const auto  mo         = maxout.to_vector<double>();
  for (int i = 0; i < 6; ++i) check::Near(mo[i], selected[i], 0, "maxout pair values");
  double squared_distance = 0, dot = 0;
  for (int j = 0; j < 3; ++j) {
    const double a = nv[j], b = nv[3 + j];
    squared_distance += (a - b) * (a - b);
    dot += a * b;
  }
  check::Near(squared_distance, 2 - 2 * dot, 2e-7, "unit embedding cosine/distance equivalence");
  std::cout << "  L2 output/every-gradient host error=" << error
            << " including zero/floor; maxout unique/tied Jacobians and sphere distance identity passed" << std::endl;
}

void TripletChecks() {
  const std::vector<float>                v = {0, 0, 1, 0, .5, 0, 2, 0, 0, 1, 0, 1};
  Tensor                                  x = nn::from_vector(v, {6, 2}).requires_grad_();
  const std::vector<metric::TripletIndex> triplets = {{0, 1, 2}, {0, 1, 3}, {4, 5, 0}, {1, 4, 2}};
  const Tensor loss = metric::MeanTriplet(x, triplets), per = metric::TripletHinge(x, triplets);
  const Tensor empty = metric::MeanTriplet(x, {}), boundary = metric::MeanTriplet(x, {{0, 2, 1}}, .75f);
  const auto   grad = nn::grad(loss, {x})[0].to_vector<double>(), emptygrad = nn::grad(empty, {x})[0].to_vector<double>(),
             boundarygrad = nn::grad(boundary, {x})[0].to_vector<double>();
  const auto pv   = per.to_vector<double>();
  double     host = 0, expected[12] = {}, error = 0;
  int        i    = 0;
  for (const auto& t : triplets) {
    double ap = 0, an = 0;
    for (int c = 0; c < 2; ++c) {
      ap += std::pow(double(v[2 * t[0] + c]) - v[2 * t[1] + c], 2);
      an += std::pow(double(v[2 * t[0] + c]) - v[2 * t[2] + c], 2);
    }
    const double value = std::max(0., ap - an + double(.2f));
    check::Near(pv[i++], value, 1e-7, "each squared-distance hinge");
    host += value / triplets.size();
    if (value > 0)
      for (int c = 0; c < 2; ++c) {
        expected[2 * t[0] + c] += 2 * (v[2 * t[2] + c] - v[2 * t[1] + c]) / triplets.size();
        expected[2 * t[1] + c] += 2 * (v[2 * t[1] + c] - v[2 * t[0] + c]) / triplets.size();
        expected[2 * t[2] + c] += 2 * (v[2 * t[0] + c] - v[2 * t[2] + c]) / triplets.size();
      }
  }
  check::Near(loss.item(), host, 1e-7, "mean triplet host value");
  for (int j = 0; j < 12; ++j) {
    error = std::max(error, std::abs(grad[j] - expected[j]));
    check::Near(emptygrad[j], 0, 0, "empty derivative");
    check::Near(boundarygrad[j], 0, 0, "margin-boundary ReLU subgradient");
  }
  check::Near(error, 0, 1e-7, "all active/inactive/repeated-index triplet derivatives");
  check::Near(empty.item(), 0, 0, "empty triplet loss");
  check::Near(boundary.item(), 0, 0, "margin boundary loss");
  std::cout << "  squared-L2 triplet values/every-coordinate Jacobian error=" << error
            << "; inactive/repeated-index/empty/exact-margin cases passed" << std::endl;
}

void MiningChecks() {
  const std::vector<float> values = {0, 1, .5, 1.5, 2, -1, 1.5, 4};
  const std::vector<int>   labels = {0, 0, 1, 1, 2, 2, 3, 3};
  const auto               mined  = metric::MineSemiHard(nn::from_vector(values, {8, 1}), labels, 2.f);
  std::vector<metric::TripletIndex> expected;
  int                               pairs = 0;
  for (int a = 0; a < 8; ++a)
    for (int p = 0; p < 8; ++p)
      if (a != p && labels[a] == labels[p]) {
        ++pairs;
        const double                       ap = std::pow(values[a] - values[p], 2);
        std::vector<std::pair<double, int>> eligible;
        for (int n = 0; n < 8; ++n)
          if (labels[n] != labels[a]) {
            const double an = std::pow(values[a] - values[n], 2);
            if (ap < an && an < ap + 2) eligible.push_back({an, n});
          }
        std::sort(eligible.begin(), eligible.end());
        if (!eligible.empty()) expected.push_back({a, p, eligible.front().second});
      }
  check::Require(mined.triplets == expected && mined.positive_pairs == pairs && mined.without_semihard == pairs - int(expected.size()),
                 "exhaustive sorted semi-hard reference, tie/index/missing accounting");
  check::Require(std::find(mined.triplets.begin(), mined.triplets.end(), metric::TripletIndex{0, 1, 3}) != mined.triplets.end(),
                 "closest eligible negative deterministic tie");
  const Tensor equal = nn::zeros({4, 2});
  check::Require(metric::MineSemiHard(equal, {0, 0, 1, 1}).triplets.empty(), "collapsed equal distances have no strict semi-hard fallback");
  check::Require(metric::MineSemiHard(equal, {0, 0, 0, 0}).triplets.empty() && metric::MineSemiHard(equal, {0, 1, 2, 3}).positive_pairs == 0,
                 "no negative or positive cases");
  const auto exact = metric::MineSemiHard(nn::tensor({0, 1, 2}, {3, 1}), {0, 0, 1}, 3.f);
  check::Require(std::find(exact.triplets.begin(), exact.triplets.end(), metric::TripletIndex{0, 1, 2}) == exact.triplets.end(),
                 "strict upper margin equality excluded");
  const std::vector<metric::VerificationPair> p = {{0, 1, 1}, {0, 2, 1}, {0, 3, 0}, {0, 4, 0}};
  const std::vector<double>                   d = {.1, .2, .8, .9};
  const double threshold = metric::CalibrateMetricThreshold(d, p);
  const auto   score     = metric::ScoreMetricPairs(d, p, threshold);
  check::Near(threshold, .5, 1e-12, "validation-only optimal threshold midpoint");
  check::Near(score.accuracy, 1, 0, "known verification accuracy");
  check::Near(score.auc, 1, 0, "known verification AUC");
  check::Near(metric::ScoreMetricPairs({1, 1, 1, 1}, p, 1).auc, .5, 0, "tied AUC half credit");
  nn::ImageDataset data;
  data.images = nn::zeros({50, 28, 28, 1});
  std::vector<std::int64_t> l(50);
  for (int i = 0; i < 50; ++i) l[i] = i / 10;
  data.labels        = nn::from_vector(l);
  const auto buckets = metric::SiameseBuckets(data, {0, 1, 2, 3, 4}, 10);
  const auto batch   = metric::MetricBatchRows(buckets, 8, 11);
  check::Require(batch == metric::MetricBatchRows(buckets, 8, 11) && batch != metric::MetricBatchRows(buckets, 8, 12),
                 "P-by-K batch reproducibility");
  std::set<std::size_t> unique(batch.begin(), batch.end());
  int                   counts[5] = {};
  std::vector<int>      y;
  for (std::size_t row : batch) {
    ++counts[l[row]];
    y.push_back(int(l[row]));
  }
  check::Require(unique.size() == 40, "40 distinct batch rows");
  for (int c : counts) check::Require(c == 8, "exact P5K8 class balance");
  for (const auto& t : metric::FixedMetricTriplets(y, 100, 7))
    check::Require(t[0] != t[1] && y[t[0]] == y[t[1]] && y[t[0]] != y[t[2]], "fixed probe triplet identities/labels");
  std::cout << "  exhaustive strict semi-hard mining/ties/equalities/no-fallback; P5K8 uniqueness; calibration and AUC fixtures passed"
            << std::endl;
}

void ModelChecks(bool full) {
  {
    int tensors = 0;
    check::Require(metric::FaceNetCensus({}, &tensors) == 140025664 && tensors == 28, "full NN1 independent parameter census");
    if (full) {
      nn::ParameterStore p(3);
      const auto         net = metric::FaceNetNN1(p, nn::uniform({1, 220, 220, 3}), {});
      check::Require(p.count() == 140025664 && p.size() == 28, "full NN1 executed parameter census");
      const auto gradients = nn::grad(nn::sum(net.embedding), p.parameters());
      check::Require(gradients.size() == 28, "all full NN1 parameters connected");
      std::cout << "  full NN1:140025664 parameters,28 connected derivatives; executed on one 220px image" << std::endl;
    } else {
      std::cout << "  full NN1:140025664 parameters,28 tensors from the layer shapes (TF: graph-only; full=1 executes it)" << std::endl;
    }
  }
  nn::ParameterStore p(17);
  std::vector<float> w(256), in(2 * 28 * 28);
  for (int i = 0; i < 256; ++i) w[i] = std::sin(i * .17f);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = .5 + .45 * std::sin(i * .113f);
  const auto net = metric::FaceNetNN1(p, nn::from_vector(in, {2, 28, 28, 1}), metric::FaceNetConfig::Compact());
  const auto gradients = nn::grad(nn::sum(net.embedding * nn::from_vector(w, {2, 128})), p.parameters());
  check::Require(p.count() == 108040 && p.size() == 28 && metric::FaceNetCensus(metric::FaceNetConfig::Compact()) == 108040,
                 "compact NN1 independent parameter census");
  check::Require(net.embedding.shape() == nn::Shape({2, 128}), "128D compact embedding");
  const int sides[] = {14, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2}, channels[] = {8, 8, 24, 24, 48, 48, 32, 32, 32, 32, 32};
  for (int i = 0; i < 11; ++i)
    check::Require(net.stages[i].shape() == nn::Shape({2, sides[i], sides[i], channels[i]}), "all NN1 convolution stage shapes");
  for (int i = 11; i < 13; ++i) check::Require(net.stages[i].shape() == nn::Shape({2, 128}), "two maxout output shapes");
  const auto e = net.embedding.to_vector<double>();
  for (int i = 0; i < 2; ++i) {
    double norm = 0;
    for (int j = 0; j < 128; ++j) {
      check::Require(std::isfinite(e[i * 128 + j]), "finite sphere embedding");
      norm += e[i * 128 + j] * e[i * 128 + j];
    }
    check::Near(norm, 1, 2e-6, "unit sphere compact output");
  }
  double low = INFINITY, high = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double n = nn::norm(gradients[i]).item();
    check::Require(std::isfinite(n) && n > 0, "every compact NN1 gradient nonzero: " + p.names()[i]);
    low = std::min(low, n), high = std::max(high, n);
  }
  std::cout << "  compact NN1:108040 parameters; all13 stages/28 numerical derivatives/unit128D outputs passed; norms[" << low
            << ',' << high << ']' << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks_only = false, accuracy = false, full = false;
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
      else if (arg == "full=0" || arg == "full=1")
        full = arg == "full=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 accuracy=0|1 full=0|1 threads=N");
    }
    NormalizationChecks();
    TripletChecks();
    MiningChecks();
    ModelChecks(full);
    if (checks_only) {
      std::cout << "All FaceNet-style fixtures passed (training not requested).\n";
      return 0;
    }
    metric::TrainFaceNet(dir, epochs, accuracy, paths::OutputDir("models/facenet_digits"));
    std::cout << "All FaceNet-style triplet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
