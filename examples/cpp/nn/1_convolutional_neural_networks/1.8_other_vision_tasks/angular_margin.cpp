// CMD: just run-nn angular_margin
// ArcFace additive angular / CosFace additive cosine margin classification.
// https://arxiv.org/abs/1801.07698 https://arxiv.org/abs/1801.09414
// Normalize feature rows and bias-free classifier columns; modify only the
// training target: cos(theta+m) or cos(theta)-m, then multiply all logits by s.
// ArcFace clips target cosines by1e-6 and uses a monotonic tail c-m*sin(m)
// for c<=cos(pi-m); exact angular identity remains configurable without tail.
// True analytic autodiff is used, including normalization and angle derivative.
// Inference uses plain scaled cosines and accepts no labels.
// Full NN1+128x1000 head graph140153664params/all29 derivatives; compact
// NN1+128x5 normalized head108680params/all29 numerical gradients.
// Default3epochs/model,1280fit/320validation/640test digits0-4,Adam.001,
// batch32,L2.0001; compact choices scale16, ArcFace m.3 / CosFace m.2.
// Restore smallest plain validation CE. Separate unseen verification calibrates
// on digits5-6, tests7-9; report nearest-support3-way retrieval with no updates.
// Export unseen embeddings and retrieval PNG/TSV. No real face/pretrained
// or paper backbone/training/accuracy claims.
//
// Port of tflow's angular_margin. MNIST is read from $DATAMUNGE_DATA_DIR/mnist
// (tflow's dataset cache); rows, pairs, tasks and the batch order are tflow's
// (std::mt19937 / nn::BatchOrder seed 29); weights come from datamunge's
// generator, so trained numbers match TF statistically.
// TF -> datamunge: the full 220 px NN1+1000-way head is graph-only in TF; here
// its census comes from the layer shapes (full=1 also executes it, ~1.5 GB).
// Checkpoints are .npz. Outputs go to output/nn/models/{arcface,cosface}_digits.
// Options: epochs=3 model=arc|cos (default both) data=PATH checks_only=0|1
// accuracy=0|1 full=0|1 threads=N.
// Optional accuracy=1 gates closed-set.85, unseen verification/retrieval.7.

#include <algorithm>
#include <cmath>
#include <iostream>

#include "common/angular_margin.hpp"

// Experiment workflow from examples/nn/common/angular_margin.hpp.
namespace metric {
using namespace ::metric;
inline void TrainAngularMargin(const std::filesystem::path& dir, int epochs, bool cosface, bool accuracy,
                               const std::filesystem::path& models) {
  check::Require(epochs >= 2, "at least two epochs for margin learning demo");
  const auto [train, test] = LoadDigits(dir);
  const auto fitting = SiameseBuckets(train, {0, 1, 2, 3, 4}, 256), combined = SiameseBuckets(train, {0, 1, 2, 3, 4}, 320),
             seen_test = SiameseBuckets(test, {0, 1, 2, 3, 4}, 128), calibration = SiameseBuckets(train, {5, 6}, 128),
             unseen = SiameseBuckets(test, {7, 8, 9}, 128);
  std::vector<std::size_t> fit_rows, val_rows, test_rows;
  for (int c = 0; c < 5; ++c) {
    fit_rows.insert(fit_rows.end(), fitting[c].begin(), fitting[c].end());
    val_rows.insert(val_rows.end(), combined[c].begin() + 256, combined[c].end());
    test_rows.insert(test_rows.end(), seen_test[c].begin(), seen_test[c].end());
  }
  check::Require(fit_rows.size() == 1280 && val_rows.size() == 320 && test_rows.size() == 640, "fitting/validation/test row counts");
  std::set<std::size_t> fit_set(fit_rows.begin(), fit_rows.end());
  for (std::size_t row : val_rows) check::Require(!fit_set.count(row), "fitting/validation image disjointness");
  const auto calibration_pairs = SiamesePairs(calibration, 256, 97), test_pairs = SiamesePairs(unseen, 512, 101);
  const auto tasks = SiameseTasks(unseen, 256, 103);
  AngularModel model(cosface);
  const auto   probe   = MetricBatchRows(fitting, 8, 107);
  const auto   probe_x = SiameseImages(train, probe), probe_y = AngularLabels(train, probe);
  const double initial = model.Loss(probe_x, probe_y);
  const auto   untrained = ScoreMetricTasks(EmbedDigitPool(model, test, unseen), tasks);
  const std::string name      = cosface ? "cosface" : "arcface";
  const auto        directory = models / (name + "_digits");
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_margin_ce\tvalidation_plain_ce\tvalidation_accuracy\n";
  double              best     = INFINITY;
  int                 selected = -1;
  std::vector<double> losses;
  nn::BatchOrder      order(fit_rows.size(), 32, 29);
  std::cout << "== " << name << ":" << model.parameters.count() << " parameters, NN1 compact+normalized128x5 bias-free head; scale16,margin"
            << (cosface ? .2 : .3) << ",3 epochs default,1280fit/320validation/640test closed-set digits0-4" << std::endl;
  std::cout << "  Adam.001,batch32,L2.0001; unseen calibration5-6/test7-9,128 images/class; initial fixed fitting margin CE="
            << initial << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      std::vector<std::size_t> rows;
      for (auto id : ids) rows.push_back(fit_rows[std::size_t(id)]);
      sum += model.Loss(SiameseImages(train, rows), AngularLabels(train, rows), true) * double(rows.size());
      seen += rows.size();
    }
    check::Require(seen == fit_rows.size(), "all fitting rows used each epoch");
    losses.push_back(sum / double(seen));
    const auto validation = ScoreAngularClasses(model, train, val_rows);
    if (validation.loss < best) {
      best = validation.loss, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << losses.back() << '\t' << validation.loss << '\t' << validation.accuracy << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": training margin CE=" << losses.back() << ", validation plain CE=" << validation.loss
              << ", accuracy=" << validation.accuracy << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(ScoreAngularClasses(model, train, val_rows).loss, best, 1e-8, "restored label-free validation CE");
  const double learned = model.Loss(probe_x, probe_y);
  check::Require(learned < initial - 1e-4 && losses.back() < losses.front() - 1e-4, "fixed fitting and mean epoch margin objectives improve");
  const auto   classification = ScoreAngularClasses(model, test, test_rows);
  const auto   cal            = EmbedDigitPool(model, train, calibration);
  const double threshold      = CalibrateMetricThreshold(MetricPairDistances(cal, calibration_pairs), calibration_pairs);
  const auto   pool           = EmbedDigitPool(model, test, unseen);
  const auto   verification   = ScoreMetricPairs(MetricPairDistances(pool, test_pairs), test_pairs, threshold);
  const auto   retrieval      = ScoreMetricTasks(pool, tasks);
  std::cout << "  selected=" << selected << "; restored validation CE=" << best << " verified; fixed fitting margin CE " << initial
            << " -> " << learned << std::endl;
  std::cout << "  all640 held-out closed-set images, no labels supplied to inference: CE=" << classification.loss
            << ", accuracy=" << classification.accuracy << "; confusion truth rows:" << std::endl;
  for (int y = 0; y < 5; ++y) {
    std::cout << "   ";
    for (int p = 0; p < 5; ++p) std::cout << classification.confusion[y][p] << ' ';
    std::cout << std::endl;
  }
  std::cout << "  unseen512 balanced pairs: validation-only threshold=" << threshold << ", accuracy=" << verification.accuracy
            << ", AUC=" << verification.auc << ", TP/FP/FN/TN=" << verification.tp << '/' << verification.fp << '/'
            << verification.fn << '/' << verification.tn << ", mean same/different distance=" << verification.same << '/'
            << verification.different << std::endl;
  std::cout << "  unseen2563-way one-support retrieval tasks: trained=" << retrieval.accuracy << ", untrained=" << untrained.accuracy
            << ", chance=0.333333; no unseen updates" << std::endl;
  ExportMetricTasks(test, tasks, retrieval, pool, directory);
  check::Require(bool(history), "write angular history");
  if (accuracy)
    check::Require(classification.accuracy >= .85 && verification.accuracy >= .7 && retrieval.accuracy >= .7,
                   "optional closed-set/unseen margin accuracy benchmark");
  std::cout << "  checkpoint/history,384 unseen128D embeddings and retrieval scores/preview: " << directory.string()
            << " (training " << seconds << " s)\n  full short " << name << " demonstration passed" << std::endl;
}
}  // namespace metric



namespace nn = datamunge::nn;
using nn::Tensor;
using metric::AngularMarginConfig;
using metric::AngularMarginKind;

namespace {

std::pair<double, double> HostPhi(double value, const AngularMarginConfig& cfg) {
  if (cfg.margin == 0) return {value, 1};
  if (cfg.kind == AngularMarginKind::kCosFace) return {value - cfg.margin, 1};
  const double low = -1.f + cfg.epsilon, high = 1.f - cfg.epsilon, c = std::clamp(value, low, high);
  const double clip = value > low && value < high ? 1 : 0;
  if (cfg.monotonic && c <= std::cos(3.14159265358979323846f - cfg.margin)) return {c - std::sin(cfg.margin) * cfg.margin, clip};
  const double sine = std::sqrt(std::max(1 - c * c, double(cfg.epsilon)));
  return {c * std::cos(cfg.margin) - sine * std::sin(cfg.margin), (std::cos(cfg.margin) + c / sine * std::sin(cfg.margin)) * clip};
}

double HostCE(const std::vector<double>& logits, int target, std::vector<double>* gradient = nullptr) {
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double       sum     = 0;
  for (double value : logits) sum += std::exp(value - maximum);
  if (gradient) {
    gradient->clear();
    for (std::size_t i = 0; i < logits.size(); ++i) gradient->push_back(std::exp(logits[i] - maximum) / sum - (int(i) == target));
  }
  return maximum + std::log(sum) - logits[target];
}

void FormulaChecks() {
  for (bool cosface : {false, true})
    for (bool monotonic : {false, true}) {
      AngularMarginConfig cfg;
      cfg.kind      = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
      cfg.margin    = cosface ? .35f : .5f;
      cfg.scale     = 7;
      cfg.monotonic = monotonic;
      const float target[] = {.8f, .1f, -.95f, -1, 1, 0, std::cos(3.14159265358979323846f - .5f), -.6f};
      std::vector<float>        values(24);
      std::vector<std::int64_t> labels(8);
      for (int i = 0; i < 8; ++i) {
        labels[i] = i % 3;
        for (int j = 0; j < 3; ++j) values[i * 3 + j] = j == i % 3 ? target[i] : j == 0 ? -.2f : .3f;
      }
      Tensor       cosine = nn::from_vector(values, {8, 3}).requires_grad_();
      const Tensor y      = nn::from_vector(labels);
      const Tensor logits = metric::ApplyAngularMargin(cosine, y, 3, cfg);
      const Tensor loss   = metric::AngularCrossEntropy(logits, y, 3);
      const auto   gradient = nn::grad(loss, {cosine})[0].to_vector<double>();
      auto         zero     = cfg;
      zero.margin           = 0;
      const auto plain = metric::ApplyAngularMargin(cosine, y, 3, zero).to_vector<double>();
      const auto out0  = logits.to_vector<double>();
      double     host = 0, error = 0;
      for (int i = 0; i < 8; ++i) {
        const int           label = int(labels[i]);
        std::vector<double> expected;
        for (int j = 0; j < 3; ++j) {
          const double raw = values[i * 3 + j];
          expected.push_back(cfg.scale * (j == label ? HostPhi(raw, cfg).first : raw));
          check::Near(out0[i * 3 + j], expected.back(), 2e-6, "target-only angular/cosine transformed value");
          check::Near(plain[i * 3 + j], cfg.scale * raw, 2e-7, "zero margin normalized-softmax identity");
        }
        std::vector<double> d;
        host += HostCE(expected, label, &d) / 8;
        for (int j = 0; j < 3; ++j) {
          const double derivative = cfg.scale * d[j] * (j == label ? HostPhi(values[i * 3 + j], cfg).second : 1) / 8;
          error = std::max(error, std::abs(gradient[i * 3 + j] - derivative));
        }
      }
      check::Near(loss.item(), host, 2e-6, "host stabilized margin CE");
      check::Near(error, 0, 2e-5, "every margin/CE gradient including tail and endpoint clips");
      std::cout << "  " << (cosface ? "CosFace" : "ArcFace") << " monotonic=" << monotonic << ": host CE/every-logit derivative error="
                << error << "; target-only,zero margin,endpoints passed" << std::endl;
    }
  std::vector<float> grid(1001 * 2);
  for (int i = 0; i <= 1000; ++i) grid[i * 2] = -1.f + 2.f * i / 1000, grid[i * 2 + 1] = .17f;
  AngularMarginConfig cfg;
  cfg.scale = 1;
  const auto monotone =
      metric::ApplyAngularMargin(nn::from_vector(grid, {1001, 2}), nn::zeros({1001}, nn::DType::Int64), 2, cfg).to_vector<float>();
  for (int i = 0; i <= 1000; ++i) {
    check::Require(monotone[i * 2] <= grid[i * 2] + 1e-6, "positive angular margin never boosts target logit");
    if (i) check::Require(monotone[i * 2] >= monotone[(i - 1) * 2] - 1e-6, "tail extension preserves monotonic target curve");
    check::Near(monotone[i * 2 + 1], .17f, 0, "all non-target logits unchanged");
  }
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto bad = cfg;
    if (invalid == 0) bad.margin = -.1f;
    if (invalid == 1) bad.margin = 2;
    if (invalid == 2) bad.scale = 0;
    if (invalid == 3) bad.epsilon = 0;
    if (invalid == 4) bad.margin = NAN;
    check::Throws([&] { bad.Validate(); }, "reject invalid margin configuration");
  }
  std::cout << "  1001-point monotonic tail/target penalty grid and invalid-configuration checks passed" << std::endl;
}

void NormalizedHeadChecks() {
  for (bool cosface : {false, true}) {
    nn::ParameterStore  p(17);
    AngularMarginConfig cfg;
    cfg.kind   = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = cosface ? .2f : .3f;
    cfg.scale  = 5;
    const std::vector<float> xv = {.3f, .7f, -.4f, -.8f, .2f, .5f, .1f, -.6f, .9f},
                             wv = {.4f, -.3f, .8f, .1f, -.2f, .6f, .1f, -.5f, .7f, .2f, -.4f, .9f};
    const Tensor target = nn::tensor({2, 0, 3}).to(nn::DType::Int64);
    const std::int64_t tv[] = {2, 0, 3};
    // The weight is created by the head, then overwritten with the controlled values.
    {
      nn::NoGradGuard guard;
      metric::NormalizedAngularHead(p, nn::zeros({3, 3}), target, 3, 4, cfg);
      p.set("angular/W", nn::from_vector(wv, {3, 4}));
    }
    Tensor     raw  = nn::from_vector(xv, {3, 3}).requires_grad_();
    const auto head = metric::NormalizedAngularHead(p, raw, target, 3, 4, cfg);
    const Tensor loss = metric::AngularCrossEntropy(head.training, target, 4);
    const auto   gradient = nn::grad(loss, {raw, p.at("angular/W")});
    const auto   cos_out = head.cosine.to_vector<double>(), emb = head.embedding.to_vector<double>(),
               wout = head.weights.to_vector<double>(), gx = gradient[0].to_vector<double>(), gw = gradient[1].to_vector<double>();
    double u[3][3] = {}, v[3][4] = {}, xn[3] = {}, wn[4] = {}, du[3][3] = {}, dv[3][4] = {}, dx[3][3] = {}, dw[3][4] = {}, host = 0,
           error = 0;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) xn[i] += double(xv[3 * i + k]) * xv[3 * i + k];
      xn[i] = std::sqrt(xn[i]);
      for (int k = 0; k < 3; ++k) u[i][k] = xv[3 * i + k] / xn[i];
    }
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; ++k) wn[j] += double(wv[4 * k + j]) * wv[4 * k + j];
      wn[j] = std::sqrt(wn[j]);
      for (int k = 0; k < 3; ++k) v[k][j] = wv[4 * k + j] / wn[j];
    }
    for (int i = 0; i < 3; ++i) {
      std::vector<double> logits, c;
      for (int j = 0; j < 4; ++j) {
        double value = 0;
        for (int k = 0; k < 3; ++k) value += u[i][k] * v[k][j];
        c.push_back(value);
        check::Near(cos_out[i * 4 + j], value, 2e-7, "feature-row/class-column normalization cosine");
        logits.push_back(cfg.scale * (j == tv[i] ? HostPhi(value, cfg).first : value));
      }
      std::vector<double> d;
      host += HostCE(logits, int(tv[i]), &d) / 3;
      for (int j = 0; j < 4; ++j) {
        const double dc = d[j] * cfg.scale * (j == tv[i] ? HostPhi(c[j], cfg).second : 1) / 3;
        for (int k = 0; k < 3; ++k) {
          du[i][k] += dc * v[k][j];
          dv[k][j] += dc * u[i][k];
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) dot += u[i][k] * du[i][k];
      for (int k = 0; k < 3; ++k) {
        dx[i][k] = (du[i][k] - u[i][k] * dot) / xn[i];
        error    = std::max(error, std::abs(gx[i * 3 + k] - dx[i][k]));
        check::Near(emb[i * 3 + k], u[i][k], 1e-7, "unit feature values");
      }
    }
    for (int j = 0; j < 4; ++j) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) dot += v[k][j] * dv[k][j];
      for (int k = 0; k < 3; ++k) {
        dw[k][j] = (dv[k][j] - v[k][j] * dot) / wn[j];
        error    = std::max(error, std::abs(gw[k * 4 + j] - dw[k][j]));
        check::Near(wout[k * 4 + j], v[k][j], 1e-7, "unit class-column values");
      }
    }
    check::Near(loss.item(), host, 1e-6, "normalized head host CE");
    // Keep the nearly aligned target fixture. Check all coordinates by a
    // separate central difference as well as the double-precision formula.
    auto evaluate = [&](const std::vector<float>& features, const std::vector<float>& weights) {
      nn::NoGradGuard guard;
      p.set("angular/W", nn::from_vector(weights, {3, 4}));
      const auto h = metric::NormalizedAngularHead(p, nn::from_vector(features, {3, 3}), target, 3, 4, cfg);
      return metric::AngularCrossEntropy(h.training, target, 4).item();
    };
    double      difference_error = 0;
    const float h                = .002f;
    for (int coordinate = 0; coordinate < 21; ++coordinate) {
      auto   features = xv, weights = wv;
      float& value = coordinate < 9 ? features[coordinate] : weights[coordinate - 9];
      const float base = value;
      value            = base + h;
      const double plus = evaluate(features, weights);
      value             = base - h;
      const double minus    = evaluate(features, weights);
      const double expected = coordinate < 9 ? dx[coordinate / 3][coordinate % 3] : dw[(coordinate - 9) / 4][(coordinate - 9) % 4];
      difference_error      = std::max(difference_error, std::abs((plus - minus) / (2 * h) - expected));
    }
    check::Near(error, 0, 1e-5, "all raw feature and class-weight derivatives, float32 near-collinear case");
    check::Near(difference_error, 0, 1e-3, "independent central differences for every feature and weight");
    // Positive feature/column scaling must disappear under normalization.
    std::vector<float> scaled_x(9), scaled_w(12);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) scaled_x[3 * i + k] = xv[3 * i + k] * (i + 2);
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 4; ++j) scaled_w[4 * k + j] = wv[4 * k + j] * (j + 2);
    std::vector<double> original, scaled;
    {
      nn::NoGradGuard guard;
      p.set("angular/W", nn::from_vector(wv, {3, 4}));
      original = metric::NormalizedAngularHead(p, nn::from_vector(xv, {3, 3}), Tensor(), 3, 4, cfg).inference.to_vector<double>();
      p.set("angular/W", nn::from_vector(scaled_w, {3, 4}));
      scaled = metric::NormalizedAngularHead(p, nn::from_vector(scaled_x, {3, 3}), Tensor(), 3, 4, cfg).inference.to_vector<double>();
    }
    for (int i = 0; i < 12; ++i) check::Near(scaled[i], original[i], 1e-6, "independent row/column positive scale invariance");
    check::Require(p.count() == 12 && p.size() == 1, "bias-free normalized head has one shared weight matrix");
    std::cout << "  normalized " << (cosface ? "CosFace" : "ArcFace") << ": all feature/column-weight Jacobian error=" << error
              << ", central difference error=" << difference_error
              << "; label-free inference and independent scale invariance passed" << std::endl;
  }
}

void GeometryChecks() {
  for (bool cosface : {false, true}) {
    const double angle = 1, margin = cosface ? .2 : .3,
                 boundary = cosface ? angle / 2 - std::asin(margin / (2 * std::sin(angle / 2))) : (angle - margin) / 2;
    std::vector<float> c(6);
    for (int i = 0; i < 3; ++i) {
      const double theta = boundary + (i - 1) * .02;
      c[i * 2]           = float(std::cos(theta));
      c[i * 2 + 1]       = float(std::cos(angle - theta));
    }
    AngularMarginConfig cfg;
    cfg.kind   = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = float(margin);
    cfg.scale  = 1;
    const auto out = metric::ApplyAngularMargin(nn::from_vector(c, {3, 2}), nn::zeros({3}, nn::DType::Int64), 2, cfg).to_vector<float>();
    check::Require(out[0] > out[1] && out[4] < out[5], "margin boundary sides");
    check::Near(out[2], out[3], 2e-7, "analytic two-class boundary");
    check::Require(boundary < angle / 2, "margin shrinks target training region");
  }
  std::cout << "  two-class angular/cosine boundaries and opposite decision sides passed" << std::endl;
}

void ModelChecks(bool full) {
  for (bool compact : {false, true})
    for (bool cosface : {false, true}) {
      const auto          cfg     = compact ? metric::FaceNetConfig::Compact() : metric::FaceNetConfig{};
      const int           classes = compact ? 5 : 1000;
      AngularMarginConfig margin;
      margin.kind   = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
      margin.margin = cosface ? .2f : .3f;
      margin.scale  = 16;
      int tensors   = 0;
      const std::int64_t census = metric::FaceNetCensus(cfg, &tensors) + 128 * classes;
      check::Require(census == (compact ? 108680 : 140153664) && tensors + 1 == 29, "independent encoder+head census");
      if (!compact && !full) {
        std::cout << "  full " << (cosface ? "CosFace" : "ArcFace")
                  << ":NN1+128x1000 head,140153664 parameters/29 tensors from the layer shapes (TF: graph-only; full=1 executes it)"
                  << std::endl;
        continue;
      }
      nn::ParameterStore p(17);
      std::vector<float> in(std::size_t(2) * cfg.side * cfg.side * cfg.input);
      for (std::size_t i = 0; i < in.size(); ++i) in[i] = .5 + .45 * std::sin(i * .113f);
      const Tensor label = nn::tensor({0, 3}).to(nn::DType::Int64);
      const auto   net   = metric::FaceNetNN1(p, nn::from_vector(in, {2, cfg.side, cfg.side, cfg.input}), cfg);
      const auto   head  = metric::NormalizedAngularHead(p, net.raw, label, 128, classes, margin);
      const Tensor loss  = metric::AngularCrossEntropy(head.training, label, classes);
      const auto   gradient = nn::grad(loss, p.parameters());
      check::Require(p.count() == census && gradient.size() == 29, "independent encoder+head census and gradient connectivity");
      check::Require(std::isfinite(loss.item()) && head.inference.shape() == nn::Shape({2, classes}), "finite compact loss/classifier shape");
      double low = INFINITY, high = 0;
      for (std::size_t i = 0; i < gradient.size(); ++i) {
        const double n = nn::norm(gradient[i]).item();
        check::Require(std::isfinite(n) && n > 0, "every CNN/head gradient finite nonzero: " + p.names()[i]);
        low = std::min(low, n), high = std::max(high, n);
      }
      std::cout << "  " << (compact ? "compact " : "full ") << (cosface ? "CosFace" : "ArcFace") << ":" << census
                << " parameters/all29 numerical derivatives, norms[" << low << ',' << high << ']' << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks_only = false, accuracy = false, full = false;
    std::filesystem::path dir    = paths::Dataset("mnist");
    std::vector<bool>     models = {false, true};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        dir = arg.substr(5);
      else if (arg == "model=arc")
        models = {false};
      else if (arg == "model=cos")
        models = {true};
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg == "full=0" || arg == "full=1")
        full = arg == "full=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N model=arc|cos data=PATH checks_only=0|1 accuracy=0|1 full=0|1 threads=N");
    }
    FormulaChecks();
    NormalizedHeadChecks();
    GeometryChecks();
    ModelChecks(full);
    if (checks_only) {
      std::cout << "All angular margin fixtures passed (training not requested).\n";
      return 0;
    }
    for (bool cosface : models) metric::TrainAngularMargin(dir, epochs, cosface, accuracy, paths::OutputDir("models"));
    std::cout << "All ArcFace/CosFace checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
