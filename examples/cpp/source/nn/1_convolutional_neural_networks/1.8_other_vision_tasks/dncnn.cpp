// CMD: just run-nn dncnn
// DnCNN additive Gaussian denoising with residual noise and moving batch norm.
// https://arxiv.org/abs/1608.03981
// https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py
// Seventeen SAME3x3 convolutions:Conv/ReLU,15 Conv/BN/ReLU,linear noise Conv.
// Subtract signed predicted noise from noisy input;no output activation/clamp.
// Author-style orthogonal kernels,first bias only,BN eps1e-4/old-retention.05.
// Full64-wide graph census/48 derivatives;compact16 retains all17layers.
// Default3epochs,256fit/32validation/32test natural CIFAR grayscale32px,
// sigma25/255 AWGN,batch8,Adam.001,fresh unbounded training noise each epoch.
// Residual half-MSE averaged over pixels;select moving-BN validation minimum.
// Restore weights AND BN into independent noisy-only inference graph,verify
// parity,immutable state and single/batch behavior. Report raw/clipped PSNR
// against noisy input and reflected3x3 mean filter;inspect saved preview.
// No Train400/BSD68/pretrained or published denoising-quality claim.
//
// Port of tflow's dncnn. CIFAR-10 binaries are read from
// $DATAMUNGE_DATA_DIR/cifar10 (data_batch_1.bin, test_batch.bin). Clean,
// noisy and noise tensors and the batch order are bit-identical to TF's
// (tflow's mt19937_64 Box-Muller noise, nn::BatchOrder seed 59); the
// orthogonal kernels come from datamunge's generator (TF: RandomNormal seed
// 410+layer + QR), so trained numbers match TF statistically. TF builds the
// full 64-wide model as a graph only; here it also runs forward/backward on
// the 2x19x21 fixture. Checkpoint best.npz holds weights and moving BN state.
// Outputs go to output/nn/models/dncnn.
// Options: epochs=3 data=PATH checks_only=0|1 accuracy=0|1 threads=N.
// Optional accuracy requires raw PSNR>.1dB above BOTH baselines.

#include <cmath>
#include <iostream>

#include "common/denoising.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/denoising.hpp.
namespace denoise {
using namespace ::denoise;
inline void TrainDenoising(const std::filesystem::path& root, int epochs, bool accuracy, const std::filesystem::path& dir) {
  check::Require(epochs >= 2, "at least two epochs for denoising learning check");
  constexpr double sigma = 25. / 255;
  const auto train = data::LoadCifar({root / "data_batch_1.bin"}, data::Cifar::k10),
             testing = data::LoadCifar({root / "test_batch.bin"}, data::Cifar::k10);
  check::Require(train.size() == 10000 && testing.size() == 10000, "original CIFAR denoising splits");
  const auto fit   = DnImages(train.images, Range(0, 256));
  const auto fixed = DnCorrupt(fit, sigma, 41), val = DnCorrupt(DnImages(train.images, Range(256, 288)), sigma, 43),
             test = DnCorrupt(DnImages(testing.images, Range(128, 160)), sigma, 47);
  const auto   probe = DnBatch(fixed, Range(0, 8));
  DnModel      model;
  const auto   initial_state = model.Snapshot();
  const double initial_train = model.Loss(probe, true), initial_infer = model.Loss(probe);
  std::filesystem::create_directories(dir);
  const auto    prefix = (dir / "best.npz").string();
  std::ofstream history(dir / "history.tsv");
  history << "epoch\ttraining_half_mse\tvalidation_half_mse\n";
  double              best     = INFINITY;
  int                 selected = -1;
  std::vector<double> losses;
  nn::BatchOrder      order(256, 8, 59);
  std::cout << "DnCNN17x16: " << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors,"
            << model.state.count() << " moving BN values;" << epochs << " epochs,batch8,Adam.001\n"
            << " sigma25/255 AWGN,no noisy-input clamp,fresh training corruption each epoch;fixed val/test seeds43/47\n"
            << " fit rows0..255,val256..287 first train batch;official-test128..159;original32px,no augmentation/decay\n"
            << " fixed fitting initial half-MSE train-mode=" << initial_train << ",moving-BN inference=" << initial_infer << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const auto fresh = DnCorrupt(fit, sigma, 5300 + std::uint64_t(epoch));
    order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      std::vector<std::size_t> rows(ids.begin(), ids.end());
      sum += double(rows.size()) * model.Loss(DnBatch(fresh, rows), true, true);
      seen += rows.size();
    }
    check::Require(seen == 256, "all clean images used with fresh noise per epoch");
    losses.push_back(sum / double(seen));
    const double value = DnLoss(model, val);
    if (value < best) {
      best = value, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << losses.back() << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ": training half-MSE=" << losses.back() << ",inference validation=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(DnLoss(model, val), best, 1e-8, "restored denoising validation agreement");
  const double learned_train = model.Loss(probe, true), learned_infer = model.Loss(probe);
  check::Require(learned_train < initial_train * (1 - 1e-4) && losses.back() < losses.front() * (1 - 1e-4),
                 "fixed train-mode and epoch-average noise learning");
  const auto trained_state = model.Snapshot();
  bool       changed       = false;
  for (std::size_t i = model.parameters.size(); i < trained_state.size(); ++i) changed |= trained_state[i] != initial_state[i];
  check::Require(changed, "training updates moving BN state");
  DnModel deployed(false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer && deployed.state.size() == 30 && deployed.parameters.size() == 48,
                 "deployment: no optimizer, 15 inference BN layers (30 moving tensors), noisy image is the only input");
  const auto before     = deployed.Snapshot();
  const auto prediction = DnPredict(deployed, test.noisy), reference = DnPredict(model, test.noisy),
             repeat     = DnPredict(deployed, test.noisy);
  const auto pv = prediction.to_vector<float>(), rv = reference.to_vector<float>(), qv = repeat.to_vector<float>();
  double     error = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    error = std::max(error, std::abs(double(pv[i]) - rv[i]));
    check::Near(qv[i], pv[i], 0, "deterministic repeated denoising inference");
  }
  check::Near(error, 0, 1e-7, "standalone restored denoising output parity");
  const auto individual = deployed.Predict(DnBatch(test.noisy, {3})).to_vector<float>();
  for (int i = 0; i < 1024; ++i) check::Near(individual[i], pv[3 * 1024 + std::size_t(i)], 1e-6, "single versus batched denoising");
  check::Require(before == deployed.Snapshot(), "inference never changes weights or moving BN state");
  const auto box    = DnBoxFilter(test.noisy);
  const auto result = DnScore(prediction, test.clean), baseline = DnScore(test.noisy, test.clean), filtered = DnScore(box, test.clean);
  const auto heldout_snapshot = model.Snapshot();
  check::Near(2 * DnLoss(model, test), result.mse, 1e-8, "residual loss equals clean-image MSE");
  check::Require(heldout_snapshot == model.Snapshot(), "held-out scoring changes no BN state");
  auto print = [&](const char* name, const DnMetrics& m) {
    std::cout << " " << name << ": raw MSE=" << m.mse << ",mean raw PSNR=" << m.psnr << "dB,clipped-display PSNR=" << m.clipped_psnr
              << "dB,outside[0,1]=" << 100 * m.outside << "%\n";
  };
  std::cout << " selected=" << selected << ",restored validation=" << best << " verified;fixed fitting train-mode " << initial_train
            << " -> " << learned_train << ",inference " << initial_infer << " -> " << learned_infer << "\n";
  print("learned DnCNN", result);
  print("unprocessed noisy", baseline);
  print("3x3 reflected box", filtered);
  std::cout << " standalone parity error=" << error << ",all weights/BN immutable,single/batch agreement verified\n";
  DnPreview(test, prediction, box, dir / "comparison.png");
  nn::save_npz((dir / "test_outputs.npz").string(), {{"clean", test.clean},
                                                       {"noisy", test.noisy},
                                                       {"true_noise", test.noise},
                                                       {"prediction", prediction},
                                                       {"box_baseline", box}});
  check::Require(bool(history), "write denoising history");
  if (accuracy) check::Require(result.psnr > std::max(baseline.psnr, filtered.psnr) + .1, "optional .1dB raw PSNR gain over noisy and box baselines");
  std::cout << " checkpoint/BN/history/all32 tensors/noisy-box-learned-clean-4xerror preview: " << dir.string() << " (training "
            << seconds << " s)\nComplete short DnCNN denoising demonstration passed.\n";
}
}  // namespace denoise



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace denoise;

namespace {

void NoiseAndMetricChecks() {
  check::Near(DnGray(1, 0, 0), .299, 0, "full-range encoded-RGB grayscale red");
  check::Near(DnGray(1, 1, 1), 1, 1e-15, "grayscale white");
  std::vector<std::int32_t> raw(4 * 3072);
  for (int i = 0; i < int(raw.size()); ++i) raw[i] = (i * 13 + i / 3072 * 41) % 256;
  const Tensor source = nn::from_vector(raw, {4, 32, 32, 3});
  const Tensor all = DnImages(source, {0, 1, 2, 3}), reordered = DnImages(source, {3, 1});
  const auto   av = all.to_vector<float>(), rv = reordered.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 1024; ++i) check::Near(rv[n * 1024 + i], av[(n ? 1 : 3) * 1024 + i], 0, "clean grayscale reorder");
  const auto noisy = DnCorrupt(all, 25. / 255, 41), repeat = DnCorrupt(all, 25. / 255, 41), fresh = DnCorrupt(all, 25. / 255, 42),
             zero = DnCorrupt(all, 0, 41), pair = DnBatch(noisy, {3, 1});
  const auto nv = noisy.noisy.to_vector<float>(), ev = noisy.noise.to_vector<float>();
  check::Require(nv == repeat.noisy.to_vector<float>() && nv != fresh.noisy.to_vector<float>(),
                 "fixed seeds reproduce and fresh seeds change Gaussian noise");
  check::Require(zero.noisy.to_vector<float>() == av, "sigma0 exact identity");
  for (std::size_t i = 0; i < av.size(); ++i) check::Near(nv[i] - av[i], ev[i], 0, "actual signed additive residual target");
  const auto pn = pair.noisy.to_vector<float>(), pe = pair.noise.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 1024; ++i) {
      const int j = (n ? 1 : 3) * 1024 + i;
      check::Near(pn[n * 1024 + i], nv[j], 0, "reordered noisy rows");
      check::Near(pe[n * 1024 + i], ev[j], 0, "noise target remains paired");
    }
  std::vector<float> large(128 * 1024);
  for (std::size_t i = 0; i < large.size(); ++i) large[i] = float(i % 2);
  const auto sample = DnCorrupt(nn::from_vector(large, {128, 32, 32, 1}), 25. / 255, 101);
  const auto sn = sample.noise.to_vector<float>(), sy = sample.noisy.to_vector<float>();
  double     mean = 0, second = 0, fourth = 0;
  int        outside = 0;
  for (std::size_t i = 0; i < large.size(); ++i) {
    const double z = sn[i] / (25. / 255);
    mean += z, second += z * z, fourth += z * z * z * z;
    outside += sy[i] < 0 || sy[i] > 1;
  }
  mean /= double(large.size()), second /= double(large.size()), fourth /= double(large.size());
  check::Near(mean, 0, .012, "Gaussian empirical zero mean");
  check::Near(second, 1, .02, "Gaussian unit variance");
  check::Near(fourth, 3, .12, "Gaussian fourth moment");
  check::Require(outside > 50000, "no input clipping at black/white boundaries");
  for (double sigma : {-1., double(NAN), double(INFINITY)}) check::Throws([&] { DnCorrupt(all, sigma, 1); }, "reject invalid sigma");
  check::Near(DnPSNR(.01), 20, 1e-12, "known PSNR");
  check::Near(DnPSNR(1), 0, 0, "unit-error PSNR");
  check::Require(std::isinf(DnPSNR(0)), "perfect reconstruction infinite PSNR");
  const auto perfect = DnScore(all, all);
  check::Require(perfect.mse == 0 && std::isinf(perfect.psnr), "perfect score preserves infinity");
  const auto score = DnScore(nn::tensor({.1, .2}, {2, 1, 1, 1}), nn::zeros({2, 1, 1, 1}));
  check::Near(score.psnr, (20 - 20 * std::log10(.2)) / 2, 1e-6, "mean per-image PSNR differs from pooled PSNR");
  check::Near(score.mse, .025, 1e-8, "pooled pixel MSE");
  const auto   box        = DnBoxFilter(nn::tensor({0, 1, 2, 3, 4, 5}, {1, 2, 3, 1})).to_vector<float>();
  const double expected[] = {8. / 3, 3, 10. / 3, 5. / 3, 2, 7. / 3};
  for (int i = 0; i < 6; ++i) check::Near(box[i], expected[i], 2e-7, "independent reflected box values/edges");
  std::cout << " Gaussian fixed/fresh seeds,signed residual pairing,no clipping,131072 draws mean/variance/fourth=" << mean << "/"
            << second << "/" << fourth << ",known/perfect/mean PSNR and reflected box fixtures passed\n";
}

void ResidualChecks() {
  const std::vector<float> y = {-.2f, .4f, 1.2f, .9f, .1f, -.1f}, r = {.1f, -.3f, .4f, -.1f, .2f, -.5f},
                           v = {-.1f, .1f, .2f, .1f, -.2f, .3f};
  Tensor       noisy = nn::from_vector(y, {2, 1, 3, 1}).requires_grad_(), noise = nn::from_vector(r, {2, 1, 3, 1}).requires_grad_();
  const Tensor clean     = DnSubtract(noisy, noise);
  const Tensor objective = DnResidualLoss(noise, nn::from_vector(v, {2, 1, 3, 1}));
  const auto   gradient  = nn::grad(objective, {noise})[0].to_vector<double>();
  const auto   identity  = nn::grad(nn::sum(clean), {noisy, noise});
  const auto   c = clean.to_vector<double>(), i0 = identity[0].to_vector<double>(), i1 = identity[1].to_vector<double>();
  double       loss = 0, error = 0;
  for (int i = 0; i < 6; ++i) {
    const double difference = double(r[i]) - v[i];
    loss += .5 * difference * difference / 6;
    check::Near(c[i], double(y[i]) - r[i], 1e-7, "signed noise is subtracted without activation");
    error = std::max(error, std::abs(gradient[i] - difference / 6));
    check::Near(i0[i], 1, 0, "noisy shortcut derivative+1");
    check::Near(i1[i], -1, 0, "predicted-noise derivative-1");
  }
  check::Near(objective.item(), loss, 1e-8, "host half-MSE average");
  check::Near(error, 0, 1e-8, "every residual-loss coordinate derivative");
  std::cout << " independent signed residual/output/half-MSE algebra and every-coordinate Jacobian max error=" << error << " passed\n";
}

void BatchNormChecks() {
  nn::ParameterStore p(1), state(2);
  std::vector<float> value(24), upstream(24);
  const double       gamma[] = {1.2, -.7}, beta[] = {-.1, .4};
  for (int i = 0; i < 24; ++i) value[i] = float((i % 9 - 4) * .17 + (i / 12) * .3), upstream[i] = float((i % 7 - 3) * .11);
  {
    nn::NoGradGuard guard;
    DnNorm(p, state, "bn", nn::zeros({2, 2, 3, 2}), false);
    p.set("bn/gamma", nn::tensor({1.2, -.7}));
    p.set("bn/beta", nn::tensor({-.1, .4}));
  }
  Tensor       x        = nn::from_vector(value, {2, 2, 3, 2}).requires_grad_();
  const Tensor training = DnNorm(p, state, "bn", x, true, false);
  const auto   gradient = nn::grad(nn::sum(training * nn::from_vector(upstream, {2, 2, 3, 2})), {x, p.at("bn/gamma"), p.at("bn/beta")});
  const auto   out = training.to_vector<double>(), gx = gradient[0].to_vector<double>(), gg = gradient[1].to_vector<double>(),
             gb = gradient[2].to_vector<double>();
  double mean[2] = {}, variance[2] = {}, du[2] = {}, dhat[2] = {}, error = 0;
  for (int i = 0; i < 24; ++i) mean[i % 2] += value[i] / 12.;
  for (int i = 0; i < 24; ++i) variance[i % 2] += std::pow(value[i] - mean[i % 2], 2) / 12;
  for (int i = 0; i < 24; ++i) {
    const int    c = i % 2;
    const double z = (value[i] - mean[c]) / std::sqrt(variance[c] + .0001);
    du[c] += upstream[i], dhat[c] += upstream[i] * z;
    check::Near(out[i], gamma[c] * z + beta[c], 5e-7, "BN biased population variance in training");
  }
  for (int i = 0; i < 24; ++i) {
    const int    c  = i % 2;
    const double z  = (value[i] - mean[c]) / std::sqrt(variance[c] + .0001);
    const double dx = gamma[c] / std::sqrt(variance[c] + .0001) * (upstream[i] - du[c] / 12 - z * dhat[c] / 12);
    error           = std::max(error, std::abs(gx[i] - dx));
  }
  for (int c = 0; c < 2; ++c) {
    error = std::max(error, std::abs(gg[c] - dhat[c]));
    error = std::max(error, std::abs(gb[c] - du[c]));
  }
  check::Near(error, 0, 2e-6, "every BN input/affine derivative");
  {
    nn::NoGradGuard guard;
    DnNorm(p, state, "bn", x.detach(), true, true);  // the moving-state update
  }
  const auto inference = DnNorm(p, state, "bn", x.detach(), false).to_vector<double>();
  const auto mm = state.at("bn/moving_mean").to_vector<double>(), mv = state.at("bn/moving_var").to_vector<double>();
  for (int c = 0; c < 2; ++c) {
    check::Near(mm[c], .95 * mean[c], 2e-7, "old-retention.05/new-batch.95 mean");
    check::Near(mv[c], .05 + .95 * variance[c] * 12 / 11, 2e-7, "unbiased variance used in moving BN state");
  }
  for (int i = 0; i < 24; ++i) {
    const int    c = i % 2;
    const double expected =
        gamma[c] * (value[i] - .95 * mean[c]) / std::sqrt(.05 + .95 * variance[c] * 12 / 11 + .0001) + beta[c];
    check::Near(inference[i], expected, 5e-7, "inference uses saved moving mean/variance");
  }
  std::cout << " independent BN train/infer/old-retention.05/unbiased-moving variance and every input/gamma/beta derivative max error="
            << error << " passed\n";
}

void ModelChecks() {
  for (int width : {64, 16}) {
    nn::ParameterStore p(3), state(4);
    std::vector<float> values(2 * 19 * 21), target(2 * 19 * 21);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(.5 + .7 * std::sin(i * .17)), target[i] = float(.1 * std::cos(i * .13));
    const Tensor input = nn::from_vector(values, {2, 19, 21, 1});
    const auto   model = DnCNN(p, state, input, width, true, false);
    const auto   loss  = DnResidualLoss(model.noise, nn::from_vector(target, {2, 19, 21, 1}));
    const auto   gradient = nn::grad(loss, p.parameters());
    check::Require(p.count() == (width == 64 ? 556096 : 35344) && state.count() == (width == 64 ? 1920 : 480) && p.size() == 48 &&
                       state.size() == 30,
                   "exact17-layer DnCNN parameter/BN census");
    check::Require(model.stages.size() == 17 && state.size() / 2 == 15, "all17 conv stages and15 BN updates");
    for (int i = 0; i < 17; ++i)
      check::Require(model.stages[i].shape() == nn::Shape({2, 19, 21, i == 16 ? 1 : width}), "exact odd-rectangular DnCNN batch/spatial/channel shape");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradient) {
      const double norm = nn::norm(g).item();
      check::Require(std::isfinite(norm), "finite DnCNN parameter gradient");
      check::Require(norm > 0, "every compact DnCNN tensor receives gradient");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " " << (width == 64 ? "full" : "compact") << " DnCNN17x" << width << ":" << p.count()
              << " parameters/all48 derivatives (executed),"
              << state.count() << " moving values\n";
    if (width == 64) continue;
    double orthogonal_error = 0;
    for (const auto& name : p.names()) {
      if (name.size() <= 2 || name.substr(name.size() - 2) != "/W") continue;
      const Tensor kernel = p.at(name);
      const auto   k      = kernel.to_vector<double>();
      const int    columns = int(kernel.size(3)), rows = int(kernel.numel() / columns), rank = std::min(rows, columns);
      for (int i = 0; i < rank; ++i)
        for (int j = 0; j < rank; ++j) {
          double dot = 0;
          for (int q = 0; q < std::max(rows, columns); ++q)
            dot += rows >= columns ? k[q * columns + i] * k[q * columns + j] : k[i * columns + q] * k[j * columns + q];
          orthogonal_error = std::max(orthogonal_error, std::abs(dot - (i == j ? 1 : 0)));
        }
    }
    check::Near(orthogonal_error, 0, 1e-6, "every first/hidden/final kernel orthogonal rows or columns");
    {
      nn::NoGradGuard guard;
      for (const auto& name : p.names()) p.set(name, nn::zeros(p.at(name).shape()));
    }
    const auto inference = DnCNN(p, state, input, 16, false);
    check::Require(inference.clean.to_vector<float>() == values, "zero predicted noise preserves every signed noisy pixel exactly");
    for (float v : inference.noise.to_vector<float>()) check::Near(v, 0, 0, "zero-model noise is exactly zero");
    std::cout << " compact all17 kernel orthogonality error=" << orthogonal_error << ",all48 numerical derivative norms[" << minimum
              << "," << maximum << "],17 exact stage shapes and unclipped identity passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks_only = false, accuracy = false;
    std::filesystem::path data = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        data = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 accuracy=0|1 threads=N");
    }
    NoiseAndMetricChecks();
    ResidualChecks();
    BatchNormChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "DnCNN fixtures passed (training not requested).\n";
      return 0;
    }
    TrainDenoising(data, epochs, accuracy, paths::OutputDir("models/dncnn"));
    std::cout << "All DnCNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
