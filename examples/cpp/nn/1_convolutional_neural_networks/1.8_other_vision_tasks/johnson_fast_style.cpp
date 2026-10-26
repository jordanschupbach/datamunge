// CMD: just run-nn johnson_fast_style
// Johnson feed-forward single-style transformation with frozen VGG-16 losses.
// https://arxiv.org/abs/1603.08155
// https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
// Reflection40,9x9conv,2 stride2 downs,5 VALID/cropped residual blocks,
// 2 learned stride2 transposes,9x9conv/scaled tanh. BN follows all but final
// conv; residual additions have no ReLU. Full base32 model 1679235 parameters;
// compact base8 training108771 retains every stage and exact spatial geometry.
// Explicit padding/cropping reproduces Torch k3/s2/p1/adjustment1 alignment.
// Frozen actual ImageNet VGG16/max pools, relu2_2 content and relu1_2,2_2,
// 3_3,4_3 style. Gram normalization1/(C*H*W), Frobenius style, quadratic TV.
// Default3epochs,24fit/8validation/8test CIFAR natural contents resized32->64,
// train rows0..23,val24..31,official test8..15; test0..7 used only in development.
// batch2 Adam.001,no decay. Fixed initial scaling balances content/style/TV once; select
// validation perceptual objective, checkpoint generator and moving BN state.
// Test via a separate generator-only model: one forward, no VGG/optimization.
// Export all8 tensors/preview and measured small-image CPU latency, no
// real-time or COCO/paper-quality claim.
//
// Port of tflow's johnson_fast_style. Assets come from tflow's
// `just style-assets; just johnson-assets` in $DATAMUNGE_DATA_DIR/style_transfer
// (vgg16.npz, johnson_reference.npz Keras fixture, style.jpg) and CIFAR-10
// binaries from $DATAMUNGE_DATA_DIR/cifar10. Content images (legacy
// ResizeBilinear), the style image (libjpeg-turbo decode, as TF DecodeJpeg) and
// the batch order (nn::BatchOrder seed 29) are tflow's exactly; generator
// weights come from datamunge's generator, so trained numbers match TF
// statistically. TF builds the full base32 model as a graph only; here it also
// runs forward/backward once on a 2x256x256 batch. Option seed= (default 17)
// seeds the generator weights.
// The checkpoint is best.npz (generator weights + moving BN). Outputs go to
// output/nn/models/johnson_fast_style.
// Options: epochs=3 checks_only=0|1 assets=PATH data=PATH threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/fast_style_transfer.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/fast_style_transfer.hpp.
namespace style {
using namespace ::style;
inline void TrainJohnson(const std::filesystem::path& assets, const std::filesystem::path& data_root, const TensorMap& weights, int epochs,
                         const std::filesystem::path& directory, std::uint64_t seed = 17) {
  check::Require(epochs >= 2, ">=2 epochs required for short generator learning check");
  const auto train = data::LoadCifar({data_root / "data_batch_1.bin"}, data::Cifar::k10),
             test  = data::LoadCifar({data_root / "test_batch.bin"}, data::Cifar::k10);
  check::Require(train.size() == 10000 && test.size() == 10000, "official CIFAR-10 binary batch sizes");
  std::vector<std::size_t> fit_rows(24), val_rows(8), test_rows(8);
  std::iota(fit_rows.begin(), fit_rows.end(), 0);
  std::iota(val_rows.begin(), val_rows.end(), 24);
  std::iota(test_rows.begin(), test_rows.end(), 8);
  const auto fitting = FastStyleImages(train, fit_rows), validation = FastStyleImages(train, val_rows), testing = FastStyleImages(test, test_rows);
  const auto style = LoadStyleImage(assets / "style.jpg", 64);
  FastStyleModel model(weights, seed);
  model.Style(style);
  const auto fit_targets = CacheFastStyleContent(model, fitting), val_targets = CacheFastStyleContent(model, validation),
             test_targets = CacheFastStyleContent(model, testing);
  const auto probe = FastStyleBatch(fitting, {0, 1}), probe_target = FastStyleBatch(fit_targets, {0, 1});
  const auto raw   = model.Run(probe, probe_target, true);
  check::Require(raw[1] > 0 && raw[2] > 0 && raw[3] > 1e-12, "nonzero initial perceptual losses");
  // TF feeds the denominators as float32 scalars.
  model.content_denominator = double(float(raw[1]));
  model.style_denominator   = double(float(raw[2]));
  model.tv_denominator      = double(float(raw[3]));
  const auto initial = model.Run(probe, probe_target, true), untrained = ScoreFastStyle(model, testing, test_targets);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_objective\tvalidation_objective\tvalidation_content\tvalidation_style\n";
  double              best     = INFINITY;
  int                 selected = -1;
  std::vector<double> losses;
  nn::BatchOrder      order(24, 2, 29);
  std::cout << "== Johnson fast style:" << model.parameters.count() << " trainable parameters/" << model.state.count()
            << " BN state values; full5 cropped residual blocks, BN and learned transpose upsamples" << std::endl;
  std::cout << " " << epochs
            << "epochs default,24fit/8validation/8official-test CIFAR images (test rows8..15;0..7 diagnostic-only),32->64px,batch2,Adam.001,no "
               "decay; fixed ImageNet VGG16 relu2_2 and4 style layers/max pools"
            << std::endl;
  std::cout << " fixed initial content/style/TV denominators=" << raw[1] << '/' << raw[2] << '/' << raw[3]
            << ", all three fixed-weight contributions initially1; initial fitting objective=" << initial[0] << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum   = 0;
    int    count = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const std::vector<std::size_t> rows(ids.begin(), ids.end());
      const auto value = model.Run(FastStyleBatch(fitting, rows), FastStyleBatch(fit_targets, rows), true, true);
      sum += value[0] * double(rows.size());
      count += int(rows.size());
    }
    check::Require(count == 24, "all fitting images used each epoch");
    losses.push_back(sum / count);
    const auto val = ScoreFastStyle(model, validation, val_targets);
    if (val[0] < best) {
      best = val[0], selected = epoch;
      SaveJohnson(model.parameters, model.state, prefix);
    }
    history << epoch << '\t' << losses.back() << '\t' << val[0] << '\t' << val[1] << '\t' << val[2] << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": mean training objective=" << losses.back() << ", inference validation=" << val[0]
              << ", relative content/style=" << val[1] / raw[1] << '/' << val[2] / raw[2] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  RestoreJohnson(model.parameters, model.state, prefix);
  check::Near(ScoreFastStyle(model, validation, val_targets)[0], best, 1e-6, "restored validation perceptual objective");
  const auto learned = model.Run(probe, probe_target, true), result = ScoreFastStyle(model, testing, test_targets);
  check::Require(losses.back() < losses.front() - 1e-4 && learned[0] < initial[0] - 1e-4 && learned[2] < initial[2] * .999,
                 "mean/fixed fitting objectives and fitting Gram style must improve");
  model.vgg.CheckFrozen();
  JohnsonInference standalone(prefix);
  const auto       before = standalone.Snapshot();
  const auto       predictions = standalone.Predict(testing), reference = model.Predict(testing);
  const auto       pv = predictions.to_vector<float>(), rv = reference.to_vector<float>();
  double           error = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) error = std::max(error, std::abs(double(pv[i]) - rv[i]));
  check::Near(error, 0, 2e-6, "separate generator-only graph matches trained inference");
  double                 sensitivity  = 0;
  constexpr std::int64_t image_values = 64 * 64 * 3;
  for (int n = 1; n < 8; ++n)
    for (std::int64_t i = 0; i < image_values; ++i)
      sensitivity += std::abs(pv[std::size_t(n * image_values + i)] - pv[std::size_t(i)]);
  sensitivity /= 7 * image_values;
  check::Require(sensitivity > 1e-4, "distinct contents must not collapse to the same stylized image");
  const auto one = FastStyleBatch(testing, {0});
  standalone.Predict(one);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) standalone.Predict(one);
  const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 3;
  check::Require(before == standalone.Snapshot(), "single-pass inference never updates weights or BN state");
  const auto         tv = testing.to_vector<float>(), sv = style.to_vector<float>();
  std::vector<float> preview(std::size_t(8 * 64) * 192 * 3);
  for (int n = 0; n < 8; ++n)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c) {
          const std::size_t row = std::size_t(n * 64 + y) * 192, i = std::size_t(((n * 64 + y) * 64 + x) * 3 + c),
                            s = std::size_t((y * 64 + x) * 3 + c);
          preview[(row + std::size_t(x)) * 3 + std::size_t(c)]       = tv[i];
          preview[(row + 64 + std::size_t(x)) * 3 + std::size_t(c)]  = sv[s];
          preview[(row + 128 + std::size_t(x)) * 3 + std::size_t(c)] = pv[i];
        }
  WriteStylePng(nn::from_vector(preview, {1, 8 * 64, 192, 3}), directory / "comparison.png");
  nn::save_npz((directory / "test_outputs.npz").string(), {{"input", testing}, {"stylized", predictions}});
  if (std::filesystem::exists(assets / "johnson_manifest.json"))
    std::filesystem::copy_file(assets / "johnson_manifest.json", directory / "assets_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "write generator history");
  std::cout << " selected=" << selected << "; restored validation=" << best << " verified; fixed fitting objective " << initial[0] << " -> "
            << learned[0] << ", relative fitting content/style=" << learned[1] / raw[1] << '/' << learned[2] / raw[2] << std::endl;
  std::cout << " all8 held-out contents: objective " << untrained[0] << " -> " << result[0] << ", relative content/style=" << result[1] / raw[1]
            << '/' << result[2] / raw[2] << ", normalized TV=" << result[3] << ", raw-TV ratio vs initial fitting output=" << result[3] / raw[3]
            << ", output change across distinct test inputs=" << sensitivity << std::endl;
  std::cout << " generator-only checkpoint model (no VGG/targets/optimizer), output parity error=" << error
            << ", 3-call CPU single-image mean=" << milliseconds << "ms; all weights/BN state unchanged during inference" << std::endl;
  std::cout << " all26 pretrained VGG16 tensors/14714688 values unchanged bitwise; checkpoint/history/input-style-output preview and all8 output "
               "tensors: "
            << directory.string() << " (training " << seconds << " s)\n full short Johnson training and one-pass inference demonstration passed"
            << std::endl;
}
}  // namespace style



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace style;

namespace {

double Error(const Tensor& actual, const Tensor& expected) {
  check::Require(actual.dtype() == nn::DType::Float32 && actual.shape() == expected.shape(), "perceptual reference shape/dtype");
  const auto a = actual.contiguous().to_vector<float>(), b = expected.contiguous().to_vector<float>();
  double     difference = 0, scale = 1;
  for (std::size_t i = 0; i < a.size(); ++i) {
    check::Require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite perceptual values");
    difference = std::max(difference, std::abs(double(a[i]) - b[i]));
    scale      = std::max(scale, std::abs(double(b[i])));
  }
  return difference / scale;
}

void ReflectResidualChecks() {
  std::vector<float> values(12), upstream(56);
  for (int i = 0; i < 12; ++i) values[std::size_t(i)] = float(i - 6);
  for (int i = 0; i < 56; ++i) upstream[std::size_t(i)] = .1f * float(i % 7 - 3);
  Tensor       x         = nn::from_vector(values, {1, 3, 4, 1}).requires_grad_();
  const Tensor reflected = JohnsonReflect(x, 2);
  const auto   gradient  = nn::grad(nn::sum(reflected * nn::from_vector(upstream, {1, 7, 8, 1})), {x})[0].to_vector<float>();
  const auto   rv        = reflected.to_vector<float>();
  auto         reflect   = [](int i, int n) { return i < 0 ? -i : i >= n ? 2 * n - 2 - i : i; };
  double       dx[12]    = {};
  for (int y = 0; y < 7; ++y)
    for (int col = 0; col < 8; ++col) {
      const int source = reflect(y - 2, 3) * 4 + reflect(col - 2, 4);
      check::Near(rv[std::size_t(y * 8 + col)], values[std::size_t(source)], 0, "REFLECT edge/corner/signed values");
      dx[source] += upstream[std::size_t(y * 8 + col)];
    }
  for (int i = 0; i < 12; ++i) check::Near(gradient[std::size_t(i)], dx[i], 2e-7, "every reflected pixel derivative");
  nn::ParameterStore p(17), state(18);
  std::vector<float> signed_input(9 * 11 * 2);
  for (std::size_t i = 0; i < signed_input.size(); ++i) signed_input[i] = .01f * (float(i) - 100);
  Tensor input = nn::from_vector(signed_input, {1, 9, 11, 2}).requires_grad_();
  {
    nn::NoGradGuard guard;
    JohnsonResidual(p, state, "res", input, true, false);
    for (const auto& name : p.names())
      if (name.find("/conv") != std::string::npos) p.set(name, nn::zeros(p.at(name).shape()));
  }
  const Tensor residual = JohnsonResidual(p, state, "res", input, true);
  const auto   rg       = nn::grad(nn::sum(residual), {input})[0].to_vector<float>();
  const auto   out      = residual.to_vector<float>();
  check::Require(residual.shape() == nn::Shape({1, 5, 7, 2}), "two VALID convs shrink four, shortcut crop two each side");
  for (int y = 0; y < 9; ++y)
    for (int col = 0; col < 11; ++col)
      for (int c = 0; c < 2; ++c) {
        const bool inside = y >= 2 && y < 7 && col >= 2 && col < 9;
        check::Near(rg[std::size_t((y * 11 + col) * 2 + c)], inside ? 1 : 0, 0, "cropped shortcut exact input Jacobian");
        if (inside)
          check::Near(out[std::size_t(((y - 2) * 7 + col - 2) * 2 + c)], signed_input[std::size_t((y * 11 + col) * 2 + c)], 0,
                      "negative residual identity survives without post-add ReLU");
      }
  std::cout << "  REFLECT corners/every derivative, two-pixel shortcut crop and signed identity/no-post-add-ReLU fixtures passed" << std::endl;
}

void StrideChecks() {
  for (bool transpose : {false, true}) {
    const int          h = transpose ? 2 : 4, w = transpose ? 3 : 6, oh = transpose ? 4 : 2, ow = transpose ? 6 : 3, ci = 2, co = 3;
    nn::ParameterStore p(17);
    std::vector<float> x(std::size_t(h * w * ci)), kernel(std::size_t(9 * ci * co)), g(std::size_t(oh * ow * co));
    const std::vector<float> bias = {-.1f, .2f, .05f};
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = .3f * std::sin(float(i) * .37f);
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] = .1f * std::cos(float(i) * .29f);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = .02f * float(int(i % 9) - 3);
    Tensor input = nn::from_vector(x, {1, h, w, ci}).requires_grad_();
    {
      nn::NoGradGuard guard;
      transpose ? JohnsonUp(p, "layer", input, co) : JohnsonDown(p, "layer", input, co);
      p.set("layer/W", nn::from_vector(kernel, transpose ? nn::Shape{3, 3, co, ci} : nn::Shape{3, 3, ci, co}));
      p.set("layer/b", nn::from_vector(bias));
    }
    const Tensor output   = transpose ? JohnsonUp(p, "layer", input, co) : JohnsonDown(p, "layer", input, co);
    const auto   gradient = nn::grad(nn::sum(output * nn::from_vector(g, {1, oh, ow, co})), {input, p.at("layer/W"), p.at("layer/b")});
    std::vector<double> expected(std::size_t(oh * ow * co)), dx(x.size()), dw(kernel.size()), db(co);
    for (int y = 0; y < oh; ++y)
      for (int col = 0; col < ow; ++col)
        for (int o = 0; o < co; ++o) {
          expected[std::size_t((y * ow + col) * co + o)] = bias[std::size_t(o)];
          db[std::size_t(o)] += g[std::size_t((y * ow + col) * co + o)];
        }
    for (int y = 0; y < (transpose ? h : oh); ++y)
      for (int col = 0; col < (transpose ? w : ow); ++col)
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx)
            for (int c = 0; c < ci; ++c)
              for (int o = 0; o < co; ++o) {
                const int iy = transpose ? y : 2 * y + ky - 1, ix = transpose ? col : 2 * col + kx - 1, oy = transpose ? 2 * y + ky - 1 : y,
                          ox = transpose ? 2 * col + kx - 1 : col;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w || oy < 0 || oy >= oh || ox < 0 || ox >= ow) continue;
                const int input_index = (iy * w + ix) * ci + c, output_index = (oy * ow + ox) * co + o,
                          wi = transpose ? ((ky * 3 + kx) * co + o) * ci + c : ((ky * 3 + kx) * ci + c) * co + o;
                const double value = x[std::size_t(input_index)], weight = kernel[std::size_t(wi)], up = g[std::size_t(output_index)];
                expected[std::size_t(output_index)] += value * weight;
                dx[std::size_t(input_index)] += weight * up;
                dw[std::size_t(wi)] += value * up;
              }
    const auto ov = output.to_vector<float>(), gx = gradient[0].to_vector<float>(), gw = gradient[1].to_vector<float>(),
               gb = gradient[2].to_vector<float>();
    double error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) error = std::max(error, std::abs(ov[i] - expected[i]));
    for (std::size_t i = 0; i < dx.size(); ++i) error = std::max(error, std::abs(gx[i] - dx[i]));
    for (std::size_t i = 0; i < dw.size(); ++i) error = std::max(error, std::abs(gw[i] - dw[i]));
    for (int i = 0; i < co; ++i) error = std::max(error, std::abs(gb[std::size_t(i)] - db[std::size_t(i)]));
    check::Near(error, 0, 1e-6, "independent stride forward/input/filter/bias derivatives");
    std::cout << "  Torch-aligned " << (transpose ? "transpose" : "down-convolution") << " host forward/every-gradient error=" << error << std::endl;
  }
}

void LossChecks() {
  std::vector<float> values(24), target(24);
  for (int i = 0; i < 24; ++i) values[std::size_t(i)] = .1f * float(i % 11 - 5), target[std::size_t(i)] = .05f * float(i % 7 - 3);
  const std::vector<float> style = {.2f, .03f, .03f, .1f};
  Tensor                   x     = nn::from_vector(values, {2, 2, 3, 2}).requires_grad_();
  const Tensor             st    = nn::from_vector(style, {1, 2, 2});
  const Tensor             gram  = JohnsonGram(x);
  const Tensor content = JohnsonContent(x, nn::from_vector(target, {2, 2, 3, 2})), style_loss = JohnsonStyle({x, x, x, x}, {st, st, st, st}),
               tv = JohnsonTV(x);
  const auto gc = nn::grad(content, {x})[0].to_vector<float>(), gs = nn::grad(style_loss, {x})[0].to_vector<float>(),
             gt = nn::grad(tv, {x})[0].to_vector<float>();
  const auto gv = gram.to_vector<float>();
  double     g[2][2][2] = {}, host_c = 0, host_s = 0, host_tv = 0, dtv[24] = {}, error = 0;
  for (int b = 0; b < 2; ++b)
    for (int c = 0; c < 2; ++c)
      for (int d = 0; d < 2; ++d) {
        for (int m = 0; m < 6; ++m) g[b][c][d] += double(values[std::size_t((b * 6 + m) * 2 + c)]) * values[std::size_t((b * 6 + m) * 2 + d)] / 12;
        check::Near(gv[std::size_t((b * 2 + c) * 2 + d)], g[b][c][d], 1e-8, "Gram includes channel AND spatial normalization");
        host_s += 4 * std::pow(g[b][c][d] - style[std::size_t(c * 2 + d)], 2) / 2;
      }
  for (int i = 0; i < 24; ++i) {
    const double diff = double(values[std::size_t(i)]) - target[std::size_t(i)];
    host_c += diff * diff / 24;
    error = std::max(error, std::abs(gc[std::size_t(i)] - 2 * diff / 24));
    const int b = i / 12, c = i % 2, m = (i % 12) / 2;
    double    ds = 0;
    for (int j = 0; j < 2; ++j) ds += 16 * values[std::size_t((b * 6 + m) * 2 + j)] * (g[b][j][c] - style[std::size_t(j * 2 + c)]) / (2 * 6 * 2);
    error = std::max(error, std::abs(gs[std::size_t(i)] - ds));
  }
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 1; ++y)
      for (int col = 0; col < 2; ++col)
        for (int c = 0; c < 2; ++c) {
          const int    i = ((b * 2 + y) * 3 + col) * 2 + c, right = i + 2, down = i + 6;
          const double dx = double(values[std::size_t(i)]) - values[std::size_t(right)], dy = double(values[std::size_t(i)]) - values[std::size_t(down)];
          host_tv += (dx * dx + dy * dy) / 16;
          dtv[i] += (dx + dy) / 8;
          dtv[right] -= dx / 8;
          dtv[down] -= dy / 8;
        }
  for (int i = 0; i < 24; ++i) error = std::max(error, std::abs(gt[std::size_t(i)] - dtv[i]));
  check::Near(content.item(), host_c, 1e-8, "mean feature reconstruction");
  check::Near(style_loss.item(), host_s, 1e-7, "sum four Frobenius layer losses, average independent batches");
  check::Near(tv.item(), host_tv, 5e-8, "quadratic TV joint interior stencil, float32 rounding");
  check::Near(error, 0, 1e-7, "all content/style/TV derivatives including unused bottom-right corner");
  std::cout << "  channel-spatial Gram/content/Frobenius/TV values and every feature/pixel Jacobian error=" << error << std::endl;
}

void KerasParity(const TensorMap& weights, const TensorMap& fixture) {
  const FrozenVgg16                 vgg(weights);
  Tensor                            x        = fixture.at("input").clone().requires_grad_();
  const auto                        features = vgg.Extract(x);
  const std::array<std::string, 4> layers   = {"block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"};
  std::array<Tensor, 4>             targets;
  for (std::size_t i = 0; i < 4; ++i) targets[i] = fixture.at(layers[i] + "_gram");
  const Tensor c = JohnsonContent(features.content, fixture.at("content_target")), st = JohnsonStyle(features.style, targets), tv = JohnsonTV(x);
  const Tensor objective = 1e-5 * c + 1e-7 * st + .03 * tv;
  const Tensor gradient  = nn::grad(objective, {x})[0];
  double       feature_error = 0, loss_error = 0;
  for (std::size_t i = 0; i < 4; ++i) feature_error = std::max(feature_error, Error(features.style[i], fixture.at(layers[i])));
  const Tensor     outs[] = {c, st, tv, objective};
  const char*      keys[] = {"content_loss", "style_loss", "tv", "objective"};
  for (int i = 0; i < 4; ++i) loss_error = std::max(loss_error, Error(nn::reshape(outs[i], fixture.at(keys[i]).shape()), fixture.at(keys[i])));
  const double gradient_error = Error(gradient, fixture.at("image_gradient"));
  check::Near(feature_error, 0, 3e-4, "all VGG16 selected pretrained features");
  check::Near(loss_error, 0, 3e-4, "all independent Keras perceptual/TV scalars");
  check::Near(gradient_error, 0, 5e-4, "every batched input pixel gradient vs GradientTape");
  std::cout << "  genuine pretrained VGG16 Keras/C++ parity: features=" << feature_error << ", all losses=" << loss_error
            << ", every batched pixel derivative=" << gradient_error << std::endl;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(18);
    const Tensor       x   = nn::full({2, 256, 256, 3}, .5);
    const auto         net = JohnsonTransform(p, state, x, 32, true, false);
    const auto         g   = nn::grad(nn::mean(nn::square(net.image)), p.parameters());
    check::Require(p.count() == 1679235 && state.count() == 3200 && g.size() == 62 && state.size() == 30 && state.size() / 2 == 15,
                   "full parameter/BN census and15 normalizations");
    check::Require(net.stages[0].size(1) == 336 && net.stages[3].size(1) == 84 && net.stages[8].size(1) == 64 && net.image.size(1) == 256,
                   "paper256->336->84->64->256 geometry");
    std::cout << "  full Johnson32/64/128:" << p.count() << " trainable/" << state.count() << " BN-state values," << g.size()
              << " derivatives; paper256->336->84->64->256 geometry (executed)" << std::endl;
  }
  nn::ParameterStore p(17), state(18);
  {
    nn::NoGradGuard guard;
    JohnsonTransform(p, state, nn::zeros({2, 64, 64, 3}), 8, false);
  }
  check::Require(p.count() == 108771 && state.count() == 800 && p.size() == 62, "compact independent parameter census");
  double low = INFINITY, high = 0;
  int    zero_bias = 0;
  for (int side : {64, 80}) {
    std::vector<float> input(std::size_t(2 * side * side * 3));
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(.5 + .45 * std::sin(float(i) * .113f));
    const Tensor x        = nn::from_vector(input, {2, side, side, 3});
    const auto   training = JohnsonTransform(p, state, x, 8, true, false);
    Tensor       inferred;
    {
      nn::NoGradGuard guard;
      inferred = JohnsonTransform(p, state, x, 8, false).image;
    }
    const auto g = nn::grad(nn::mean(nn::squared_difference(training.image, x * .4)), p.parameters());
    check::Require(g.size() == 62, "compact Johnson all numerical derivatives");
    check::Require(training.image.shape() == x.shape() && inferred.shape() == x.shape(), "exact input/output alignment");
    const std::vector<int> sides    = {side + 80,           side + 80,           (side + 80) / 2,      (side + 80) / 4, (side + 80) / 4 - 4, (side + 80) / 4 - 8,
                                       (side + 80) / 4 - 12, (side + 80) / 4 - 16, side / 4, side / 2, side},
                           channels = {3, 8, 16, 32, 32, 32, 32, 32, 32, 16, 8};
    check::Require(training.stages.size() == sides.size(), "all11 Johnson stages exposed");
    for (std::size_t i = 0; i < sides.size(); ++i)
      check::Require(training.stages[i].shape() == nn::Shape({2, sides[i], sides[i], channels[i]}), "every pad/encoder/cropped residual/decoder stage");
    for (const Tensor& image : {training.image, inferred})
      for (float v : image.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "bounded scaled-tanh train/infer images");
    for (std::size_t i = 0; i < g.size(); ++i) {
      const double norm = nn::norm(g[i]).item();
      check::Require(std::isfinite(norm), "every generator derivative finite");
      const auto name           = p.names()[i];
      const bool redundant_bias = name.size() >= 2 && name.substr(name.size() - 2) == "/b" && name != "output/b";
      if (norm == 0) {
        check::Require(redundant_bias, "only shift-invariant pre-BN conv biases may have zero derivative");
        ++zero_bias;
      } else {
        low = std::min(low, norm), high = std::max(high, norm);
      }
    }
  }
  std::cout << "  compact108771/800:64px and80px all11 stage shapes/62 finite derivatives passed; nonzero norms[" << low << ',' << high
            << "], zero pre-BN bias tensors=" << zero_bias << " (allowed by BN shift invariance)" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    std::uint64_t         seed        = 17;
    bool                  checks_only = false;
    std::filesystem::path assets = paths::Dataset("style_transfer"), data = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("assets=", 0) == 0)
        assets = arg.substr(7);
      else if (arg.rfind("data=", 0) == 0)
        data = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("seed=", 0) == 0)
        seed = std::stoull(arg.substr(5));
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N assets=PATH data=PATH checks_only=0|1 threads=N seed=N");
    }
    ReflectResidualChecks();
    StrideChecks();
    LossChecks();
    check::Require(std::filesystem::exists(assets / "vgg16.npz"), "VGG16 missing; run tflow's just johnson-assets after just style-assets");
    const auto weights = LoadTensorMap((assets / "vgg16.npz").string()), fixture = LoadTensorMap((assets / "johnson_reference.npz").string());
    KerasParity(weights, fixture);
    ModelChecks();
    if (checks_only) {
      std::cout << "All Johnson fixtures passed (training not requested).\n";
      return 0;
    }
    TrainJohnson(assets, data, weights, epochs, paths::OutputDir("models/johnson_fast_style"), seed);
    std::cout << "All Johnson fast style transfer checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
