// CMD: just run-nn receptive_fields
// Receptive fields and trained CNN feature maps, with numerical validation.
//
// Port of tflow's receptive_fields. Preparation: just run-nn cnn_mnist (writes
// output/nn/models/cnn_mnist/best.npz), then run this example; tflow's
// python3 scripts/plot_cnn_features.py draws the PNG/PDF figures from
// features.npz (not ported). MNIST comes from tflow's dataset cache (data=PATH).
// Restore the exact shared MNIST CNN, verify its test accuracy, and export
// intermediate activations for the first test example of every digit.
// Track receptive-field width, jump and center through conv/pool layers:
// width 1 -> 3 -> 4 -> 8 -> 10; final jump 4 and first center 2.
// Check the full predicted support with positive convolutions/average pools,
// including boundary clipping. The trained model's input gradient for one
// pooled feature lies inside that box; ReLU and max pooling may reduce it.
// A finite difference checks an active derivative, and an outside pixel
// perturbation leaves the selected feature unchanged. This local derivative
// is an illustration for one image/unit, not a dataset-level explanation.
// Data, layer geometry: output/nn/visualizations/receptive_fields/.
// Options: data=PATH checkpoint=PATH threads=N.

#include "common/dataset_assets.hpp"
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include "common/checks.hpp"
#include "common/cnn_baselines.hpp"
#include "common/mnist.hpp"
#include "common/paths.hpp"
#include "common/receptive_field.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using receptive::ReceptiveField;

namespace {

ReceptiveField Geometry() {
  const auto    path = paths::OutputDir("visualizations/receptive_fields");
  std::ofstream table(path / "geometry.tsv");
  table << "stage\twidth\tjump\tfirst_center\n";
  ReceptiveField field;
  table << "input\t1\t1\t0.5\n";
  int index = 0;
  for (const auto& name : {"conv1", "pool1", "conv2", "pool2"}) {
    const bool pool = index++ % 2;
    field           = field.After(pool ? 2 : 3, pool ? 2 : 1, pool ? 0 : 1);
    table << name << '\t' << field.width << '\t' << field.jump << '\t' << field.start << '\n';
  }
  check::Require(bool(table), "write receptive-field geometry");
  check::Require(field.width == 10 && field.jump == 4 && field.start == 2, "final receptive-field geometry");
  // The width of a dilated layer is an enclosing box, not a count of samples.
  check::Require(ReceptiveField{}.After(3, 1, 2, 2).width == 5, "dilation expands effective width");
  return field;
}

void SupportChecks(const ReceptiveField& field) {
  const Tensor kernel = nn::ones({3, 3, 1, 1});
  for (int test = 0; test < 2; ++test) {
    const int coordinate = test == 0 ? 0 : 3;
    Tensor    x          = nn::zeros({1, 28, 28, 1});
    x.requires_grad_();
    auto first   = nn::avg_pool2d(nn::conv2d(x, kernel, nn::conv_options(1, nn::Padding::Same)), 2, 2);
    auto second  = nn::avg_pool2d(nn::conv2d(first, kernel, nn::conv_options(1, nn::Padding::Same)), 2, 2);
    auto feature = nn::sum(nn::slice(second, {0, coordinate, coordinate, 0}, {1, 1, 1, 1}));
    const auto derivative = nn::grad(feature, {x})[0].to_vector<float>();
    const auto [lo, hi]   = field.Bounds(coordinate);
    int pixels_in_support = 0;
    for (int y = 0; y < 28; ++y)
      for (int xx = 0; xx < 28; ++xx) {
        const bool inside = y >= lo && y < hi && xx >= lo && xx < hi;
        check::Require((derivative[static_cast<std::size_t>(y * 28 + xx)] > 0) == inside,
                       "analytic and numerical support agree");
        pixels_in_support += inside;
      }
    std::cout << "  positive linear network: " << (test == 0 ? "corner" : "center")
              << " support=" << pixels_in_support << " pixels\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path dir        = paths::Dataset("mnist");
    std::filesystem::path checkpoint = paths::OutputRoot() / "models/cnn_mnist/best.npz";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value argument");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data")
        dir = value;
      else if (key == "checkpoint")
        checkpoint = value;
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else
        throw std::invalid_argument("unknown argument: " + key);
    }
    const auto field = Geometry();
    SupportChecks(field);
    check::Require(std::filesystem::exists(checkpoint), "run cnn_mnist first to create trained checkpoint");
    const auto dataset = dataset_assets::LoadMnist(dir.string(), "test");
    const auto count   = dataset.images.size(0);
    const auto pixels  = nn::reshape(dataset.images, {count, 784});

    nn::ParameterStore store(17);
    auto               logits_of = [](nn::ParameterStore& p, const Tensor& x, bool) {
      return cnn_baselines::MnistCnn(p, x).logits;
    };
    {
      nn::NoGradGuard guard;
      logits_of(store, nn::slice_axis(pixels, 0, 0, 1), false);
    }
    store.load(checkpoint.string());
    std::vector<std::int64_t> all(static_cast<std::size_t>(count));
    std::iota(all.begin(), all.end(), 0);
    const auto evaluation = mnist::Evaluate(store, logits_of, pixels, dataset.labels, all);
    check::TrainingQuality(evaluation.accuracy.value() >= .98, "restored shared CNN preserves test performance");
    std::cout << "  restored checkpoint test accuracy=" << evaluation.accuracy.value() << '\n';

    const auto                labels = dataset.labels.to_vector<std::int64_t>();
    std::vector<std::int64_t> rows(10, count);
    for (std::int64_t i = 0; i < count; ++i)
      if (rows[static_cast<std::size_t>(labels[static_cast<std::size_t>(i)])] == count)
        rows[static_cast<std::size_t>(labels[static_cast<std::size_t>(i)])] = i;
    for (auto row : rows) check::Require(row < count, "one example of each digit");
    nn::NamedTensors arrays;
    int              selected_channel = 0;
    {
      nn::NoGradGuard guard;
      const auto      x       = nn::take_rows(pixels, rows);
      const auto      network = cnn_baselines::MnistCnn(store, x);
      arrays = {{"conv1", network.conv1},
                {"conv2", network.conv2},
                {"images", network.image},
                {"labels", nn::take_rows(dataset.labels, rows)},
                {"pool2", network.pool2},
                {"predictions", nn::argmax(network.logits, 1)}};
      for (int c = 1; c < 32; ++c)
        if (network.pool2.at({0, 3, 3, c}) > network.pool2.at({0, 3, 3, selected_channel})) selected_channel = c;
      check::Require(network.pool2.at({0, 3, 3, selected_channel}) > 0, "active selected feature");
    }
    // Differentiate in Float64 so shrinking the finite-difference interval
    // does not amplify Float32 cancellation. A tiny deterministic texture
    // breaks blank-background max-pool ties in a lightly trained quick model.
    nn::ParameterStore derivative_store(17, nn::DType::Float64);
    for (const auto& name : store.names()) derivative_store.set(name, store.at(name).to(nn::DType::Float64));
    auto probe = nn::take_rows(pixels, {rows[0]}).to_vector<double>();
    if (!example_runtime::Full())
      for (std::size_t i = 0; i < probe.size(); ++i) probe[i] += .0001 * std::sin(.37 * i + .1);
    const auto single = nn::from_vector(probe, {1, 784});
    arrays.emplace_back("gradient_probe", single);
    auto       evaluate = [&](const Tensor& input, bool derivative) {
      Tensor x = input.clone();
      if (derivative) x.requires_grad_();
      const auto network  = cnn_baselines::MnistCnn(derivative_store, x);
      auto       selected = nn::sum(nn::slice(network.pool2, {0, 3, 3, selected_channel}, {1, 1, 1, 1}));
      Tensor     gradient = derivative ? nn::grad(selected, {x})[0] : Tensor();
      return std::pair<double, Tensor>{selected.item(), gradient};
    };
    const auto reference = evaluate(single, true);
    arrays.emplace_back("gradient", reference.second);
    const auto         dx = reference.second.to_vector<double>();
    std::vector<float> support(28 * 28, 0.f);
    const auto [lo, hi] = field.Bounds(3);
    int active = 0, strongest = 0;
    for (int py = 0; py < 28; ++py)
      for (int px = 0; px < 28; ++px) {
        const int   i          = py * 28 + px;
        const float derivative = dx[static_cast<std::size_t>(i)];
        const bool  inside     = py >= lo && py < hi && px >= lo && px < hi;
        support[static_cast<std::size_t>(i)] = inside;
        if (!inside) check::Near(derivative, 0, 0, "gradient stays inside theoretical receptive field");
        active += derivative != 0;
        if (std::abs(derivative) > std::abs(dx[static_cast<std::size_t>(strongest)])) strongest = i;
      }
    check::Require(active > 0, "nonzero feature gradient");
    nn::NoGradGuard guard;
    const auto      base    = single.to_vector<double>();
    auto            changed = base;
    changed[0] += .5f;
    check::Near(evaluate(nn::from_vector(changed, {1, 784}), false).first, reference.first, 0,
                "outside perturbation cannot affect feature");
    // A short-trained checkpoint can place a ReLU/max-pool boundary inside
    // the original .001 interval. Shrink the interval until both one-sided
    // slopes agree; retain the original derivative tolerance.
    double difference = 0;
    bool stable = false;
    for (double h : {.001, .0002, .0001, .00005, .00002, .000001, .00000001}) {
      auto plus = base, minus = base;
      plus[static_cast<std::size_t>(strongest)] += h;
      minus[static_cast<std::size_t>(strongest)] -= h;
      const double forward = (evaluate(nn::from_vector(plus, {1, 784}), false).first - reference.first) / h;
      const double backward = (reference.first - evaluate(nn::from_vector(minus, {1, 784}), false).first) / h;
      if (std::abs(forward - backward) <= .001) {
        difference = (forward + backward) / 2;
        stable = true;
        break;
      }
    }
    check::Require(stable, "finite-difference interval stays within one active branch");
    check::Near(difference, dx[static_cast<std::size_t>(strongest)], .001, "active derivative finite difference");
    arrays.emplace_back("support", nn::from_vector(support, {28, 28}));
    arrays.emplace_back("unit", nn::from_vector(std::vector<std::int32_t>{3, 3, selected_channel}, {3}));
    const auto out = paths::OutputDir("visualizations/receptive_fields") / "features.npz";
    nn::save_npz(out.string(), arrays);
    std::cout << "  pool2[3,3," << selected_channel << "]: theoretical box=[" << lo << ',' << hi
              << ") squared; nonzero input derivatives=" << active << "/100\n"
              << "  finite difference=" << difference << "; analytic=" << dx[static_cast<std::size_t>(strongest)]
              << '\n'
              << "  export: " << out.string() << "\nAll receptive-field checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
