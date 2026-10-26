// CMD: just run-nn glow_density
// Glow exact likelihood/inverse/Jacobians on a reproducible analytic banana density.
//
// Port of tflow's glow_density (common/ml/glow_density.h, common/nn/glow_density.h
// -> examples/nn/common/{flow_density,glow_density}.hpp). Four two-channel Glow
// steps: ActNorm (data-initialised ONCE from the first 32 fit points), a learned
// invertible 2x2 mixing (initial rotations), an affine RealNVP coupling; 296
// parameters / 28 tensors. Exact log|det| includes the ActNorm scales, the matrix
// determinant (log|det| handles a negative determinant: signed_mixing_fixture)
// and the input-dependent coupling scale. Data: SYNTHETIC banana, bit-identical
// to tflow's. Three epochs of Adam .003, batch 32 in BatchOrder(512, 32, 9521).
//
// Checks (as tflow): the signed-mixing fixture (W = [[1,2],[3,4]]: log|det| =
// log 2, inverse -log 2, roundtrip); census; restore; actual Jacobians vs
// analytic per-point logdets both ways; roundtrips; repeat sampling; 1024
// normalization identities; per-point logdet cancellation; spatially varying
// Jacobian; parity; immutability. Added: when TF's initial arrays are replayed,
// re-deriving the ActNorm initialisation from TF's other initial weights must
// reproduce TF's ActNorm values; data and TF replay checks. Deviations:
// final.npz, no independent_audit.json.
// Options: epochs=3 out=DIR init=auto|tf|nn|PATH tf=DIR threads=4 verbose=0.
// Output: output/nn/models/glow_density.
#include "common/glow_density.hpp"

// Experiment workflow from examples/nn/common/flow_density.hpp.
namespace flowd {
using namespace ::flowd;
template <typename F>
int Main(int argc, char** argv, const std::string& name, F&& body) {
  try {
    const auto o = ParseOptions(argc, argv, name);
    body(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
}  // namespace flowd



// Experiment workflow from examples/nn/common/flow_density.hpp.
namespace flowd {
using namespace ::flowd;
inline Result Run(const Spec& spec, const Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root = spec.variant.empty() ? o.out : o.out / spec.variant;
  const fs::path tf   = spec.variant.empty() ? o.tf : o.tf / spec.variant;
  fs::create_directories(root);
  if (spec.before) spec.before(root);
  const Data data = MakeData(spec.kind);
  nn::save_npz((root / "data.npz").string(), replay::ToNamed(data.arrays));
  {
    std::ofstream protocol(root / "source_protocol.json");
    protocol << spec.protocol;
    check::Require(bool(protocol), "source protocol written");
  }
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = replay::ToMap(nn::load_npz((tf / "data.npz").string()));
    for (const auto& [k, v] : data.arrays)
      check::Require(theirs.count(k) && Bits(v) == Bits(theirs.at(k)), "data array " + k + " bit-identical to TF's data.npz");
    std::cout << "data: all " << data.arrays.size() << " arrays bit-identical to TF's data.npz\n";
  }
  const Tensor& fit = data.pools[0];
  auto          model = spec.make(true);
  check::Require(model->store.count() == spec.parameters && std::int64_t(model->store.size()) == spec.tensors,
                 std::to_string(spec.parameters) + " " + spec.tag + " parameters / " + std::to_string(spec.tensors) + " tensors");
  model->Prepare(fit, root);
  const fs::path init = InitFile(o.init, tf);
  if (!init.empty()) {
    replay::LoadInto(model->store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
    if (spec.replayed) spec.replayed(*model, data);
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), replay::ToNamed(model->Arrays()));
  nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(model->Evaluate(data.pools[2])));
  nn::BatchOrder order(512, 32, spec.batch_seed);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << (spec.global_logdet ? "epoch\tvalidation_nll\tvalidation_forward_logdet\n"
                                 : "epoch\tvalidation_nll\tvalidation_mean_forward_logdet\n");
  batches << "epoch\tstep\trow\n";
  int    steps = 0;
  Tensor first;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor batch = nn::take_rows(fit, rows);
      if (!steps) {
        first = batch;
        nn::save_npz((root / "first_batch.npz").string(), {{"points", batch}});
        nn::save_npz((root / "first_forward.npz").string(), replay::ToNamed(model->Evaluate(batch)));
        nn::save_npz((root / "initial_gradients.npz").string(), replay::ToNamed(model->Gradient(batch)));
      }
      model->Update(batch);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), replay::ToNamed(model->Arrays()));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\n';
      ++steps;
    }
    const auto   output = model->Evaluate(data.pools[1]);
    const double loss = output.at("loss").item<float>(), ld = Mean(output.at("forward_logdet"));
    history << epoch << '\t' << loss << '\t' << ld << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), replay::ToNamed(model->Arrays()));
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_validation.npz")).string(), replay::ToNamed(output));
    std::cout << "epoch" << epoch << ":validationNLL=" << loss << ",forwardlogdet=" << ld << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model->store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(model->Gradient(first)));
  auto deployed = spec.make(false);
  deployed->store.load((root / "final.npz").string(), true);
  check::Require(SameBits(model->Arrays(), deployed->Arrays()), "all " + spec.tag + " weights BITEXACT restore");
  if (spec.after) spec.after(*deployed, data, root);
  const auto before = deployed->Arrays();
  const auto output = deployed->Evaluate(data.pools[2]);
  nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(output));
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(deployed->Arrays()));

  // Exact (or, for the CNF, numerical) inverse and analytic logdet against the actual Jacobian.
  const auto test = data.pools[2].to_vector<float>();
  const auto rec = output.at("reconstruction").to_vector<float>(), j0 = output.at("jacobian_row0").to_vector<float>(),
             j1 = output.at("jacobian_row1").to_vector<float>(), fld = output.at("forward_logdet").to_vector<float>();
  double roundtrip = 0, det_error = 0;
  for (int b = 0; b < 128; ++b) {
    for (int c = 0; c < 2; ++c) roundtrip = std::max(roundtrip, std::abs(double(rec[b * 2 + c]) - test[b * 2 + c]));
    const double determinant = double(j0[b * 2]) * j1[b * 2 + 1] - double(j0[b * 2 + 1]) * j1[b * 2];
    check::Require(determinant > 0, "orientation preserving " + spec.tag + " Jacobian");
    det_error = std::max(det_error, std::abs(std::log(determinant) - At(fld, b)));
  }
  check::Require(roundtrip < 5e-6 && det_error < 5e-6, std::string(spec.numerical ? "numerical" : "exact") +
                                                           " inverse and analytic logdet within F32");
  const auto samples = deployed->Sample(data.prior);
  check::Require(SameBits(samples, deployed->Sample(data.prior)), "latent ONLY repeat " + spec.tag + " samples and Jacobians BITEXACT");
  const auto pv = data.prior.to_vector<float>(), pr = samples.at("prior_roundtrip").to_vector<float>(),
             i0 = samples.at("inverse_jacobian_row0").to_vector<float>(),
             i1 = samples.at("inverse_jacobian_row1").to_vector<float>(), ild = samples.at("inverse_logdet").to_vector<float>();
  double prior_error = 0;
  for (int b = 0; b < 1024; ++b) {
    for (int c = 0; c < 2; ++c) prior_error = std::max(prior_error, std::abs(double(pr[b * 2 + c]) - pv[b * 2 + c]));
    const double determinant = double(i0[b * 2]) * i1[b * 2 + 1] - double(i0[b * 2 + 1]) * i1[b * 2];
    check::Require(determinant > 0, "inverse orientation preserving");
    check::Near(std::log(determinant), At(ild, b), 5e-6, "analytic INVERSE logdet matches actual Jacobian");
  }
  check::Require(prior_error < 5e-6, "forward(inverse(prior)) roundtrip");
  const auto sample_density = deployed->Evaluate(samples.at("samples"));
  const auto sdl            = sample_density.at("log_probability").to_vector<float>();
  for (int b = 0; b < 1024; ++b) {
    const double base_log = -std::log(2 * M_PI) - .5 * (std::pow(double(pv[b * 2]), 2) + std::pow(double(pv[b * 2 + 1]), 2));
    check::Near(sdl[b] + At(ild, b), base_log, 2e-5, "normalization CHANGE-OF-VARIABLES identity for all ancestral samples");
  }
  nn::save_npz((root / "samples.npz").string(), replay::ToNamed(samples));
  nn::save_npz((root / "sample_density.npz").string(), replay::ToNamed(sample_density));

  // Density grid and plot.
  std::vector<float>  grid(128 * 128 * 2);
  std::vector<double> truth(128 * 128);
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 128; ++x) {
      const int j = y * 128 + x;
      grid[j * 2] = float((x + .5) / 16 - 4), grid[j * 2 + 1] = float(4 - (y + .5) / 16);
      truth[j] = TrueLog(grid[j * 2], grid[j * 2 + 1], spec.kind);
    }
  const Tensor grid_t  = nn::from_vector(grid, {128 * 128, 2});
  const auto   density = deployed->Evaluate(grid_t);
  nn::save_npz((root / "grid.npz").string(), {{"points", grid_t},
                                              {"model_log_probability", density.at("log_probability")},
                                              {"true_log_probability", nn::from_vector(truth, {128 * 128})}});
  Plot(root / "density.png", density.at("log_probability"), truth, data.pools[2], samples.at("samples"), output.at("latent"),
       data.prior, spec.density_scale);

  // Fit-only Gaussian baselines and the analytic truth.
  const auto fv = fit.to_vector<float>();
  double     mean[2] = {0, 0}, cov[2][2] = {{0, 0}, {0, 0}};
  for (int b = 0; b < 512; ++b)
    for (int c = 0; c < 2; ++c) mean[c] += fv[b * 2 + c] / 512.;
  for (int b = 0; b < 512; ++b)
    for (int c = 0; c < 2; ++c)
      for (int d = 0; d < 2; ++d) cov[c][d] += (double(fv[b * 2 + c]) - mean[c]) * (double(fv[b * 2 + d]) - mean[d]) / 512.;
  const double det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
  check::Require(det > 0, "fitting MLE Gaussian positive covariance");
  double              full_nll = 0, diag_nll = 0, true_nll = 0, identity_nll = 0;
  std::vector<double> true_logp(128);
  for (int b = 0; b < 128; ++b) {
    const double x = test[b * 2], y = test[b * 2 + 1], a = x - mean[0], c = y - mean[1];
    true_logp[b] = TrueLog(x, y, spec.kind);
    true_nll -= true_logp[b] / 128;
    identity_nll += (std::log(2 * M_PI) + .5 * (x * x + y * y)) / 128;
    full_nll += (std::log(2 * M_PI) + .5 * std::log(det) + .5 * (a * a * cov[1][1] - 2 * a * c * cov[0][1] + c * c * cov[0][0]) / det) / 128;
    diag_nll += (std::log(2 * M_PI) + .5 * std::log(cov[0][0] * cov[1][1]) + .5 * (a * a / cov[0][0] + c * c / cov[1][1])) / 128;
  }
  nn::save_npz((root / "baseline.npz").string(),
               {{"fit_mean", nn::from_vector(std::vector<double>{mean[0], mean[1]}, {2})},
                {"fit_covariance", nn::from_vector(std::vector<double>{cov[0][0], cov[0][1], cov[1][0], cov[1][1]}, {2, 2})},
                {"true_test_log_probability", nn::from_vector(true_logp, {128})}});
  const auto single = deployed->Evaluate(nn::take_rows(data.pools[2], {0}));
  const auto sl = single.at("latent").to_vector<float>(), ol = output.at("latent").to_vector<float>();
  for (int c = 0; c < 2; ++c) check::Near(sl[c], ol[c], 2e-6, "single/batch " + spec.tag + " parity");
  check::Require(SameBits(before, deployed->Arrays()), "density/Jacobian/inverse/sampling ALL immutable");
  double min_ld = INFINITY, max_ld = -INFINITY;
  if (!spec.global_logdet) {
    const auto inv_ld = output.at("inverse_logdet").to_vector<float>();
    for (int b = 0; b < 128; ++b) {
      min_ld = std::min(min_ld, double(fld[b]));
      max_ld = std::max(max_ld, double(fld[b]));
      check::Near(fld[b] + inv_ld[b], 0, 5e-6, "forward and reconstruction inverse logdets cancel PER POINT");
    }
    check::Require(max_ld - min_ld > 1e-5, "spatially varying non-volume-preserving Jacobian");
  }

  // flow_density_2d's sample diagnostics.
  double diagnostic = 0;
  std::string diagnostics_json;
  if (spec.kind >= 0) {
    const auto           sv = samples.at("samples").to_vector<float>();
    double               support = 0, curvature = 0;
    std::array<int, 8>   mode_counts{};
    for (int b = 0; b < 1024; ++b) {
      const double x = sv[b * 2], y = sv[b * 2 + 1];
      support += CheckerSupport(x, y) / 1024.;
      curvature += y * (x * x - 1) / 1024.;
      double nearest = INFINITY;
      int    mode    = -1;
      for (int j = 0; j < 8; ++j) {
        const double d = std::hypot(x - 2 * std::cos(2 * M_PI * j / 8), y - 2 * std::sin(2 * M_PI * j / 8));
        if (d < nearest) nearest = d, mode = j;
      }
      if (nearest <= .45) ++mode_counts[mode];
    }
    int covered = 0, total_near = 0;
    for (int count : mode_counts) covered += count > 0, total_near += count;
    nn::save_npz((root / "sample_diagnostics.npz").string(),
                 {{"mode_counts_within_three_sigma", nn::from_vector(std::vector<std::int32_t>(mode_counts.begin(), mode_counts.end()), {8})}});
    std::ofstream diagnostics(root / "sample_diagnostics.json");
    diagnostics.precision(17);
    diagnostics << "{\"checkerboard_support_fraction\":" << support << ",\"eight_gaussian_fraction_within_three_sigma_of_centers\":"
                << total_near / 1024. << ",\"eight_gaussian_modes_covered_at_three_sigma\":" << covered
                << ",\"banana_curvature_moment\":" << curvature << "}\n";
    check::Require(bool(diagnostics), "sample diagnostics written");
    diagnostic = spec.kind == 0 ? curvature : spec.kind == 1 ? total_near / 1024. : support;
  }

  const double  nll = output.at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << spec.parameters << ",\"parameter_tensors\":" << spec.tensors
       << ",\"coupling_layers\":4,\"hidden\":16,\"base\":\"standardGaussian2D\",\"epochs\":" << o.epochs
       << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll\":" << nll
       << ",\"true_test_nll\":" << true_nll << ",\"fit_full_gaussian_nll\":" << full_nll << ",\"fit_diagonal_gaussian_nll\":" << diag_nll
       << ",\"identity_gaussian_nll\":" << identity_nll;
  if (spec.global_logdet) meta << ",\"forward_logdet\":" << fld[0];
  else meta << ",\"forward_logdet_min\":" << min_ld << ",\"forward_logdet_max\":" << max_ld << ",\"mean_forward_logdet\":" << Mean(output.at("forward_logdet"));
  meta << ",\"forward_inverse_max_error\":" << roundtrip << ",\"inverse_forward_max_error\":" << prior_error
       << ",\"actual_vs_analytic_logdet_max_error\":" << det_error << ",\"ancestral_samples\":1024,\"train_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>" << spec.title
       << "</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}img{max-width:100%}</style>" << spec.html
       << "<img src='density.png'><p>Panels: top left analytic true PDF, top right learned PDF (fixed density scale 0.." << spec.density_scale
       << "); bottom left held-out orange and generated blue, bottom right transformed held-out purple and prior gray. Axes [-4, 4], "
          "outside points hidden only for the plot. See metrics.json, history.tsv and source_protocol.json. Short training can "
          "remain Gaussian-like; no fit-quality guarantee.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete " + spec.tag + " records");
  std::cout << (spec.kind >= 0 ? "dataset=" + std::to_string(spec.kind) + "," : std::string()) << "testNLL=" << nll << ",true=" << true_nll
            << ",fitGaussian=" << full_nll << ",inverse/Jacobianerrors=" << roundtrip << '/' << det_error << ";"
            << (spec.numerical ? "numerical" : "exact") << "inverse/logdet/normalization/restore/1024samples/immutable PASS\n";

  // ---- TF parity ----
  const auto close = [](double d, double tol) { return d >= 0 && d <= tol; };
  for (const auto& f : spec.extra_files) Compare(tf, root, f, o.verbose);
  if (!init.empty() && init == tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << tf.string() << "):\n";
    double first_diff = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz"})
      first_diff = std::max(first_diff, Compare(tf, root, f, o.verbose));
    check::Require(close(first_diff, 2e-5), "replayed initial outputs/first forward/gradients match TF");
    check::Require(close(Compare(tf, root, "first_after.npz", o.verbose), 1e-4), "first Adam update matches TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-30s max |ours - TF| validation NLL = %.3g\n", "history.tsv", hd);
      for (auto f : {"epoch3_arrays.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz", "samples.npz", "sample_density.npz",
                     "grid.npz", "baseline.npz"})
        Compare(tf, root, f, o.verbose);
      check::Require(hd < 2e-3, "replayed validation history follows TF's");
    }
  }
  if (fs::exists(tf / "model_arrays.npz") && fs::exists(tf / "outputs.npz")) {
    auto fresh = spec.make(false);
    replay::LoadInto(fresh->store, tf / "model_arrays.npz");
    std::cout << "from TF's final weights:\n";
    const auto   out = fresh->Evaluate(data.pools[2]);
    const double d   = Compare(tf, "outputs.npz", out, o.verbose);
    const double s   = Compare(tf, "samples.npz", fresh->Sample(data.prior), o.verbose);
    check::Require(close(d, 1e-4) && close(s, 1e-4), "TF's final weights reproduce TF's test outputs and samples");
    std::printf("  test NLL %.8f (TF %.8f)\n", out.at("loss").item<float>(),
                replay::Get(nn::load_npz((tf / "outputs.npz").string()), "loss").item<float>());
  }
  std::cout << "training " << train_seconds << " s, wall "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
  return {nll, true_nll, full_nll, diagnostic};
}
}  // namespace flowd



using namespace flowd;

namespace {

void MixingFixture(const fs::path& root) {
  nn::ParameterStore p(1);
  const Tensor       input = nn::from_vector(std::vector<float>{.2f, -.7f, 1.1f, .4f, -.3f, 2.f}, {3, 2});
  GlowMix(p, input, 0);
  const Tensor matrix = nn::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2});
  p.set("mix0/W", matrix);
  const auto forward = GlowMix(p, input, 0), reverse = GlowMix(p, forward.value, 0, true);
  const auto gradient = nn::grad(nn::sum(nn::square(forward.value)) - forward.logdet, {p.at("mix0/W")})[0];
  check::Near(forward.logdet.item<float>(), std::log(2.), 1e-6, "negative determinant log ABS");
  check::Near(reverse.logdet.item<float>(), -std::log(2.), 1e-6, "negative determinant inverse log ABS");
  const auto r = reverse.value.to_vector<float>(), x = input.to_vector<float>();
  for (int j = 0; j < 6; ++j) check::Near(r[j], x[j], 2e-6, "negative determinant roundtrip");
  nn::save_npz((root / "signed_mixing_fixture.npz").string(),
               {{"input", input}, {"matrix", matrix}, {"forward", forward.value.detach()}, {"inverse", reverse.value.detach()},
                {"forward_logabsdet", forward.logdet.detach()}, {"inverse_logabsdet", reverse.logdet.detach()}, {"gradient", gradient}});
}

}  // namespace

int main(int argc, char** argv) {
  return Main(argc, argv, "glow_density", [](const Options& o) {
    Spec s;
    s.name = "glow_density", s.tag = "Glow", s.title = "Glow density";
    s.parameters = 296, s.tensors = 28, s.batch_seed = 9521;
    s.protocol    = BananaProtocol();
    s.extra_files = {"signed_mixing_fixture.npz", "calibration.npz"};
    s.before      = MixingFixture;
    s.make        = [](bool training) { return std::make_unique<GlowDensity>(training); };
    s.replayed    = [](FlowModel& model, const Data& data) {
      auto&      glow = dynamic_cast<GlowDensity&>(model);
      const auto tf   = glow.Arrays();
      glow.Calibrate(nn::take_rows(data.pools[0], {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                   23, 24, 25, 26, 27, 28, 29, 30, 31}));
      double d = 0;
      for (const auto& [k, v] : glow.Arrays())
        if (k.rfind("actnorm", 0) == 0) d = std::max(d, replay::MaxAbsDiff(v, tf.at(k)));
      std::printf("ActNorm re-derived from TF's initial weights: max |ours - TF| = %.3g\n", d);
      check::Require(d < 1e-5, "ActNorm data-dependent initialisation reproduces TF's");
      for (const auto& [k, v] : tf) glow.store.set(k, v);
    };
    s.html =
        "<h1>Glow: ActNorm, invertible mixing and coupling</h1><p><a href='https://arxiv.org/abs/1807.03039'>Glow</a> "
        "steps for two channels: ActNorm, learned invertible 2x2 mixing (the 1x1 convolution of one pixel) and an affine "
        "coupling, four steps; all four mixing entries are trainable. Exact log|det| includes ActNorm scales, the matrix "
        "determinant and the input-dependent coupling scale; the analytic inverse reverses coupling, matrix and ActNorm. 296 "
        "parameters / 28 tensors.</p><p>ActNorm initialises ONCE from the FIRST 32 fitting points, sequentially after the "
        "previous initialised steps: bias = -mean, log scale = -.5 log(population variance + 1e-6). A two-channel 2-D CPU "
        "teaching adaptation, not full multiscale image Glow. Same banana and three epochs (48 Adam .003 steps); bounded "
        "coupling log scale .8 tanh(raw).</p>";
    Run(s, o);
  });
}
