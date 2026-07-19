module app;

import std.stdio : writeln;
import std.math : log, exp, PI, sqrt, log1p;
import datamunge;

// Note: the C++ bayes_ex.cpp builds its models with datamunge::bayes::AutodiffModel, which
// wraps a Tape/Var lambda for automatic differentiation of the log-posterior -- but
// AutodiffModel is explicitly C++-only (never bound via SWIG, in any language), since a
// scripting-language callback can't be traced by the C++ autodiff tape. This port instead
// subclasses optim::DifferentiableFunction directly (a director, available in this binding)
// and supplies the log-posterior and its gradient by hand -- exactly what AutodiffModel would
// have computed automatically. MAP/HMC/NUTS consume this identically either way.

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

double normal_lpdf(double x, double mean, double sd) {
  return -0.5 * log(2.0 * PI * sd * sd) - (x - mean) ^^ 2 / (2.0 * sd * sd);
}

double normal_dlpdf(double x, double mean, double sd) {
  return -(x - mean) / (sd * sd);
}

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double bernoulli_logit_lpmf(double y, double eta) {
  return y * eta - (eta > 0 ? eta + log1p(exp(-eta)) : log1p(exp(eta)));
}

double mean_of(DVectorVector samples, size_t dim) {
  double s = 0.0;
  for (size_t i = 0; i < samples.size(); i++) s += samples[i][dim];
  return s / samples.size();
}

double sd_of(DVectorVector samples, size_t dim, double mean) {
  double s = 0.0;
  for (size_t i = 0; i < samples.size(); i++) s += (samples[i][dim] - mean) ^^ 2;
  return sqrt(s / (samples.size() - 1));
}

class ConjugateModel : DifferentiableFunction {
  double[] y;
  double mu0, tau0, sigma;
  this(double[] yData, double mu0_, double tau0_, double sigma_) {
    super();
    y = yData; mu0 = mu0_; tau0 = tau0_; sigma = sigma_;
  }
  override double evaluate(DVector params) {
    double mu = params[0];
    double lp = normal_lpdf(mu, mu0, tau0);
    foreach (yi; y) lp += normal_lpdf(yi, mu, sigma);
    return lp;
  }
  override DVector gradient(DVector params) {
    double mu = params[0];
    double d = normal_dlpdf(mu, mu0, tau0);
    foreach (yi; y) d += -normal_dlpdf(yi, mu, sigma);
    auto g = new DVector();
    g.push_back(d);
    return g;
  }
}

class LogisticModel : DifferentiableFunction {
  double[] xs;
  double[] ys;
  this(double[] xData, double[] yData) { super(); xs = xData; ys = yData; }
  override double evaluate(DVector params) {
    double b0 = params[0];
    double b1 = params[1];
    double lp = normal_lpdf(b0, 0.0, 10.0) + normal_lpdf(b1, 0.0, 10.0);
    foreach (i, xi; xs) lp += bernoulli_logit_lpmf(ys[i], b0 + b1 * xi);
    return lp;
  }
  override DVector gradient(DVector params) {
    double b0 = params[0];
    double b1 = params[1];
    double d0 = normal_dlpdf(b0, 0.0, 10.0);
    double d1 = normal_dlpdf(b1, 0.0, 10.0);
    foreach (i, xi; xs) {
      double resid = ys[i] - sigmoid(b0 + b1 * xi);
      d0 += resid;
      d1 += resid * xi;
    }
    auto g = new DVector();
    g.push_back(d0);
    g.push_back(d1);
    return g;
  }
}

void main() {
  writeln("=================== Normal-Normal conjugate model: MAP, HMC, NUTS vs. the exact posterior ===================");
  double[] y = [2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95];
  double sigma = 1.0;
  double mu0 = 0.0;
  double tau0 = 5.0;

  auto n = y.length;
  double precision_post = 1.0 / (tau0 * tau0) + n / (sigma * sigma);
  double y_sum = 0.0;
  foreach (v; y) y_sum += v;
  double exact_mean = (mu0 / (tau0 * tau0) + y_sum / (sigma * sigma)) / precision_post;
  double exact_sd = sqrt(1.0 / precision_post);
  writeln("Exact posterior: N(", exact_mean, ", ", exact_sd, "^2)\n");

  auto conjugate_model = new ConjugateModel(y, mu0, tau0, sigma);

  auto coords = dv([0.0]);
  double log_post = (new MAP()).optimize(conjugate_model, coords);
  writeln("MAP:  mu = ", coords[0], " (log-posterior = ", log_post, ")");

  auto hmc_options = new HMCOptions();
  hmc_options.num_warmup = 1000;
  hmc_options.num_samples = 4000;
  hmc_options.num_leapfrog_steps = 15;
  hmc_options.initial_step_size = 0.3;
  auto result = (new HMC(hmc_options)).sample(conjugate_model, dv([0.0]));
  double m = mean_of(result.samples(), 0);
  double s = sd_of(result.samples(), 0, m);
  writeln("HMC:  mu ~ N(", m, ", ", s, "^2), accept rate = ", result.accept_rate(), ", step size = ", result.final_step_size());

  auto nuts_options = new NUTSOptions();
  nuts_options.num_warmup = 1000;
  nuts_options.num_samples = 4000;
  nuts_options.initial_step_size = 0.3;
  auto result2 = (new NUTS(nuts_options)).sample(conjugate_model, dv([0.0]));
  m = mean_of(result2.samples(), 0);
  s = sd_of(result2.samples(), 0, m);
  writeln("NUTS: mu ~ N(", m, ", ", s, "^2), accept rate = ", result2.accept_rate(), ", step size = ", result2.final_step_size(),
          ", divergences = ", result2.num_divergences());

  writeln("\n=================== Bayesian logistic regression vs. GLM's MLE (iris) ===================");
  auto iris = DataFrame.iris();
  double[] is_virginica;
  double[] petal_length;
  auto ni = iris.nrows();
  for (size_t i = 0; i < ni; i++) {
    auto species = iris.string_at("Species", i);
    if (species != "versicolor" && species != "virginica") continue;
    is_virginica ~= (species == "virginica" ? 1.0 : 0.0);
    petal_length ~= iris.numeric_at("Petal.Length", i);
  }

  auto df = new DataFrame();
  df.add_numeric_column("Petal.Length", dv(petal_length));
  df.add_numeric_column("is_virginica", dv(is_virginica));
  auto glm = new GLM(df, "is_virginica ~ Petal.Length", "binomial");
  auto glmCoefs = glm.coefficients();
  double[] glmArr;
  for (size_t i = 0; i < glmCoefs.size(); i++) glmArr ~= glmCoefs[i];
  writeln("GLM MLE:         ", glmArr);

  auto logistic_model = new LogisticModel(petal_length, is_virginica);
  auto coords2 = dv([0.0, 0.0]);
  (new MAP()).optimize(logistic_model, coords2);
  writeln("Bayes MAP:       [", coords2[0], ",", coords2[1], "] (weak Normal(0, 10) priors)");

  auto nuts_options2 = new NUTSOptions();
  nuts_options2.num_warmup = 1000;
  nuts_options2.num_samples = 3000;
  nuts_options2.initial_step_size = 0.05;
  auto result3 = (new NUTS(nuts_options2)).sample(logistic_model, dv([0.0, 0.0]));
  writeln("Bayes NUTS mean: [", mean_of(result3.samples(), 0), ", ", mean_of(result3.samples(), 1),
          "] (posterior mean, accept rate = ", result3.accept_rate(), ")");
  writeln("(Petal.Length nearly separates these two species, so the unregularized MLE inflates toward the\n" ~
          " separating boundary; the weak Normal(0, 10) prior visibly pulls the Bayesian estimate back --\n" ~
          " a real, expected difference, not a bug.)");
}
