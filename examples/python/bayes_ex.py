# Note: the C++ bayes_ex.cpp builds its models with datamunge::bayes::AutodiffModel, which
# wraps a Tape/Var lambda for automatic differentiation of the log-posterior -- but
# AutodiffModel is explicitly C++-only (never bound via SWIG, in any language), since a
# scripting-language callback can't be traced by the C++ autodiff tape. This port instead
# subclasses optim::DifferentiableFunction directly (a director, available in this binding)
# and supplies the log-posterior and its gradient by hand -- exactly what AutodiffModel would
# have computed automatically. MAP/HMC/NUTS consume this identically either way.
import math

from pydatamunge import datamunge as dm


def normal_lpdf(x, mean, sd):
    return -0.5 * math.log(2.0 * math.pi * sd * sd) - (x - mean) ** 2 / (2.0 * sd * sd)


def normal_dlpdf(x, mean, sd):
    return -(x - mean) / (sd * sd)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def bernoulli_logit_lpmf(y, eta):
    # log(1+exp(eta)), computed in a numerically stable way.
    return y * eta - (eta + math.log1p(math.exp(-eta)) if eta > 0 else math.log1p(math.exp(eta)))


def mean_of(samples, dim):
    return sum(row[dim] for row in samples) / len(samples)


def sd_of(samples, dim, mean):
    s = sum((row[dim] - mean) ** 2 for row in samples)
    return math.sqrt(s / (len(samples) - 1))


class ConjugateModel(dm.DifferentiableFunction):
    def __init__(self, y, mu0, tau0, sigma):
        super().__init__()
        self.y, self.mu0, self.tau0, self.sigma = y, mu0, tau0, sigma

    def evaluate(self, params):
        mu = params[0]
        lp = normal_lpdf(mu, self.mu0, self.tau0)
        for yi in self.y:
            lp += normal_lpdf(yi, mu, self.sigma)
        return lp

    def gradient(self, params):
        mu = params[0]
        d = normal_dlpdf(mu, self.mu0, self.tau0)
        for yi in self.y:
            d += -normal_dlpdf(yi, mu, self.sigma)  # d/dmu of normal_lpdf(yi; mu, sigma) = (yi-mu)/sigma^2
        return [d]


class LogisticModel(dm.DifferentiableFunction):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y

    def evaluate(self, params):
        b0, b1 = params
        lp = normal_lpdf(b0, 0.0, 10.0) + normal_lpdf(b1, 0.0, 10.0)
        for xi, yi in zip(self.x, self.y):
            lp += bernoulli_logit_lpmf(yi, b0 + b1 * xi)
        return lp

    def gradient(self, params):
        b0, b1 = params
        d0 = normal_dlpdf(b0, 0.0, 10.0)
        d1 = normal_dlpdf(b1, 0.0, 10.0)
        for xi, yi in zip(self.x, self.y):
            resid = yi - sigmoid(b0 + b1 * xi)
            d0 += resid
            d1 += resid * xi
        return [d0, d1]


print("=================== Normal-Normal conjugate model: MAP, HMC, NUTS vs. the exact posterior "
      "===================")
y = [2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95]
sigma, mu0, tau0 = 1.0, 0.0, 5.0

n = len(y)
precision_post = 1.0 / (tau0 * tau0) + n / (sigma * sigma)
exact_mean = (mu0 / (tau0 * tau0) + sum(y) / (sigma * sigma)) / precision_post
exact_sd = math.sqrt(1.0 / precision_post)
print(f"Exact posterior: N({exact_mean}, {exact_sd}^2)\n")

conjugate_model = ConjugateModel(y, mu0, tau0, sigma)

# MAP().optimize() mutates its coordinates argument in place, so it needs a real DVector
# (a plain Python list works for by-value/const-ref arguments, but not this in/out one).
coords = dm.DVector([0.0])
log_post = dm.MAP().optimize(conjugate_model, coords)
print(f"MAP:  mu = {coords[0]} (log-posterior = {log_post})")

hmc_options = dm.HMCOptions()
hmc_options.num_warmup, hmc_options.num_samples = 1000, 4000
hmc_options.num_leapfrog_steps, hmc_options.initial_step_size = 15, 0.3
result = dm.HMC(hmc_options).sample(conjugate_model, [0.0])
m, s = mean_of(result.samples, 0), 0.0
s = sd_of(result.samples, 0, m)
print(f"HMC:  mu ~ N({m}, {s}^2), accept rate = {result.accept_rate}, step size = {result.final_step_size}")

nuts_options = dm.NUTSOptions()
nuts_options.num_warmup, nuts_options.num_samples, nuts_options.initial_step_size = 1000, 4000, 0.3
result = dm.NUTS(nuts_options).sample(conjugate_model, [0.0])
m = mean_of(result.samples, 0)
s = sd_of(result.samples, 0, m)
print(f"NUTS: mu ~ N({m}, {s}^2), accept rate = {result.accept_rate}, step size = {result.final_step_size}, "
      f"divergences = {result.num_divergences}")

print("\n=================== Bayesian logistic regression vs. GLM's MLE (iris) ===================")
iris = dm.DataFrame.iris()
is_virginica, petal_length = [], []
for i in range(iris.nrows()):
    species = iris.string_at("Species", i)
    if species not in ("versicolor", "virginica"):
        continue
    is_virginica.append(1.0 if species == "virginica" else 0.0)
    petal_length.append(iris.numeric_at("Petal.Length", i))

df = dm.DataFrame()
df.add_numeric_column("Petal.Length", petal_length)
df.add_numeric_column("is_virginica", is_virginica)
glm = dm.GLM(df, "is_virginica ~ Petal.Length", "binomial")
print("GLM MLE:        ", list(glm.coefficients()))

logistic_model = LogisticModel(petal_length, is_virginica)
coords = dm.DVector([0.0, 0.0])
dm.MAP().optimize(logistic_model, coords)
print("Bayes MAP:      ", list(coords), "(weak Normal(0, 10) priors)")

nuts_options = dm.NUTSOptions()
nuts_options.num_warmup, nuts_options.num_samples, nuts_options.initial_step_size = 1000, 3000, 0.05
result = dm.NUTS(nuts_options).sample(logistic_model, [0.0, 0.0])
print(f"Bayes NUTS mean: [{mean_of(result.samples, 0)}, {mean_of(result.samples, 1)}] "
      f"(posterior mean, accept rate = {result.accept_rate})")
print("(Petal.Length nearly separates these two species, so the unregularized MLE inflates toward the\n"
      " separating boundary; the weak Normal(0, 10) prior visibly pulls the Bayesian estimate back --\n"
      " a real, expected difference, not a bug.)")
