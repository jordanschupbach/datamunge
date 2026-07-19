require "octruby"

# Note: the C++ bayes_ex.cpp builds its models with datamunge::bayes::AutodiffModel, which
# wraps a Tape/Var lambda for automatic differentiation of the log-posterior -- but
# AutodiffModel is explicitly C++-only (never bound via SWIG, in any language), since a
# scripting-language callback can't be traced by the C++ autodiff tape. This port instead
# subclasses optim::DifferentiableFunction directly (a director, available in this binding)
# and supplies the log-posterior and its gradient by hand -- exactly what AutodiffModel would
# have computed automatically. MAP/HMC/NUTS consume this identically either way.

def normal_lpdf(x, mean, sd)
  -0.5 * Math.log(2.0 * Math::PI * sd * sd) - (x - mean)**2 / (2.0 * sd * sd)
end

def normal_dlpdf(x, mean, sd)
  -(x - mean) / (sd * sd)
end

def sigmoid(z)
  1.0 / (1.0 + Math.exp(-z))
end

def log1p(x)
  # Ruby's Math module has no log1p; this is precise enough for the small |x| used here.
  Math.log(1.0 + x)
end

def bernoulli_logit_lpmf(y, eta)
  # log(1+exp(eta)), computed in a numerically stable way.
  y * eta - (eta > 0 ? eta + log1p(Math.exp(-eta)) : log1p(Math.exp(eta)))
end

def mean_of(samples, dim)
  (0...samples.size).sum { |i| samples[i][dim] } / samples.size
end

def sd_of(samples, dim, mean)
  s = (0...samples.size).sum { |i| (samples[i][dim] - mean)**2 }
  Math.sqrt(s / (samples.size - 1))
end

class ConjugateModel < Datamunge::DifferentiableFunction
  def initialize(y, mu0, tau0, sigma)
    super()
    @y = y
    @mu0 = mu0
    @tau0 = tau0
    @sigma = sigma
  end

  def evaluate(params)
    mu = params[0]
    lp = normal_lpdf(mu, @mu0, @tau0)
    @y.each { |yi| lp += normal_lpdf(yi, mu, @sigma) }
    lp
  end

  def gradient(params)
    mu = params[0]
    d = normal_dlpdf(mu, @mu0, @tau0)
    @y.each { |yi| d += -normal_dlpdf(yi, mu, @sigma) }  # d/dmu of normal_lpdf(yi; mu, sigma) = (yi-mu)/sigma^2
    [d]
  end
end

class LogisticModel < Datamunge::DifferentiableFunction
  def initialize(x, y)
    super()
    @x = x
    @y = y
  end

  def evaluate(params)
    b0, b1 = params
    lp = normal_lpdf(b0, 0.0, 10.0) + normal_lpdf(b1, 0.0, 10.0)
    @x.zip(@y).each { |xi, yi| lp += bernoulli_logit_lpmf(yi, b0 + b1 * xi) }
    lp
  end

  def gradient(params)
    b0, b1 = params
    d0 = normal_dlpdf(b0, 0.0, 10.0)
    d1 = normal_dlpdf(b1, 0.0, 10.0)
    @x.zip(@y).each do |xi, yi|
      resid = yi - sigmoid(b0 + b1 * xi)
      d0 += resid
      d1 += resid * xi
    end
    [d0, d1]
  end
end

puts "=================== Normal-Normal conjugate model: MAP, HMC, NUTS vs. the exact posterior ===================\n"
y = [2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95]
sigma = 1.0
mu0 = 0.0
tau0 = 5.0

n = y.length
precision_post = 1.0 / (tau0 * tau0) + n / (sigma * sigma)
exact_mean = (mu0 / (tau0 * tau0) + y.sum / (sigma * sigma)) / precision_post
exact_sd = Math.sqrt(1.0 / precision_post)
puts "Exact posterior: N(#{exact_mean}, #{exact_sd}^2)\n\n"

conjugate_model = ConjugateModel.new(y, mu0, tau0, sigma)

# MAP().optimize() mutates its coordinates argument in place, so it needs a real DVector
# (a plain Ruby array works for by-value/const-ref arguments, but not this in/out one).
coords = Datamunge::DVector.new([0.0])
log_post = Datamunge::MAP.new.optimize(conjugate_model, coords)
puts "MAP:  mu = #{coords[0]} (log-posterior = #{log_post})"

hmc_options = Datamunge::HMCOptions.new
hmc_options.num_warmup = 1000
hmc_options.num_samples = 4000
hmc_options.num_leapfrog_steps = 15
hmc_options.initial_step_size = 0.3
result = Datamunge::HMC.new(hmc_options).sample(conjugate_model, [0.0])
m = mean_of(result.samples, 0)
s = sd_of(result.samples, 0, m)
puts "HMC:  mu ~ N(#{m}, #{s}^2), accept rate = #{result.accept_rate}, step size = #{result.final_step_size}"

nuts_options = Datamunge::NUTSOptions.new
nuts_options.num_warmup = 1000
nuts_options.num_samples = 4000
nuts_options.initial_step_size = 0.3
result = Datamunge::NUTS.new(nuts_options).sample(conjugate_model, [0.0])
m = mean_of(result.samples, 0)
s = sd_of(result.samples, 0, m)
puts "NUTS: mu ~ N(#{m}, #{s}^2), accept rate = #{result.accept_rate}, step size = #{result.final_step_size}, " \
     "divergences = #{result.num_divergences}"

puts "\n=================== Bayesian logistic regression vs. GLM's MLE (iris) ===================\n"
iris = Datamunge::DataFrame.iris
is_virginica = []
petal_length = []
iris.nrows.times do |i|
  species = iris.string_at("Species", i)
  next unless ["versicolor", "virginica"].include?(species)
  is_virginica << (species == "virginica" ? 1.0 : 0.0)
  petal_length << iris.numeric_at("Petal.Length", i)
end

df = Datamunge::DataFrame.new
df.add_numeric_column("Petal.Length", petal_length)
df.add_numeric_column("is_virginica", is_virginica)
glm = Datamunge::GLM.new(df, "is_virginica ~ Petal.Length", "binomial")
puts "GLM MLE:         #{glm.coefficients.to_a}"

logistic_model = LogisticModel.new(petal_length, is_virginica)
coords = Datamunge::DVector.new([0.0, 0.0])
Datamunge::MAP.new.optimize(logistic_model, coords)
puts "Bayes MAP:       #{coords.to_a} (weak Normal(0, 10) priors)"

nuts_options = Datamunge::NUTSOptions.new
nuts_options.num_warmup = 1000
nuts_options.num_samples = 3000
nuts_options.initial_step_size = 0.05
result = Datamunge::NUTS.new(nuts_options).sample(logistic_model, [0.0, 0.0])
puts "Bayes NUTS mean: [#{mean_of(result.samples, 0)}, #{mean_of(result.samples, 1)}] " \
     "(posterior mean, accept rate = #{result.accept_rate})"
puts "(Petal.Length nearly separates these two species, so the unregularized MLE inflates toward the\n" \
     " separating boundary; the weak Normal(0, 10) prior visibly pulls the Bayesian estimate back --\n" \
     " a real, expected difference, not a bug.)"
