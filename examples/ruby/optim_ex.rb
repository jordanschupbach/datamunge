require "octruby"

# ---- Function types, defined via SWIG directors (Ruby subclasses of the bound C++
# ---- interfaces) ----

class QuadraticBowl < Datamunge::DifferentiableFunction
  # A plain 3-D quadratic bowl with a known minimum.
  def initialize(target)
    super()
    @target = target
  end

  def evaluate(coordinates)
    coordinates.zip(@target).map { |c, t| (c - t)**2 }.sum
  end

  def gradient(coordinates)
    coordinates.zip(@target).map { |c, t| 2.0 * (c - t) }
  end
end

class RosenbrockFn < Datamunge::DifferentiableFunction
  # The classic Rosenbrock "banana" function -- a much harder landscape.
  def evaluate(x)
    a = 1.0 - x[0]
    b = x[1] - x[0] * x[0]
    a * a + 100.0 * b * b
  end

  def gradient(x)
    [-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]), 200.0 * (x[1] - x[0] * x[0])]
  end
end

class LinearRegressionLoss < Datamunge::DifferentiableSeparableFunction
  # Ordinary least squares as a sum of per-example losses -- the textbook case for SGD.
  def initialize(x, y)
    super()
    @x = x
    @y = y
  end

  def num_functions
    @y.length
  end

  def predict(w, i)
    (0...w.length).sum { |j| w[j] * @x[i][j] }
  end

  def evaluate_term(w, i)
    err = predict(w, i) - @y[i]
    err * err
  end

  def gradient_term(w, i)
    err = predict(w, i) - @y[i]
    (0...w.length).map { |j| 2.0 * err * @x[i][j] }
  end
end

class BumpyFunction < Datamunge::ArbitraryFunction
  # A bumpy, multimodal landscape -- no gradient available, so only a derivative-free
  # method (SimulatedAnnealing) can be used here.
  def evaluate(x)
    bowl = (x[0] - 3.0)**2 + (x[1] + 1.0)**2
    ripples = 5.0 * Math.sin(x[0]) * Math.cos(x[1])
    bowl + ripples
  end
end

class RastriginFunction < Datamunge::ArbitraryFunction
  # The classic Rastrigin function -- highly multimodal (many local minima arranged in a
  # regular grid), global minimum f=0 at the origin. A standard torture test for
  # population-based methods, since local/gradient-based methods get stuck in the first
  # basin they land in.
  def evaluate(x)
    total = 10.0 * x.length
    x.each { |xi| total += xi * xi - 10.0 * Math.cos(2.0 * Math::PI * xi) }
    total
  end
end

# optimize() mutates its `coordinates` argument in place, so it needs a real DVector (a
# plain Ruby array works for arguments read by value, but not for this in/out one).
def dv(*values)
  Datamunge::DVector.new(values)
end

puts "=================== DifferentiableFunction: three optimizers, one bowl ==================="
target = [4.0, -2.0, 1.0]

f = QuadraticBowl.new(target)
x = dv(0.0, 0.0, 0.0)
gd_options = Datamunge::GradientDescentOptions.new
gd_options.step_size = 0.1
gd_options.momentum = 0.0
gd_options.max_iterations = 1000
gd_options.tolerance = 1e-10
value = Datamunge::GradientDescent.new(gd_options).optimize(f, x)
puts "GradientDescent: f=#{value} x=#{x.to_a}"

f = QuadraticBowl.new(target)
x = dv(0.0, 0.0, 0.0)
value = Datamunge::Adam.new.optimize(f, x)
puts "Adam:             f=#{value} x=#{x.to_a}"

f = QuadraticBowl.new(target)
x = dv(0.0, 0.0, 0.0)
value = Datamunge::LBFGS.new.optimize(f, x)
puts "LBFGS:            f=#{value} x=#{x.to_a} (converges in far fewer iterations)"

puts "\n=================== LBFGS on the Rosenbrock function ==================="
f = RosenbrockFn.new
x = dv(-1.2, 1.0)
value = Datamunge::LBFGS.new.optimize(f, x)
puts "f=#{value} x=#{x.to_a} (true minimum: f=0 at [1, 1])"

puts "\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================\n"
iris = Datamunge::DataFrame.iris
sepal_length = (0...iris.nrows).map { |i| iris.numeric_at("Sepal.Length", i) }
sepal_width = (0...iris.nrows).map { |i| iris.numeric_at("Sepal.Width", i) }
petal_length = (0...iris.nrows).map { |i| iris.numeric_at("Petal.Length", i) }

# SGD with a single constant step size converges far faster (and far more reliably) on
# standardized features -- unnormalized predictors of very different scales give the loss an
# ill-conditioned Hessian, which plain constant-step SGD handles poorly. Standard practice.
mean_of = ->(v) { v.sum / v.length }
stddev_of = ->(v, mean) { Math.sqrt(v.sum { |x| (x - mean)**2 } / v.length) }

mean1 = mean_of.call(sepal_length)
std1 = stddev_of.call(sepal_length, mean1)
mean2 = mean_of.call(sepal_width)
std2 = stddev_of.call(sepal_width, mean2)

x_std = (0...iris.nrows).map { |i| [1.0, (sepal_length[i] - mean1) / std1, (sepal_width[i] - mean2) / std2] }
loss = LinearRegressionLoss.new(x_std, petal_length)

w_std = dv(0.0, 0.0, 0.0)
sgd_options = Datamunge::SGDOptions.new
sgd_options.step_size = 0.01
sgd_options.max_epochs = 300
sgd_options.batch_size = 8
Datamunge::SGD.new(sgd_options).optimize(loss, w_std)

# Convert the standardized-space weights back to the original feature scale.
w = [w_std[0] - w_std[1] * mean1 / std1 - w_std[2] * mean2 / std2, w_std[1] / std1, w_std[2] / std2]
puts "SGD weights (intercept, Sepal.Length, Sepal.Width): #{w}"

lm = Datamunge::LM.new(iris, "Petal.Length ~ Sepal.Length + Sepal.Width")
puts "LM  weights (intercept, Sepal.Length, Sepal.Width): #{lm.coefficients.to_a} (closed-form OLS, for comparison)"

puts "\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================\n"
f = BumpyFunction.new
x = dv(0.0, 0.0)
sa_options = Datamunge::SimulatedAnnealingOptions.new
sa_options.initial_temperature = 10.0
sa_options.cooling_rate = 0.999
sa_options.max_iterations = 20_000
sa_options.step_std_dev = 0.5
value = Datamunge::SimulatedAnnealing.new(sa_options).optimize(f, x)
puts "f=#{value} x=#{x.to_a} (found without ever computing a gradient)"

puts "\n=================== Population-based methods on the Rastrigin function ===================\n"
lower = [-5.12, -5.12]
upper = [5.12, 5.12]

f = RastriginFunction.new
x = dv(3.0, -4.0)
pso_options = Datamunge::PSOOptions.new
pso_options.topology = "global"
pso_options.inertia_strategy = "constant"
value = Datamunge::PSO.new(pso_options).optimize(f, x, lower, upper)
puts "PSO (global topology, constant inertia):    f=#{value} x=#{x.to_a}"

f = RastriginFunction.new
x = dv(3.0, -4.0)
pso_options = Datamunge::PSOOptions.new
pso_options.topology = "ring"
pso_options.inertia_strategy = "linear_decay"
value = Datamunge::PSO.new(pso_options).optimize(f, x, lower, upper)
puts "PSO (ring topology, linear-decay inertia):  f=#{value} x=#{x.to_a}"

f = RastriginFunction.new
x = dv(3.0, -4.0)
de_options = Datamunge::DEOptions.new
de_options.mutation_strategy = "rand1"
de_options.crossover_strategy = "binomial"
value = Datamunge::DifferentialEvolution.new(de_options).optimize(f, x, lower, upper)
puts "DE (rand1/binomial):                        f=#{value} x=#{x.to_a}"

f = RastriginFunction.new
x = dv(3.0, -4.0)
de_options = Datamunge::DEOptions.new
de_options.mutation_strategy = "best1"
de_options.crossover_strategy = "exponential"
value = Datamunge::DifferentialEvolution.new(de_options).optimize(f, x, lower, upper)
puts "DE (best1/exponential):                     f=#{value} x=#{x.to_a}"

f = RastriginFunction.new
x = dv(3.0, -4.0)
ga_options = Datamunge::GAOptions.new
ga_options.selection_strategy = "tournament"
ga_options.crossover_strategy = "blend"
value = Datamunge::GeneticAlgorithm.new(ga_options).optimize(f, x, lower, upper)
puts "GA (tournament/blend, elitism on):           f=#{value} x=#{x.to_a}"

f = RastriginFunction.new
x = dv(3.0, -4.0)
ga_options = Datamunge::GAOptions.new
ga_options.selection_strategy = "rank"
ga_options.crossover_strategy = "uniform"
ga_options.elitism = false
value = Datamunge::GeneticAlgorithm.new(ga_options).optimize(f, x, lower, upper)
puts "GA (rank/uniform, elitism off):              f=#{value} x=#{x.to_a} (true minimum: f=0 at [0, 0])"
