require "octruby"

# Exercises three of the ten new datamunge::optim classes:
#   1. HessianFunction (subclassed here, a director callback) run through Newton.
#   2. ResidualFunction (subclassed here) run through LevenbergMarquardt for a nonlinear
#      curve fit.
#   3. BayesianOptimization using the ready-to-use RBFGaussianProcessSurrogate (no
#      subclassing needed for the surrogate itself).

def dv(*values)
  Datamunge::DVector.new(values)
end

puts "=================== HessianFunction: Newton on a 2D quadratic bowl ==================="

class QuadraticBowl2D < Datamunge::HessianFunction
  # f(x, y) = (x - a)^2 + 3*(y - b)^2, minimum at (a, b).
  def initialize(a, b)
    super()
    @a = a
    @b = b
  end

  def evaluate(coordinates)
    x, y = coordinates
    (x - @a)**2 + 3.0 * (y - @b)**2
  end

  def gradient(coordinates)
    x, y = coordinates
    [2.0 * (x - @a), 6.0 * (y - @b)]
  end

  def hessian(_coordinates)
    [[2.0, 0.0], [0.0, 6.0]]
  end
end

f = QuadraticBowl2D.new(5.0, -3.0)
x = dv(0.0, 0.0)
value = Datamunge::Newton.new.optimize(f, x)
puts "Newton: f=#{value} x=#{x.to_a} (true minimum: f=0 at [5, -3])"

puts "\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================\n"

# Fit y = A * exp(-k * t) + c to synthetic noiseless data generated from known parameters.
A_true = 2.5
k_true = 0.75
c_true = 0.3

t_data = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y_data = t_data.map { |t| A_true * Math.exp(-k_true * t) + c_true }

class ExpDecayResidual < Datamunge::ResidualFunction
  def initialize(t, y)
    super()
    @t = t
    @y = y
  end

  def residuals(coordinates)
    a, k, c = coordinates
    @t.each_with_index.map { |t, i| (a * Math.exp(-k * t) + c) - @y[i] }
  end

  def jacobian(coordinates)
    a, k, c = coordinates
    @t.map do |t|
      e = Math.exp(-k * t)
      [e, -a * t * e, 1.0]
    end
  end
end

residual_fn = ExpDecayResidual.new(t_data, y_data)
params = dv(1.0, 0.1, 0.0)
lm_value = Datamunge::LevenbergMarquardt.new.optimize(residual_fn, params)
puts "LevenbergMarquardt: sse=#{lm_value} params=#{params.to_a}"
puts "(true params: A=#{A_true} k=#{k_true} c=#{c_true})"

puts "\n=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================\n"

class Sphere1D < Datamunge::ArbitraryFunction
  # f(x) = (x - 2)^2 + 1, minimum at x=2, f=1.
  def evaluate(coordinates)
    (coordinates[0] - 2.0)**2 + 1.0
  end
end

bo_f = Sphere1D.new
bo_x = dv(-4.0)
lower = dv(-5.0)
upper = dv(5.0)
surrogate = Datamunge::RBFGaussianProcessSurrogate.new(1.0, 1e-6)

bo_options = Datamunge::BayesianOptimizationOptions.new
bo_options.initial_samples = 8
bo_options.max_iterations = 50
bo_options.seed = 42

bo_value = Datamunge::BayesianOptimization.new(bo_options).optimize(bo_f, bo_x, lower, upper, surrogate)
puts "BayesianOptimization: f=#{bo_value} x=#{bo_x.to_a} (true minimum: f=1 at x=2)"
