# CMD: just TARGET=optim_advanced_ex run-ruby
# Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Ruby.
require "datamunge"

# Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
# EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
# ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
# implement the objective. From Ruby, subclass the interface class and override its virtual
# methods; the optimizer then calls the Ruby overrides. The coordinates Array is updated in place.

# f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
class Quadratic < Datamunge::HessianFunction
  def evaluate(c) = (c[0] - 1)**2 + 10 * (c[1] + 2)**2
  def gradient(c) = [2 * (c[0] - 1), 20 * (c[1] + 2)]
  def hessian(_c) = [[2.0, 0.0], [0.0, 20.0]]
end

# Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
class Exponential < Datamunge::ResidualFunction
  def initialize(ts)
    super()
    @ts = ts
  end

  def residuals(p) = @ts.map { |t| p[0] * Math.exp(p[1] * t) - 2 * Math.exp(0.5 * t) }
  def jacobian(p) = @ts.map { |t| [Math.exp(p[1] * t), p[0] * t * Math.exp(p[1] * t)] }
end

# Leaves gradient() and hessian() out.
class Incomplete < Datamunge::HessianFunction
  def evaluate(_c) = 0.0
end

puts "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ==================="
# Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
# then check the acquisition function favors an unexplored region.
xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ys = xs.map { |x| (x - 1)**2 }
surrogate = Datamunge::RBFGaussianProcessSurrogate.new(1.0, 1e-6)
surrogate.fit(xs.map { |x| [x] }, ys)
incumbent = [0.0, ys.min].min  # best observed value so far
puts "incumbent (best observed y): #{incumbent}"
[0.25, 1.0, 2.5].each do |x|
  printf("acquisition(x=%.2f) = %.6f\n", x, surrogate.acquisition([x], incumbent))
end

puts "\n=================== Options structs + optimizer construction (all 10 new classes) ==================="
# Every new Options struct and optimizer class constructs cleanly, proving
# the binding for the whole batch is wired correctly.
D = Datamunge
pg = D::ProximalGradient.new(D::ProximalGradientOptions.new.tap { |o| o.step_size = 0.05 })
fista = D::FISTA.new(D::FISTAOptions.new)
lm = D::LevenbergMarquardt.new(D::LevenbergMarquardtOptions.new.tap { |o| o.max_iterations = 200 })
newton = D::Newton.new(D::NewtonOptions.new)
trn = D::TrustRegionNewton.new(D::TrustRegionNewtonOptions.new)
al = D::AugmentedLagrangian.new(D::AugmentedLagrangianOptions.new)
sqp = D::SQP.new(D::SQPOptions.new)
ip = D::InteriorPoint.new(D::InteriorPointOptions.new)
bo = D::BayesianOptimization.new(D::BayesianOptimizationOptions.new.tap { |o| o.initial_samples = 8 })
[pg, fista, lm, newton, trn, al, sqp, ip, bo].each do |optimizer|
  name = optimizer.class.name.split("::").last
  puts "#{name}: #{name}()"
end

puts "\n=================== Ruby objectives (subclassing the C++ interfaces) ==================="
# Newton's method on the quadratic bowl.
start = [5.0, 5.0]
value = newton.optimize(Quadratic.new, start)
printf("Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n", value, *start)

# Levenberg-Marquardt fitting y = a * exp(b t).
params = [1.0, 0.1]
sse = lm.optimize(Exponential.new([0.0, 0.5, 1.0, 1.5, 2.0]), params)
printf("LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)\n", sse, *params)

# An interface method a subclass leaves out raises when the optimizer calls it.
err = begin
  newton.optimize(Incomplete.new, [0.0, 0.0])
  nil
rescue Datamunge::Error => e  # mixed into every binding exception class
  e
end
puts "missing gradient(): ok=#{err.nil?}, error mentions it: #{!err.nil? && err.message.include?("gradient")}"
