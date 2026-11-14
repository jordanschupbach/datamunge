# CMD: just TARGET=optim_advanced_ex run-julia

# Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
# AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Julia.
#
# Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
# EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
# ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
# implement the objective. From Julia, subclass the interface with @subclass and add methods to
# its generic functions; the optimizer then calls the Julia methods. Coordinates are updated in
# place through a Ref (the C++ parameter is a non-const std::vector<double>&).
using Datamunge
using Printf
const dm = Datamunge

println("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================")
# Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
# then check the acquisition function favors an unexplored region.
let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    ys = (xs .- 1) .^ 2
    surrogate = RBFGaussianProcessSurrogate(1.0, 1e-6)
    fit(surrogate, [[x] for x in xs], ys)
    incumbent = min(0.0, minimum(ys))  # best observed value so far
    println("incumbent (best observed y): $incumbent")
    for x in (0.25, 1.0, 2.5)
        @printf("acquisition(x=%.2f) = %.6f\n", x, acquisition(surrogate, [x], incumbent))
    end
end

println("\n=================== Options structs + optimizer construction (all 10 new classes) ===================")
# Every new Options struct and optimizer class constructs cleanly, proving
# the binding for the whole batch is wired correctly.
const pg = let options = ProximalGradientOptions()
    options.step_size = 0.05
    ProximalGradient(options)
end
println("ProximalGradient: $pg")
const fista = FISTA(FISTAOptions())
println("FISTA: $fista")
const lm = let options = LevenbergMarquardtOptions()
    options.max_iterations = 200
    LevenbergMarquardt(options)
end
println("LevenbergMarquardt: $lm")
const newton = Newton(NewtonOptions())
println("Newton: $newton")
const trn = TrustRegionNewton(TrustRegionNewtonOptions())
println("TrustRegionNewton: $trn")
const al = AugmentedLagrangian(AugmentedLagrangianOptions())
println("AugmentedLagrangian: $al")
const sqp = SQP(SQPOptions())
println("SQP: $sqp")
const ip = InteriorPoint(InteriorPointOptions())
println("InteriorPoint: $ip")
const bo = let options = BayesianOptimizationOptions()
    options.initial_samples = 8
    BayesianOptimization(options)
end
println("BayesianOptimization: $bo")

println("\n=================== Julia objectives (subclassing the C++ interfaces) ===================")
# Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
@subclass mutable struct Quadratic <: HessianFunction end
dm.evaluate(::Quadratic, c) = (c[1] - 1)^2 + 10 * (c[2] + 2)^2
dm.gradient(::Quadratic, c) = [2 * (c[1] - 1), 20 * (c[2] + 2)]
dm.hessian(::Quadratic, c) = [[2.0, 0.0], [0.0, 20.0]]

let start = Ref([5.0, 5.0])
    value = optimize(newton, Quadratic(), start)
    x = start[]
    @printf("Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n", value, x[1], x[2])
end

# Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
const ts = [0.0, 0.5, 1.0, 1.5, 2.0]
@subclass mutable struct Exponential <: ResidualFunction end
dm.residuals(::Exponential, p) = [p[1] * exp(p[2] * t) - 2 * exp(0.5 * t) for t in ts]
dm.jacobian(::Exponential, p) = [[exp(p[2] * t), p[1] * t * exp(p[2] * t)] for t in ts]

let params = Ref([1.0, 0.1])
    sse = optimize(lm, Exponential(), params)
    p = params[]
    @printf("LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)\n", sse, p[1], p[2])
end

# An interface method a subclass leaves out throws when the optimizer calls it.
@subclass mutable struct Incomplete <: HessianFunction end
dm.evaluate(::Incomplete, c) = 0.0
let err = try
        optimize(newton, Incomplete(), Ref([0.0, 0.0]))
        nothing
    catch e
        e
    end
    println("missing gradient(): ok=$(err === nothing), error mentions it: $(err !== nothing && occursin("gradient", sprint(showerror, err)))")
end
