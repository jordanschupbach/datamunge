import math

from pydatamunge import datamunge as dm


# ---- New function-type interfaces (director-based extension points), matching
# ---- ArbitraryFunction/DifferentiableFunction in style but adding a second-order or
# ---- structural piece of information the corresponding optimizer needs. ----


class QuadraticBowlHessian(dm.HessianFunction):
    """A simple 2-D quadratic bowl f(x) = (x - target)^T A (x - target) with a known
    minimum, exposed with its exact (constant) Hessian -- the natural example for Newton's
    method, which needs curvature information at every step."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        # SPD matrix, so the bowl is well-conditioned and Newton converges in one step.
        self.A = [[4.0, 0.5], [0.5, 2.0]]

    def _grad_of_quadratic(self, d):
        return [
            2.0 * (self.A[0][0] * d[0] + self.A[0][1] * d[1]),
            2.0 * (self.A[1][0] * d[0] + self.A[1][1] * d[1]),
        ]

    def evaluate(self, x):
        d = [x[0] - self.target[0], x[1] - self.target[1]]
        return d[0] * (self.A[0][0] * d[0] + self.A[0][1] * d[1]) + d[1] * (self.A[1][0] * d[0] + self.A[1][1] * d[1])

    def gradient(self, x):
        d = [x[0] - self.target[0], x[1] - self.target[1]]
        return self._grad_of_quadratic(d)

    def hessian(self, x):
        return [
            [2.0 * self.A[0][0], 2.0 * self.A[0][1]],
            [2.0 * self.A[1][0], 2.0 * self.A[1][1]],
        ]


class ExpDecayResidual(dm.ResidualFunction):
    """Residuals for fitting A*exp(-k*t) + c to synthetic noisy-free data -- the classic
    nonlinear-least-squares curve fit, and the textbook use case for LevenbergMarquardt."""

    def __init__(self, ts, ys):
        super().__init__()
        self.ts = ts
        self.ys = ys

    def residuals(self, p):
        A, k, c = p
        return [A * math.exp(-k * t) + c - y for t, y in zip(self.ts, self.ys)]

    def jacobian(self, p):
        A, k, c = p
        rows = []
        for t in self.ts:
            e = math.exp(-k * t)
            rows.append([e, -A * t * e, 1.0])
        return rows


# optimize() mutates its `coordinates` argument in place, so it needs a real DVector (a
# plain Python list works for arguments read by value, but not for this in/out one).
def dv(*values):
    return dm.DVector(list(values))


print("=================== HessianFunction: Newton's method on a quadratic bowl ===================")
target = [3.0, -1.5]
f = QuadraticBowlHessian(target)
x = dv(0.0, 0.0)
newton_options = dm.NewtonOptions()
newton_options.max_iterations, newton_options.tolerance = 50, 1e-10
value = dm.Newton(newton_options).optimize(f, x)
print(f"Newton:            f={value} x={list(x)} (true minimum: f=0 at {target})")

print("\n=================== HessianFunction: TrustRegionNewton on the same bowl ===================")
f = QuadraticBowlHessian(target)
x = dv(5.0, 5.0)
trn_options = dm.TrustRegionNewtonOptions()
trn_options.max_iterations, trn_options.tolerance = 100, 1e-10
value = dm.TrustRegionNewton(trn_options).optimize(f, x)
print(f"TrustRegionNewton: f={value} x={list(x)} (true minimum: f=0 at {target})")

print("\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================")
true_A, true_k, true_c = 5.0, 0.7, 1.0
ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ys = [true_A * math.exp(-true_k * t) + true_c for t in ts]

residual_fn = ExpDecayResidual(ts, ys)
p = dv(1.0, 0.1, 0.0)
lm_options = dm.LevenbergMarquardtOptions()
lm_options.max_iterations, lm_options.tolerance = 200, 1e-12
value = dm.LevenbergMarquardt(lm_options).optimize(residual_fn, p)
print(f"LevenbergMarquardt: sum-sq-residual={value} params(A,k,c)={list(p)}")
print(f"                    true params(A,k,c)=[{true_A}, {true_k}, {true_c}]")

print("\n=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================")


class Shifted1DBowl(dm.ArbitraryFunction):
    """A simple 1-D bowl with a known minimum at x=1.7, no gradient exposed -- exactly the
    black-box setting BayesianOptimization targets."""

    def evaluate(self, x):
        return (x[0] - 1.7) ** 2 + 0.1 * math.sin(10.0 * x[0])


f = Shifted1DBowl()
x = dv(0.0)
lower, upper = [-3.0], [5.0]
surrogate = dm.RBFGaussianProcessSurrogate(1.0, 1e-6)
bo_options = dm.BayesianOptimizationOptions()
bo_options.initial_samples, bo_options.max_iterations, bo_options.seed = 10, 60, 42
value = dm.BayesianOptimization(bo_options).optimize(f, x, lower, upper, surrogate)
print(f"BayesianOptimization: f={value} x={list(x)} (target minimum near x=1.7)")
