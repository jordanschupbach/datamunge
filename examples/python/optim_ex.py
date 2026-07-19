import math

from pydatamunge import datamunge as dm


# ---- Function types, defined via SWIG directors (Python subclasses of the bound C++
# ---- interfaces) ----

class QuadraticBowl(dm.DifferentiableFunction):
    """A plain 3-D quadratic bowl with a known minimum."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def evaluate(self, coordinates):
        return sum((c - t) ** 2 for c, t in zip(coordinates, self.target))

    def gradient(self, coordinates):
        return [2.0 * (c - t) for c, t in zip(coordinates, self.target)]


class RosenbrockFn(dm.DifferentiableFunction):
    """The classic Rosenbrock "banana" function -- a much harder landscape."""

    def evaluate(self, x):
        a = 1.0 - x[0]
        b = x[1] - x[0] * x[0]
        return a * a + 100.0 * b * b

    def gradient(self, x):
        return [-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]), 200.0 * (x[1] - x[0] * x[0])]


class LinearRegressionLoss(dm.DifferentiableSeparableFunction):
    """Ordinary least squares as a sum of per-example losses -- the textbook case for SGD."""

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def num_functions(self):
        return len(self.y)

    def _predict(self, w, i):
        return sum(w[j] * self.X[i][j] for j in range(len(w)))

    def evaluate_term(self, w, i):
        err = self._predict(w, i) - self.y[i]
        return err * err

    def gradient_term(self, w, i):
        err = self._predict(w, i) - self.y[i]
        return [2.0 * err * self.X[i][j] for j in range(len(w))]


class BumpyFunction(dm.ArbitraryFunction):
    """A bumpy, multimodal landscape -- no gradient available, so only a derivative-free
    method (SimulatedAnnealing) can be used here."""

    def evaluate(self, x):
        bowl = (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
        ripples = 5.0 * math.sin(x[0]) * math.cos(x[1])
        return bowl + ripples


class RastriginFunction(dm.ArbitraryFunction):
    """The classic Rastrigin function -- highly multimodal (many local minima arranged in a
    regular grid), global minimum f=0 at the origin. A standard torture test for
    population-based methods, since local/gradient-based methods get stuck in the first
    basin they land in."""

    def evaluate(self, x):
        total = 10.0 * len(x)
        for xi in x:
            total += xi * xi - 10.0 * math.cos(2.0 * math.pi * xi)
        return total


# optimize() mutates its `coordinates` argument in place, so it needs a real DVector (a
# plain Python list works for arguments read by value, but not for this in/out one).
def dv(*values):
    return dm.DVector(list(values))


print("=================== DifferentiableFunction: three optimizers, one bowl ===================")
target = [4.0, -2.0, 1.0]

f = QuadraticBowl(target)
x = dv(0.0, 0.0, 0.0)
gd_options = dm.GradientDescentOptions()
gd_options.step_size, gd_options.momentum, gd_options.max_iterations, gd_options.tolerance = 0.1, 0.0, 1000, 1e-10
value = dm.GradientDescent(gd_options).optimize(f, x)
print(f"GradientDescent: f={value} x={list(x)}")

f = QuadraticBowl(target)
x = dv(0.0, 0.0, 0.0)
value = dm.Adam().optimize(f, x)
print(f"Adam:             f={value} x={list(x)}")

f = QuadraticBowl(target)
x = dv(0.0, 0.0, 0.0)
value = dm.LBFGS().optimize(f, x)
print(f"LBFGS:            f={value} x={list(x)} (converges in far fewer iterations)")

print("\n=================== LBFGS on the Rosenbrock function ===================")
f = RosenbrockFn()
x = dv(-1.2, 1.0)
value = dm.LBFGS().optimize(f, x)
print(f"f={value} x={list(x)} (true minimum: f=0 at [1, 1])")

print("\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================")
iris = dm.DataFrame.iris()
sepal_length = [iris.numeric_at("Sepal.Length", i) for i in range(iris.nrows())]
sepal_width = [iris.numeric_at("Sepal.Width", i) for i in range(iris.nrows())]
petal_length = [iris.numeric_at("Petal.Length", i) for i in range(iris.nrows())]

# SGD with a single constant step size converges far faster (and far more reliably) on
# standardized features -- unnormalized predictors of very different scales give the loss an
# ill-conditioned Hessian, which plain constant-step SGD handles poorly. Standard practice.


def mean_of(v):
    return sum(v) / len(v)


def stddev_of(v, mean):
    return math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))


mean1, std1 = mean_of(sepal_length), stddev_of(sepal_length, mean_of(sepal_length))
mean2, std2 = mean_of(sepal_width), stddev_of(sepal_width, mean_of(sepal_width))

X = [[1.0, (sepal_length[i] - mean1) / std1, (sepal_width[i] - mean2) / std2] for i in range(iris.nrows())]
loss = LinearRegressionLoss(X, petal_length)

w_std = dv(0.0, 0.0, 0.0)
sgd_options = dm.SGDOptions()
sgd_options.step_size, sgd_options.max_epochs, sgd_options.batch_size = 0.01, 300, 8
dm.SGD(sgd_options).optimize(loss, w_std)

# Convert the standardized-space weights back to the original feature scale.
w = [w_std[0] - w_std[1] * mean1 / std1 - w_std[2] * mean2 / std2, w_std[1] / std1, w_std[2] / std2]
print(f"SGD weights (intercept, Sepal.Length, Sepal.Width): {w}")

lm = dm.LM(iris, "Petal.Length ~ Sepal.Length + Sepal.Width")
print(f"LM  weights (intercept, Sepal.Length, Sepal.Width): {list(lm.coefficients())} (closed-form OLS, for comparison)")

print("\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================")
f = BumpyFunction()
x = dv(0.0, 0.0)
sa_options = dm.SimulatedAnnealingOptions()
sa_options.initial_temperature, sa_options.cooling_rate = 10.0, 0.999
sa_options.max_iterations, sa_options.step_std_dev = 20000, 0.5
value = dm.SimulatedAnnealing(sa_options).optimize(f, x)
print(f"f={value} x={list(x)} (found without ever computing a gradient)")

print("\n=================== Population-based methods on the Rastrigin function ===================")
lower, upper = [-5.12, -5.12], [5.12, 5.12]

f = RastriginFunction()
x = dv(3.0, -4.0)
pso_options = dm.PSOOptions()
pso_options.topology, pso_options.inertia_strategy = "global", "constant"
value = dm.PSO(pso_options).optimize(f, x, lower, upper)
print(f"PSO (global topology, constant inertia):    f={value} x={list(x)}")

f = RastriginFunction()
x = dv(3.0, -4.0)
pso_options = dm.PSOOptions()
pso_options.topology, pso_options.inertia_strategy = "ring", "linear_decay"
value = dm.PSO(pso_options).optimize(f, x, lower, upper)
print(f"PSO (ring topology, linear-decay inertia):  f={value} x={list(x)}")

f = RastriginFunction()
x = dv(3.0, -4.0)
de_options = dm.DEOptions()
de_options.mutation_strategy, de_options.crossover_strategy = "rand1", "binomial"
value = dm.DifferentialEvolution(de_options).optimize(f, x, lower, upper)
print(f"DE (rand1/binomial):                        f={value} x={list(x)}")

f = RastriginFunction()
x = dv(3.0, -4.0)
de_options = dm.DEOptions()
de_options.mutation_strategy, de_options.crossover_strategy = "best1", "exponential"
value = dm.DifferentialEvolution(de_options).optimize(f, x, lower, upper)
print(f"DE (best1/exponential):                     f={value} x={list(x)}")

f = RastriginFunction()
x = dv(3.0, -4.0)
ga_options = dm.GAOptions()
ga_options.selection_strategy, ga_options.crossover_strategy = "tournament", "blend"
value = dm.GeneticAlgorithm(ga_options).optimize(f, x, lower, upper)
print(f"GA (tournament/blend, elitism on):           f={value} x={list(x)}")

f = RastriginFunction()
x = dv(3.0, -4.0)
ga_options = dm.GAOptions()
ga_options.selection_strategy, ga_options.crossover_strategy = "rank", "uniform"
ga_options.elitism = False
value = dm.GeneticAlgorithm(ga_options).optimize(f, x, lower, upper)
print(f"GA (rank/uniform, elitism off):              f={value} x={list(x)} (true minimum: f=0 at [0, 0])")
