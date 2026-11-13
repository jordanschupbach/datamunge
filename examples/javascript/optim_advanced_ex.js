// CMD: just TARGET=optim_advanced_ex run-javascript

// Local dev: build first (`just build-javascript`), then run this example.
//
// The datamunge::optim optimizers take abstract C++ interfaces (HessianFunction /
// ResidualFunction / ArbitraryFunction / BayesianSurrogate) that callers implement.
// From JavaScript, subclass the interface with `class Mine extends datamunge.X {...}`:
// the subclass's methods override the C++ virtual methods, and C++ calls back into them
// while the real optimizer runs. Coordinates are updated in place through a
// datamunge.ref() cell (the C++ parameter is a non-const std::vector<double>&).
const datamunge = require("../../index.js");

console.log("=================== HessianFunction: Newton on a 2D quadratic bowl ===================");

// f(x, y) = (x - 3)^2 + (y + 1)^2 -- known minimum: f=0 at (3, -1).
class QuadraticBowl extends datamunge.HessianFunction {
  evaluate(x) {
    return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2;
  }
  gradient(x) {
    return [2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)];
  }
  hessian(_x) {
    return [
      [2.0, 0.0],
      [0.0, 2.0],
    ];
  }
}
const quadraticBowl = new QuadraticBowl();

const newtonStart = datamunge.ref([0.0, 0.0]);
const newtonValue = new datamunge.Newton().optimize(quadraticBowl, newtonStart);
console.log(
  `Newton: f=${newtonValue} x=[${newtonStart.value.join(", ")}] (true minimum: f=0 at [3, -1])`,
);

console.log("\n=================== HessianFunction: TrustRegionNewton on the same bowl ===================");
const trStart = datamunge.ref([-5.0, 8.0]);
const trValue = new datamunge.TrustRegionNewton().optimize(quadraticBowl, trStart);
console.log(
  `TrustRegionNewton: f=${trValue} x=[${trStart.value.join(", ")}] (true minimum: f=0 at [3, -1])`,
);

console.log(
  "\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================",
);

// Fit y = A * exp(-k * t) + c to a handful of synthetic samples generated from
// known parameters A=5, k=0.7, c=1, with a little bit of Gaussian-ish noise baked in
// (fixed values, not sampled at runtime, so the example is deterministic).
const trueA = 5.0;
const trueK = 0.7;
const trueC = 1.0;
const ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
const noise = [0.05, -0.03, 0.04, -0.02, 0.01, -0.04, 0.02, 0.0, -0.01];
const ys = ts.map((t, i) => trueA * Math.exp(-trueK * t) + trueC + noise[i]);

class CurveFit extends datamunge.ResidualFunction {
  // params = [A, k, c]
  residuals(params) {
    const [A, k, c] = params;
    return ts.map((t, i) => A * Math.exp(-k * t) + c - ys[i]);
  }
  jacobian(params) {
    const [A, k, _c] = params;
    return ts.map((t) => {
      const e = Math.exp(-k * t);
      return [e, -A * t * e, 1.0];
    });
  }
}

const lmParams = datamunge.ref([1.0, 0.1, 0.0]);
const lmValue = new datamunge.LevenbergMarquardt().optimize(new CurveFit(), lmParams);
const [fitA, fitK, fitC] = lmParams.value;
console.log(`LevenbergMarquardt: sum_sq_residuals=${lmValue}`);
console.log(`  recovered A=${fitA.toFixed(4)} k=${fitK.toFixed(4)} c=${fitC.toFixed(4)}`);
console.log(`  true      A=${trueA} k=${trueK} c=${trueC}`);

console.log(
  "\n=================== BayesianOptimization + ready-made RBFGaussianProcessSurrogate ===================",
);

// f(x) = (x - 2)^2 + 3, bounded to [-5, 5] -- known minimum: f=3 at x=2.
// RBFGaussianProcessSurrogate is a concrete C++ BayesianSurrogate implementation
// (fit()/acquisition() already implemented in C++), so no JS-side subclassing is
// needed for the surrogate itself -- only the objective subclasses ArbitraryFunction.
class Bowl1d extends datamunge.ArbitraryFunction {
  evaluate(x) {
    return (x[0] - 2.0) ** 2 + 3.0;
  }
}
const bowl1d = new Bowl1d();

const boOptions = new datamunge.BayesianOptimizationOptions();
boOptions.initial_samples = 8;
boOptions.max_iterations = 40;
boOptions.seed = 42;

const bayesStart = datamunge.ref([0.0]);
const bayesValue = new datamunge.BayesianOptimization(boOptions).optimize(
  bowl1d,
  bayesStart,
  [-5.0],
  [5.0],
  new datamunge.RBFGaussianProcessSurrogate(1.0, 1e-6),
);
console.log(
  `BayesianOptimization (RBF surrogate): f=${bayesValue} x=[${bayesStart.value.join(", ")}] (true minimum: f=3 at x=2)`,
);

console.log(
  "\n=================== BayesianOptimization with a JS-defined BayesianSurrogate ===================",
);

// Same 1D bowl, but this time the surrogate itself is also a JS subclass of
// BayesianSurrogate implementing fit(points, values) + acquisition(point, incumbent) --
// exercises BayesianSurrogate::fit/acquisition through the same virtual-override mechanism.
// A minimal nearest-neighbour surrogate with a naive "distance-weighted uncertainty"
// acquisition function -- not competitive with the real RBF surrogate, but enough to
// prove the fit()/acquisition() callbacks actually reach JS.
class NearestNeighborSurrogate extends datamunge.BayesianSurrogate {
  constructor() {
    super();
    this.points = [];
    this.values = [];
  }
  fit(newPoints, newValues) {
    this.points = newPoints;
    this.values = newValues;
  }
  acquisition(point, incumbent) {
    const points = this.points;
    const values = this.values;
    let bestDist = Infinity;
    let bestValue = incumbent;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      let distSq = 0.0;
      for (let d = 0; d < p.length; d++) distSq += (p[d] - point[d]) ** 2;
      if (distSq < bestDist) {
        bestDist = distSq;
        bestValue = values[i];
      }
    }
    // Reward points far from any sampled point (exploration) and points near a
    // predicted-good value (exploitation), loosely.
    return Math.sqrt(bestDist) - (bestValue - incumbent);
  }
}

const jsSurrogateResult = new datamunge.BayesianOptimization(boOptions).run(
  bowl1d,
  [-5.0],
  [5.0],
  new NearestNeighborSurrogate(),
);
console.log(
  `BayesianOptimization (JS surrogate): f=${jsSurrogateResult.best_value} x=[${jsSurrogateResult.best_point.join(", ")}] (true minimum: f=3 at x=2)`,
);
