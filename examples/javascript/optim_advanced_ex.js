// Local dev: build first (`just build-javascript`), then run this example.
//
// SWIG's Node/N-API backend generates no director code at all (verified:
// `grep -c -i director` on the generated wrapper is 0), so a JS class can
// never really subclass datamunge::optim::HessianFunction /
// ResidualFunction / ProximalFunction / EqualityConstrainedFunction /
// InequalityConstrainedFunction / BayesianSurrogate (or the older
// ArbitraryFunction/DifferentiableFunction) the way it can in Python/Ruby/
// Perl. Instead, `src/datamunge_js_callbacks.inl` hand-implements a
// duck-typed bridge: pass a plain JS object with the interface's required
// methods (evaluate/gradient/hessian/residuals/jacobian/...) to one of the
// `datamunge.run_*` helpers below, and C++ calls back into those methods by
// name while the real optimizer runs. This is the JS analogue of
// "subclassing" a new interface type for this backend.
const datamunge = require("../../index.js");

console.log("=================== HessianFunction: Newton on a 2D quadratic bowl ===================");

// f(x, y) = (x - 3)^2 + (y + 1)^2 -- known minimum: f=0 at (3, -1).
const quadraticBowl = {
  evaluate(x) {
    return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2;
  },
  gradient(x) {
    return [2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)];
  },
  hessian(_x) {
    return [
      [2.0, 0.0],
      [0.0, 2.0],
    ];
  },
};

const newtonResult = datamunge.run_newton(quadraticBowl, [0.0, 0.0]);
console.log(
  `Newton: f=${newtonResult.value} x=[${newtonResult.coordinates.join(", ")}] (true minimum: f=0 at [3, -1])`,
);

console.log("\n=================== HessianFunction: TrustRegionNewton on the same bowl ===================");
const trResult = datamunge.run_trust_region_newton(quadraticBowl, [-5.0, 8.0]);
console.log(
  `TrustRegionNewton: f=${trResult.value} x=[${trResult.coordinates.join(", ")}] (true minimum: f=0 at [3, -1])`,
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

const curveFit = {
  // params = [A, k, c]
  residuals(params) {
    const [A, k, c] = params;
    return ts.map((t, i) => A * Math.exp(-k * t) + c - ys[i]);
  },
  jacobian(params) {
    const [A, k, c] = params;
    return ts.map((t) => {
      const e = Math.exp(-k * t);
      return [e, -A * t * e, 1.0];
    });
  },
};

const lmResult = datamunge.run_levenberg_marquardt(curveFit, [1.0, 0.1, 0.0]);
const [fitA, fitK, fitC] = lmResult.coordinates;
console.log(`LevenbergMarquardt: sum_sq_residuals=${lmResult.value}`);
console.log(`  recovered A=${fitA.toFixed(4)} k=${fitK.toFixed(4)} c=${fitC.toFixed(4)}`);
console.log(`  true      A=${trueA} k=${trueK} c=${trueC}`);

console.log(
  "\n=================== BayesianOptimization + ready-made RBFGaussianProcessSurrogate ===================",
);

// f(x) = (x - 2)^2 + 3, bounded to [-5, 5] -- known minimum: f=3 at x=2.
// RBFGaussianProcessSurrogate is a concrete C++ BayesianSurrogate implementation
// (fit()/acquisition() already implemented in C++), so no JS-side subclassing or
// bridging is needed for the surrogate itself -- only the objective needs the
// ArbitraryFunction duck-typed bridge.
const bowl1d = {
  evaluate(x) {
    return (x[0] - 2.0) ** 2 + 3.0;
  },
};

const bayesResult = datamunge.run_bayesian_optimization_rbf(
  bowl1d,
  [0.0],
  [-5.0],
  [5.0],
  { length_scale: 1.0, noise: 1e-6 },
  { initial_samples: 8, max_iterations: 40, seed: 42 },
);
console.log(
  `BayesianOptimization (RBF surrogate): f=${bayesResult.value} x=[${bayesResult.coordinates.join(", ")}] (true minimum: f=3 at x=2)`,
);

console.log(
  "\n=================== BayesianOptimization with a JS-defined BayesianSurrogate ===================",
);

// Same 1D bowl, but this time the surrogate itself is also bridged from a plain JS
// object implementing fit(points, values) + acquisition(point, incumbent) -- exercises
// BayesianSurrogate::fit/acquisition through the same duck-typed bridge mechanism.
// A minimal nearest-neighbour surrogate with a naive "distance-weighted uncertainty"
// acquisition function -- not competitive with the real RBF surrogate, but enough to
// prove the fit()/acquisition() callback bridge actually reaches JS.
function makeNearestNeighborSurrogate() {
  let points = [];
  let values = [];
  return {
    fit(newPoints, newValues) {
      points = newPoints;
      values = newValues;
    },
    acquisition(point, incumbent) {
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
    },
  };
}

const jsSurrogateResult = datamunge.run_bayesian_optimization(
  bowl1d,
  makeNearestNeighborSurrogate(),
  [0.0],
  [-5.0],
  [5.0],
  { initial_samples: 8, max_iterations: 40, seed: 42 },
);
console.log(
  `BayesianOptimization (JS surrogate): f=${jsSurrogateResult.value} x=[${jsSurrogateResult.coordinates.join(", ")}] (true minimum: f=3 at x=2)`,
);
