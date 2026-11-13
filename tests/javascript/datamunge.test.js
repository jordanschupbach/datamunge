const fs = require("fs");
const path = require("path");
const datamunge = require("../../index.js");

test("addon loads and exposes hello()", () => {
  expect(typeof datamunge.hello).toBe("function");
  expect(() => datamunge.hello()).not.toThrow();
});

test("vectors cross as arrays and typed arrays", () => {
  expect(datamunge.seql(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  datamunge.set_seed(42);
  const a = datamunge.rnorm(0, 1, 4);
  datamunge.set_seed(42);
  expect(datamunge.rnorm(0, 1, 4)).toEqual(a);
  const result = datamunge.t_test_two_sample(new Float64Array([1, 2, 3, 4]), [2, 3, 4, 5]);
  expect(result.p_value).toBeGreaterThan(0);
  expect(result.p_value).toBeLessThanOrEqual(1);
});

test("conversion errors name the argument", () => {
  expect(() => datamunge.seql("a", 1, 5)).toThrow(TypeError);
  expect(() => datamunge.seql("a", 1, 5)).toThrow(/argument 'from'/);
});

test("JavaScript subclasses implement C++ objectives (directors)", () => {
  class Quadratic extends datamunge.DifferentiableFunction {
    evaluate(x) {
      return (x[0] - 3) ** 2 + (x[1] + 1) ** 2;
    }
    gradient(x) {
      return [2 * (x[0] - 3), 2 * (x[1] + 1)];
    }
  }
  const coordinates = datamunge.ref([0, 0]);
  const value = new datamunge.LBFGS().optimize(new Quadratic(), coordinates);
  expect(value).toBeCloseTo(0, 6);
  expect(coordinates.value[0]).toBeCloseTo(3, 4);
  expect(coordinates.value[1]).toBeCloseTo(-1, 4);
});

test("exceptions thrown by JavaScript overrides pass through C++ unchanged", () => {
  const thrown = new Error("from the objective");
  class Failing extends datamunge.DifferentiableFunction {
    evaluate() {
      throw thrown;
    }
    gradient() {
      throw thrown;
    }
  }
  let caught;
  try {
    new datamunge.LBFGS().optimize(new Failing(), datamunge.ref([0, 0]));
  } catch (error) {
    caught = error;
  }
  expect(caught).toBe(thrown);
});

test("tensors train through automatic differentiation", () => {
  const x = new datamunge.NnTensor([1, 2, 3], [3], "float64");
  x.set_requires_grad(true);
  x.square().sum().backward();
  expect(x.grad().values()).toEqual([2, 4, 6]);
});

test("TypeScript declarations cover the exports and carry the C++ documentation", () => {
  const types = fs.readFileSync(path.join(__dirname, "../../index.d.ts"), "utf8");
  expect(types).toContain("export declare abstract class DifferentiableFunction extends ArbitraryFunction {");
  expect(types).toContain("Returns the gradient of the objective at `coordinates`.");
  const declared = new Set(
    [...types.matchAll(/^export declare (?:abstract )?(?:class|function|enum|const) (\w+)/gm)].map((m) => m[1]),
  );
  for (const m of types.matchAll(/^export \{ \w+ as (\w+) \};/gm)) declared.add(m[1]);
  const undeclared = Object.keys(datamunge).filter((name) => !declared.has(name));
  expect(undeclared).toEqual([]);
});
