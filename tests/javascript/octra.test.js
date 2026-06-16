const octra = require("../../index.js");

test("addon loads and exposes hello()", () => {
  expect(typeof octra.hello).toBe("function");
  expect(() => octra.hello()).not.toThrow();
});

test("STL templates are usable", () => {
  const p = new octra.DPair(1.25, 2.5);
  expect(p.first).toBeCloseTo(1.25);
  expect(p.second).toBeCloseTo(2.5);

  const v = new octra.DVector(2);
  v.set(0, 3.0);
  v.set(1, 4.5);
  expect(v.get(0)).toBeCloseTo(3.0);
  expect(v.get(1)).toBeCloseTo(4.5);
});

test("can pass JS function into native code", () => {
  expect(octra.call_with_function(3.0, (x) => x * 2.0)).toBeCloseTo(6.0);
  expect(
    octra.map_array_with_function([1.0, 2.0, 3.0], (x) => x * 2.0),
  ).toEqual([2.0, 4.0, 6.0]);
});
