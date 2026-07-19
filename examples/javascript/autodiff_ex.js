const datamunge = require("../../index.js");

function rosenbrockDual(x0, x1) {
  const a = new datamunge.Dual(1.0, 0.0).subtract(x0);
  const b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
}

function rosenbrockValue(x0, x1) {
  const a = 1.0 - x0;
  const b = x1 - x0 * x0;
  return a * a + 100.0 * b * b;
}

function rosenbrockHyperdual(x0, x1) {
  const a = new datamunge.HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0);
  const b = x1.subtract(x0.multiply(x0));
  return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
}

console.log("=================== Forward mode: scalar derivative ===================");
const x0 = new datamunge.Dual(2.0, 1.0);
const f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0));
console.log(`f(x) = x^3 - 2x, f'(2) = ${f.derivative()} (exact: 10)`);

console.log("\n=================== Reverse mode: build a graph by hand ===================");
const tape = new datamunge.Tape();
const a = new datamunge.Var(tape, 2.0);
const b = new datamunge.Var(tape, 3.0);
const y = a.multiply(b).add(a.sin());
console.log(`y = a*b + sin(a) at a=2, b=3 -> y = ${y.value()}`);
const adjoint = tape.backward(y);
console.log(`dy/da = ${adjoint.get(a.index())} (exact: b + cos(a))`);
console.log(`dy/db = ${adjoint.get(b.index())} (exact: a)`);

console.log("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
const px = 0.0;
const py = 0.0;

const gx = rosenbrockDual(new datamunge.Dual(px, 1.0), new datamunge.Dual(py, 0.0)).derivative();
const gy = rosenbrockDual(new datamunge.Dual(px, 0.0), new datamunge.Dual(py, 1.0)).derivative();

const tape2 = new datamunge.Tape();
const vx = new datamunge.Var(tape2, px);
const vy = new datamunge.Var(tape2, py);
const va = new datamunge.Var(tape2, 1.0).subtract(vx);
const vb = vy.subtract(vx.multiply(vx));
const vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0));
const gradRev = tape2.backward(vf);

console.log(`f(0,0) = ${rosenbrockValue(px, py)}`);
console.log(`gradient (forward mode): [${gx}, ${gy}]`);
console.log(`gradient (reverse mode): [${gradRev.get(vx.index())}, ${gradRev.get(vy.index())}]`);

console.log("\n=================== Jacobian of a vector-valued function ===================");
const vxVal = 2.0;
const vyVal = 3.0;

const jac = [
  [0, 0],
  [0, 0],
  [0, 0],
];
for (const [col, seedX, seedY] of [
  [0, 1.0, 0.0],
  [1, 0.0, 1.0],
]) {
  const ox = new datamunge.Dual(vxVal, seedX).multiply(new datamunge.Dual(vxVal, seedX));
  const oy = new datamunge.Dual(vxVal, seedX).multiply(new datamunge.Dual(vyVal, seedY));
  const oz = new datamunge.Dual(vyVal, seedY).multiply(new datamunge.Dual(vyVal, seedY)).multiply(new datamunge.Dual(vyVal, seedY));
  jac[0][col] = ox.derivative();
  jac[1][col] = oy.derivative();
  jac[2][col] = oz.derivative();
}
console.log("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
for (let row = 0; row < 3; row++) {
  console.log(`  [${jac[row][0]}, ${jac[row][1]}]`);
}

console.log("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
const mx = 1.0;
const my = 1.0;

const seeds = [
  [1.0, 0.0],
  [0.0, 1.0],
];
const h = [
  [0, 0],
  [0, 0],
];
for (let i = 0; i < 2; i++) {
  for (let j = 0; j < 2; j++) {
    const e1 = seeds[i];
    const e2 = seeds[j];
    const hx = new datamunge.HyperDual(mx, e1[0], e2[0], 0.0);
    const hy = new datamunge.HyperDual(my, e1[1], e2[1], 0.0);
    h[i][j] = rosenbrockHyperdual(hx, hy).eps1eps2();
  }
}
console.log("Hessian of the Rosenbrock function at its minimum (1,1):");
for (let row = 0; row < 2; row++) {
  console.log(`  [${h[row][0]}, ${h[row][1]}]`);
}

console.log("\n=================== Gradient descent driven by reverse-mode gradients ===================");
let pointX = -1.2;
let pointY = 1.0;
const learningRate = 0.001;
const nSteps = 2000;
for (let step = 0; step < nSteps; step++) {
  const t = new datamunge.Tape();
  const vx2 = new datamunge.Var(t, pointX);
  const vy2 = new datamunge.Var(t, pointY);
  const va2 = new datamunge.Var(t, 1.0).subtract(vx2);
  const vb2 = vy2.subtract(vx2.multiply(vx2));
  const vf2 = va2.multiply(va2).add(vb2.multiply(vb2).multiply_scalar(100.0));
  const grad = t.backward(vf2);
  const loss = vf2.value();
  pointX -= learningRate * grad.get(vx2.index());
  pointY -= learningRate * grad.get(vy2.index());
  if (step === 0 || step === nSteps - 1) {
    console.log(`step ${step}: loss = ${loss}, x = [${pointX}, ${pointY}]`);
  }
}
console.log("(true minimum is at [1, 1] with loss 0)");
