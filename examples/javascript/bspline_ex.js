// CMD: just TARGET=bspline_ex run-javascript
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.

const datamunge = require("../../index.js");

// Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
const x = [];
const y = [];
const z = [];
for (let iy = 0; iy < 8; iy++) {
  for (let ix = 0; ix < 8; ix++) {
    const xv = ix / 7.0;
    const yv = iy / 7.0;
    x.push(xv);
    y.push(yv);
    z.push(Math.sin(xv) + yv * yv);
  }
}

const data = new datamunge.DataFrame();
data.add_numeric_column("x", x);
data.add_numeric_column("y", y);
data.add_numeric_column("z", z);

const surface = new datamunge.LM(data, "z ~ bs(x, y)");
console.log("Fitted z ~ bs(x, y)");
surface.print_summary();

const newPoints = new datamunge.DataFrame();
newPoints.add_numeric_column("x", [0.25, 0.75]);
newPoints.add_numeric_column("y", [0.5, 0.25]);
const pred = surface.predict(newPoints);
console.log(`Predictions: [${pred[0].toFixed(6)}, ${pred[1].toFixed(6)}]`);
