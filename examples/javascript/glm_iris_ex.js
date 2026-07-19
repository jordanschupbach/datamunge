const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

const iris = datamunge.DataFrame.iris();

const isVirginica = [];
const petalLength = [];
const petalWidth = [];
const n = iris.nrows();
for (let i = 0; i < n; i++) {
  const species = iris.string_at("Species", i);
  if (species === "versicolor" || species === "virginica") {
    isVirginica.push(species === "virginica" ? 1.0 : 0.0);
    petalLength.push(iris.numeric_at("Petal.Length", i));
    petalWidth.push(iris.numeric_at("Petal.Width", i));
  }
}

const sub = new datamunge.DataFrame();
sub.add_numeric_column("Petal.Length", dvector(petalLength));
sub.add_numeric_column("Petal.Width", dvector(petalWidth));
sub.add_numeric_column("is_virginica", dvector(isVirginica));

console.log("=================== Logistic regression (binomial, logit link) ===================");
const logit = new datamunge.GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
logit.print_summary();

const fitted = logit.fitted_values();
let correct = 0;
for (let i = 0; i < isVirginica.length; i++) {
  if ((fitted.get(i) >= 0.5) === (isVirginica[i] >= 0.5)) correct += 1;
}
console.log(`\nResubstitution accuracy at 0.5 threshold: ${(100.0 * correct) / isVirginica.length}%`);

logit.save_diagnostic_plots("glm_logistic_iris");
console.log("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

const widthMean = petalWidth.reduce((a, b) => a + b, 0) / petalWidth.length;
const gridN = 100;
const plMin = Math.min(...petalLength) - 0.3;
const plMax = Math.max(...petalLength) + 0.3;
const gridX = [];
const gridW = [];
for (let i = 0; i < gridN; i++) {
  gridX.push(plMin + ((plMax - plMin) * i) / (gridN - 1));
  gridW.push(widthMean);
}
const grid = new datamunge.DataFrame();
grid.add_numeric_column("Petal.Length", dvector(gridX));
grid.add_numeric_column("Petal.Width", dvector(gridW));
const curveFrame = logit.predict_frame(grid, "confidence");
console.log("\nPredicted-probability curve (first 5 rows):");
console.log(curveFrame.to_string(5));

console.log("\n=================== Poisson regression (log link) ===================");
const count = [];
const sepalWidth = [];
const allPetalLength = [];
for (let i = 0; i < n; i++) {
  count.push(Math.round(iris.numeric_at("Sepal.Length", i)));
  sepalWidth.push(iris.numeric_at("Sepal.Width", i));
  allPetalLength.push(iris.numeric_at("Petal.Length", i));
}
const countData = new datamunge.DataFrame();
countData.add_numeric_column("Sepal.Width", dvector(sepalWidth));
countData.add_numeric_column("Petal.Length", dvector(allPetalLength));
countData.add_numeric_column("count", dvector(count));

const poisson = new datamunge.GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson");
poisson.print_summary();
