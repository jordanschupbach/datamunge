const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

function svector(values) {
  const out = new datamunge.SVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

const hp = [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0];
const wt = [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44];
const transmission = ["manual", "manual", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic"];
const mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2];

const cars = new datamunge.DataFrame();
cars.add_numeric_column("hp", dvector(hp));
cars.add_numeric_column("wt", dvector(wt));
cars.add_string_column("transmission", svector(transmission));
cars.add_numeric_column("mpg", dvector(mpg));

console.log("Fitting: mpg ~ hp + wt + transmission\n");
const model = new datamunge.LM(cars, "mpg ~ hp + wt + transmission");
model.print_summary();

console.log("\nSequential ANOVA:");
console.log(model.anova().to_string());

const newcars = new datamunge.DataFrame();
newcars.add_numeric_column("hp", dvector([150.0, 90.0]));
newcars.add_numeric_column("wt", dvector([3.0, 2.5]));
newcars.add_string_column("transmission", svector(["manual", "automatic"]));

const frame = model.predict_frame(newcars, "confidence");
console.log("\nPredictions with 95% confidence intervals:");
console.log(frame.to_string());

model.save_diagnostic_plots("lm_ex_diagnostics");
console.log("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
