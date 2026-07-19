const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

const FORMULA = "Petal.Length ~ Petal.Width";

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols`);
console.log(`formula: ${FORMULA}\n`);

const model = new datamunge.GaussianProcessRegression(iris, FORMULA);
model.print_summary();

model.plot_fit(iris).save("gpr_iris_fit.svg");
model.plot_length_scale_profile().save("gpr_iris_length_scale_profile.svg");
console.log("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

const query = new datamunge.DataFrame();
query.add_numeric_column("Petal.Width", dvector([0.2, 1.3, 2.5, 10.0]));
const detail = model.predict_frame(query, "confidence");
console.log("\nPredictions with 95% confidence intervals:");
console.log(detail.to_string());
console.log("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)");
