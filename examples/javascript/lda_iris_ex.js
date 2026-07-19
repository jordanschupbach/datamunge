const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
model.print_summary();

console.log("\nConfusion matrix (rows = actual, cols = predicted):");
console.log(model.confusion_matrix().to_string());

const newdata = new datamunge.DataFrame();
newdata.add_numeric_column("Sepal.Length", dvector([5.1, 6.0, 6.5, 6.2]));
newdata.add_numeric_column("Sepal.Width", dvector([3.5, 2.7, 3.0, 2.8]));
newdata.add_numeric_column("Petal.Length", dvector([1.4, 4.5, 5.5, 4.8]));
newdata.add_numeric_column("Petal.Width", dvector([0.2, 1.5, 2.0, 1.8]));

console.log("\nPredictions for new flowers:");
console.log(model.predict_frame(newdata).to_string());

model.save_discriminant_plot("lda_iris_discriminants.svg");
console.log("\nSaved discriminant plot as lda_iris_discriminants.svg");
