const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

console.log("=== RBF kernel (default) ===");
const rbfModel = new datamunge.SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
rbfModel.print_summary();
console.log("\nConfusion matrix (rows = actual, cols = predicted):");
console.log(rbfModel.confusion_matrix().to_string());

console.log("\n=== Linear kernel, for comparison ===");
const linearModel = new datamunge.SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear");
console.log(`Training accuracy: ${linearModel.training_accuracy() * 100.0}%`);
console.log(`Support vectors: ${linearModel.num_support_vectors()}`);

const newdata = new datamunge.DataFrame();
newdata.add_numeric_column("Sepal.Length", dvector([5.1, 6.0, 6.5, 6.2]));
newdata.add_numeric_column("Sepal.Width", dvector([3.5, 2.7, 3.0, 2.8]));
newdata.add_numeric_column("Petal.Length", dvector([1.4, 4.5, 5.5, 4.8]));
newdata.add_numeric_column("Petal.Width", dvector([0.2, 1.5, 2.0, 1.8]));

console.log("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):");
console.log(rbfModel.predict_frame(newdata).to_string());
