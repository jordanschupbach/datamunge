const datamunge = require("../../index.js");

const penguins = datamunge.DataFrame.penguins();
console.log(`penguins: ${penguins.nrows()} rows x ${penguins.ncols()} cols\n`);

const model = new datamunge.NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
model.print_summary();

console.log("\nConfusion matrix (rows = actual, cols = predicted):");
console.log(model.confusion_matrix().to_string());

console.log("\nMisclassified rows:");
const predictions = model.predict(penguins);
let misclassified = 0;
const n = penguins.nrows();
for (let i = 0; i < n; i++) {
  if (
    !penguins.is_null("species", i) &&
    !penguins.is_null("bill_length_mm", i) &&
    !penguins.is_null("bill_depth_mm", i) &&
    !penguins.is_null("island", i) &&
    !penguins.is_null("sex", i)
  ) {
    const actual = penguins.string_at("species", i);
    const pred = predictions.get(i);
    if (pred !== actual) {
      misclassified += 1;
      console.log(
        `  row ${i}: bill_length=${penguins.numeric_at("bill_length_mm", i)} bill_depth=${penguins.numeric_at("bill_depth_mm", i)} island=${penguins.string_at("island", i)} sex=${penguins.string_at("sex", i)}  actual=${actual}  predicted=${pred}`,
      );
    }
  }
}
console.log(`${misclassified} misclassified (of ${n} rows, some incomplete)`);

const billOnly = new datamunge.NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
console.log(`\nbill-measurements-only model training accuracy: ${billOnly.training_accuracy() * 100.0}%`);
billOnly.plot_classification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg");
billOnly.plot_decision_regions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg");
console.log("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
