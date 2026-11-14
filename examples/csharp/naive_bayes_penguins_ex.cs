// CMD: just TARGET=naive_bayes_penguins_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
using Datamunge;

// Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
// when fitting, matching every other formula-based model in this library.
var penguins = Data.Penguins()!;
Console.WriteLine($"penguins: {penguins.Nrows()} rows x {penguins.Ncols()} cols\n");

// A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
// predictors in one formula -- each modeled independently given the class, per the naive
// Bayes assumption.
var model = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
model.PrintSummary();

Console.WriteLine($"\nConfusion matrix (rows = actual, cols = predicted):\n{model.ConfusionMatrix()}");

Console.WriteLine("\nMisclassified rows:");
var predictions = model.Predict(penguins);
var misclassified = 0;
string[] columns = ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"];
for (ulong i = 0; i < penguins.Nrows(); i++)
{
    if (columns.Any(c => penguins.IsNull(c, i))) continue;
    var (actual, predicted) = (penguins.StringAt("species", i), predictions[(int)i]);
    if (predicted == actual) continue;
    misclassified++;
    Console.WriteLine($"  row {i}: bill_length={penguins.NumericAt("bill_length_mm", i)} bill_depth={penguins.NumericAt("bill_depth_mm", i)} " +
                      $"island={penguins.StringAt("island", i)} sex={penguins.StringAt("sex", i)}  actual={actual}  predicted={predicted}");
}
Console.WriteLine($"{misclassified} misclassified (of {penguins.Nrows()} rows, some incomplete)");

// PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
// two-predictor model (bill measurements alone) just for visualization.
var billOnly = new NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm");
Console.WriteLine($"\nbill-measurements-only model training accuracy: {100 * billOnly.TrainingAccuracy()}%");
billOnly.PlotClassification(penguins, "bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_classification.svg");
billOnly.PlotDecisionRegions("bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_decision_regions.svg");
Console.WriteLine("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
