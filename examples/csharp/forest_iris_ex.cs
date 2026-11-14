// CMD: just TARGET=forest_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine("\nConfusion matrix (rows = actual, cols = predicted):");
Console.WriteLine(model.ConfusionMatrix());

Console.WriteLine("\nMisclassified rows:");
var predictions = model.Predict(iris);
var n = iris.Nrows();
var misclassified = 0;
for (ulong i = 0; i < n; i++)
{
    var (actual, predicted) = (iris.StringAt("Species", i), predictions[(int)i]);
    if (predicted == actual) continue;
    misclassified++;
    Console.WriteLine($"  row {i}: Petal.Length={iris.NumericAt("Petal.Length", i)} Petal.Width={iris.NumericAt("Petal.Width", i)}  actual={actual}  predicted={predicted}");
}
Console.WriteLine($"{misclassified} of {n} misclassified ({100.0 * misclassified / n}%)");

model.PlotClassification(iris, "Petal.Length", "Petal.Width").Save("forest_iris_classification.svg");
model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions.svg");
Console.WriteLine("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

// A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
// are added -- the defining random forest effect a single decision tree cannot demonstrate.
// new RandomForestClassifier(data, formula, nTrees, maxDepth, minSamplesSplit, minSamplesLeaf,
//                            maxFeatures, criterion, bootstrap, sampleFraction, seed)
var smallForest = new RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
Console.WriteLine($"\n5-tree forest:   training accuracy={smallForest.TrainingAccuracy() * 100}%  OOB accuracy={smallForest.OobAccuracy() * 100}%");
Console.WriteLine($"100-tree forest: training accuracy={model.TrainingAccuracy() * 100}%  OOB accuracy={model.OobAccuracy() * 100}%");
smallForest.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions_5trees.svg");
Console.WriteLine("Saved forest_iris_decision_regions_5trees.svg");
