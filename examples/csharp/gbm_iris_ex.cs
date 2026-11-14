// CMD: just TARGET=gbm_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine($"\nConfusion matrix (rows = actual, cols = predicted):\n{model.ConfusionMatrix()}");

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

model.PlotTrainingDeviance().Save("gbm_iris_training_deviance.svg");
model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions.svg");
Console.WriteLine("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

// A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
// at the training loss, gradually sharpening the decision boundary.
// new GBMClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
//                   minSamplesLeaf, subsample, seed)
var few = new GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
Console.WriteLine($"\n5-round ensemble:   training accuracy={100 * few.TrainingAccuracy()}%  deviance={few.TrainingDeviance()[^1]}");
Console.WriteLine($"100-round ensemble: training accuracy={100 * model.TrainingAccuracy()}%  deviance={model.TrainingDeviance()[^1]}");
few.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions_5rounds.svg");
Console.WriteLine("Saved gbm_iris_decision_regions_5rounds.svg");
