// CMD: just TARGET=pca_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
using Datamunge;

var iris = Data.Iris()!;
string[] featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

// species labels, for the grouped (colored-by-species) scores plot
var species = Enumerable.Range(0, (int)iris.Nrows()).Select(i => iris.StringAt("Species", (ulong)i)).ToList();

Console.WriteLine("=================== PCA on iris (scaled) ===================");
var pca = new PCA(iris, featureColumns, true, true);
pca.PrintSummary();

Console.WriteLine("\nPC1 loadings (which original features drive it):");
foreach (var (name, loading) in pca.FeatureNames().Zip(pca.ComponentLoadings(0)))
    Console.WriteLine($"  {name}: {loading:F4}");

Console.WriteLine($"\nTraining scores as a DataFrame:\n{pca.ScoresFrame()}");

pca.PlotScoresGrouped(species).SaveSvg("pca_iris_scores.svg");
Console.WriteLine("\nScores scatter (colored by species) saved as pca_iris_scores.svg");

pca.PlotScree().SaveSvg("pca_iris_scree.svg");
Console.WriteLine("Scree plot saved as pca_iris_scree.svg");

Console.WriteLine("\n=================== PCA on iris (unscaled) ===================");
var unscaled = new PCA(iris, featureColumns, true, false);
Console.WriteLine($"PC1 explains {100 * unscaled.ExplainedVarianceRatio()[0]:F2}% of variance (vs. {100 * pca.ExplainedVarianceRatio()[0]:F2}% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.");
