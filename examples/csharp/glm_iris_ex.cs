// CMD: just TARGET=glm_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
using Datamunge;

var iris = Data.Iris()!;
var rows = Enumerable.Range(0, (int)iris.Nrows()).Select(i => (ulong)i).ToList();

// Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
// the other two on these predictors, which sends logistic regression's coefficients toward
// +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
// binary split for a demo.
var kept = rows.Where(i => iris.StringAt("Species", i) is "versicolor" or "virginica").ToList();
var isVirginica = kept.Select(i => iris.StringAt("Species", i) == "virginica" ? 1.0 : 0.0).ToList();
var petalLength = kept.Select(i => iris.NumericAt("Petal.Length", i)).ToList();
var petalWidth = kept.Select(i => iris.NumericAt("Petal.Width", i)).ToList();

var sub = new DataFrame();
sub.AddNumericColumn("Petal.Length", petalLength);
sub.AddNumericColumn("Petal.Width", petalWidth);
sub.AddNumericColumn("is_virginica", isVirginica);

Console.WriteLine("=================== Logistic regression (binomial, logit link) ===================");
var logit = new GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
logit.PrintSummary();

var fitted = logit.FittedValues();
var n = isVirginica.Count;
var correct = Enumerable.Range(0, n).Count(i => (fitted[i] >= 0.5) == (isVirginica[i] >= 0.5));
Console.WriteLine($"\nResubstitution accuracy at 0.5 threshold: {100.0 * correct / n}%");

logit.SaveDiagnosticPlots("glm_logistic_iris");
Console.WriteLine("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

// Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
// classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
{
    const int gridN = 100;
    var widthMean = petalWidth.Average();
    double plMin = petalLength.Min() - 0.3, plMax = petalLength.Max() + 0.3;
    var grid = new DataFrame();
    grid.AddNumericColumn("Petal.Length", Enumerable.Range(0, gridN).Select(i => plMin + (plMax - plMin) * i / (gridN - 1)));
    grid.AddNumericColumn("Petal.Width", Enumerable.Repeat(widthMean, gridN));
    var curveFrame = logit.PredictFrame(grid, "confidence")!;
    Console.WriteLine("\nPredicted-probability curve (first 5 rows):");
    Console.WriteLine(curveFrame.ToString(5));
}

// Poisson regression, for contrast: same IRLS engine, different family/link.
Console.WriteLine("\n=================== Poisson regression (log link) ===================");
var countData = new DataFrame();
countData.AddNumericColumn("Sepal.Width", rows.Select(i => iris.NumericAt("Sepal.Width", i)));
countData.AddNumericColumn("Petal.Length", rows.Select(i => iris.NumericAt("Petal.Length", i)));
countData.AddNumericColumn("count", rows.Select(i => Math.Floor(iris.NumericAt("Sepal.Length", i) + 0.5)));
new GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson").PrintSummary();
