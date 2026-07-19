using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void Main() {
    var iris = DataFrame.iris();

    var isVirginica = new List<double>();
    var petalLength = new List<double>();
    var petalWidth = new List<double>();
    uint n = iris.nrows();
    for (uint i = 0; i < n; i++) {
      var species = iris.string_at("Species", i);
      if (species == "versicolor" || species == "virginica") {
        isVirginica.Add(species == "virginica" ? 1.0 : 0.0);
        petalLength.Add(iris.numeric_at("Petal.Length", i));
        petalWidth.Add(iris.numeric_at("Petal.Width", i));
      }
    }

    var sub = new DataFrame();
    sub.add_numeric_column("Petal.Length", new DVector(petalLength));
    sub.add_numeric_column("Petal.Width", new DVector(petalWidth));
    sub.add_numeric_column("is_virginica", new DVector(isVirginica));

    Console.WriteLine("=================== Logistic regression (binomial, logit link) ===================");
    var logit = new GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
    logit.print_summary();

    var fitted = logit.fitted_values();
    int correct = 0;
    for (int i = 0; i < isVirginica.Count; i++) {
      if ((fitted[i] >= 0.5) == (isVirginica[i] >= 0.5)) correct += 1;
    }
    Console.WriteLine($"\nResubstitution accuracy at 0.5 threshold: {(100.0 * correct) / isVirginica.Count}%");

    logit.save_diagnostic_plots("glm_logistic_iris");
    Console.WriteLine("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

    double widthMean = petalWidth.Average();
    int gridN = 100;
    double plMin = petalLength.Min() - 0.3;
    double plMax = petalLength.Max() + 0.3;
    var gridX = new List<double>();
    var gridW = new List<double>();
    for (int i = 0; i < gridN; i++) {
      gridX.Add(plMin + (plMax - plMin) * i / (gridN - 1));
      gridW.Add(widthMean);
    }
    var grid = new DataFrame();
    grid.add_numeric_column("Petal.Length", new DVector(gridX));
    grid.add_numeric_column("Petal.Width", new DVector(gridW));
    var curveFrame = logit.predict_frame(grid, "confidence");
    Console.WriteLine("\nPredicted-probability curve (first 5 rows):");
    Console.WriteLine(curveFrame.to_string(5));

    Console.WriteLine("\n=================== Poisson regression (log link) ===================");
    var count = new List<double>();
    var sepalWidth = new List<double>();
    var allPetalLength = new List<double>();
    for (uint i = 0; i < n; i++) {
      count.Add(Math.Round(iris.numeric_at("Sepal.Length", i)));
      sepalWidth.Add(iris.numeric_at("Sepal.Width", i));
      allPetalLength.Add(iris.numeric_at("Petal.Length", i));
    }
    var countData = new DataFrame();
    countData.add_numeric_column("Sepal.Width", new DVector(sepalWidth));
    countData.add_numeric_column("Petal.Length", new DVector(allPetalLength));
    countData.add_numeric_column("count", new DVector(count));

    var poisson = new GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson");
    poisson.print_summary();
  }
}
