// CMD: just TARGET=glm_iris_ex run-java
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
import datamunge.DataFrame;
import datamunge.GLM;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GlmIrisEx {
  static double[] column(DataFrame df, String name, List<Integer> rows) {
    return rows.stream().mapToDouble(i -> df.numericAt(name, i)).toArray();
  }

  static double mean(double[] values) {
    double sum = 0.0;
    for (double v : values) {
      sum += v;
    }
    return sum / values.length;
  }

  public static void main(String[] args) {
    var iris = Data.iris();
    var rows = new ArrayList<Integer>();
    for (int i = 0; i < iris.nrows(); i++) {
      rows.add(i);
    }

    // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
    // the other two on these predictors, which sends logistic regression's coefficients toward
    // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
    // binary split for a demo.
    var kept = new ArrayList<Integer>();
    for (int i : rows) {
      String sp = iris.stringAt("Species", i);
      if (sp.equals("versicolor") || sp.equals("virginica")) {
        kept.add(i);
      }
    }
    double[] isVirginica =
        kept.stream()
            .mapToDouble(i -> iris.stringAt("Species", i).equals("virginica") ? 1.0 : 0.0)
            .toArray();
    double[] petalLength = column(iris, "Petal.Length", kept);
    double[] petalWidth = column(iris, "Petal.Width", kept);

    var sub = new DataFrame();
    sub.addNumericColumn("Petal.Length", petalLength);
    sub.addNumericColumn("Petal.Width", petalWidth);
    sub.addNumericColumn("is_virginica", isVirginica);

    System.out.println(
        "=================== Logistic regression (binomial, logit link) ===================");
    var logit = new GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
    logit.printSummary();

    int n = isVirginica.length;
    int correct = 0;
    double[] fitted = logit.fittedValues();
    for (int i = 0; i < n; i++) {
      if ((fitted[i] >= 0.5) == (isVirginica[i] >= 0.5)) {
        correct++;
      }
    }
    System.out.println(
        "\nResubstitution accuracy at 0.5 threshold: " + (100.0 * correct / n) + "%");

    logit.saveDiagnosticPlots("glm_logistic_iris");
    System.out.println(
        "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,"
            + "residuals_vs_leverage}.svg");

    // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
    // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
    int gridN = 100;
    double widthMean = mean(petalWidth);
    double plMin = Arrays.stream(petalLength).min().getAsDouble() - 0.3;
    double plMax = Arrays.stream(petalLength).max().getAsDouble() + 0.3;
    double[] gridX = new double[gridN];
    for (int i = 0; i < gridN; i++) {
      gridX[i] = plMin + (plMax - plMin) * i / (gridN - 1);
    }
    double[] gridWidth = new double[gridN];
    Arrays.fill(gridWidth, widthMean);
    var grid = new DataFrame();
    grid.addNumericColumn("Petal.Length", gridX);
    grid.addNumericColumn("Petal.Width", gridWidth);
    var curveFrame = logit.predictFrame(grid, "confidence");
    System.out.println("\nPredicted-probability curve (first 5 rows):");
    System.out.println(curveFrame.toString(5));

    // Poisson regression, for contrast: same IRLS engine, different family/link.
    System.out.println(
        "\n=================== Poisson regression (log link) ===================");
    var countData = new DataFrame();
    countData.addNumericColumn("Sepal.Width", column(iris, "Sepal.Width", rows));
    countData.addNumericColumn("Petal.Length", column(iris, "Petal.Length", rows));
    countData.addNumericColumn(
        "count",
        rows.stream()
            .mapToDouble(i -> Math.floor(iris.numericAt("Sepal.Length", i) + 0.5))
            .toArray());
    var poisson = new GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson");
    poisson.printSummary();
  }
}
