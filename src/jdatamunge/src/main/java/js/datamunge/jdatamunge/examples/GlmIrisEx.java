package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.GLM;

import java.util.ArrayList;
import java.util.List;

public class GlmIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();

    List<Double> isVirginica = new ArrayList<>();
    List<Double> petalLength = new ArrayList<>();
    List<Double> petalWidth = new ArrayList<>();
    long n = iris.nrows();
    for (long i = 0; i < n; i++) {
      String species = iris.string_at("Species", i);
      if (species.equals("versicolor") || species.equals("virginica")) {
        isVirginica.add(species.equals("virginica") ? 1.0 : 0.0);
        petalLength.add(iris.numeric_at("Petal.Length", i));
        petalWidth.add(iris.numeric_at("Petal.Width", i));
      }
    }

    var sub = new DataFrame();
    sub.add_numeric_column("Petal.Length", new DVector(petalLength));
    sub.add_numeric_column("Petal.Width", new DVector(petalWidth));
    sub.add_numeric_column("is_virginica", new DVector(isVirginica));

    System.out.println("=================== Logistic regression (binomial, logit link) ===================");
    var logit = new GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
    logit.print_summary();

    var fitted = logit.fitted_values();
    int correct = 0;
    int nn = isVirginica.size();
    for (int i = 0; i < nn; i++) {
      if ((fitted.get(i) >= 0.5) == (isVirginica.get(i) >= 0.5)) correct++;
    }
    System.out.println("\nResubstitution accuracy at 0.5 threshold: " + (100.0 * correct / nn) + "%");

    logit.save_diagnostic_plots("glm_logistic_iris");
    System.out.println("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

    double widthMean = 0;
    for (double w : petalWidth) widthMean += w;
    widthMean /= nn;
    int gridN = 100;
    double plMin = petalLength.get(0), plMax = petalLength.get(0);
    for (double v : petalLength) {
      plMin = Math.min(plMin, v);
      plMax = Math.max(plMax, v);
    }
    plMin -= 0.3;
    plMax += 0.3;
    List<Double> gridX = new ArrayList<>();
    List<Double> gridW = new ArrayList<>();
    for (int i = 0; i < gridN; i++) {
      gridX.add(plMin + (plMax - plMin) * i / (gridN - 1));
      gridW.add(widthMean);
    }
    var grid = new DataFrame();
    grid.add_numeric_column("Petal.Length", new DVector(gridX));
    grid.add_numeric_column("Petal.Width", new DVector(gridW));
    var curveFrame = logit.predict_frame(grid, "confidence");
    System.out.println("\nPredicted-probability curve (first 5 rows):");
    System.out.println(curveFrame.to_string(5));

    System.out.println("\n=================== Poisson regression (log link) ===================");
    List<Double> count = new ArrayList<>();
    List<Double> sepalWidth = new ArrayList<>();
    List<Double> allPetalLength = new ArrayList<>();
    for (long i = 0; i < n; i++) {
      count.add((double) Math.round(iris.numeric_at("Sepal.Length", i)));
      sepalWidth.add(iris.numeric_at("Sepal.Width", i));
      allPetalLength.add(iris.numeric_at("Petal.Length", i));
    }
    var countData = new DataFrame();
    countData.add_numeric_column("Sepal.Width", new DVector(sepalWidth));
    countData.add_numeric_column("Petal.Length", new DVector(allPetalLength));
    countData.add_numeric_column("count", new DVector(count));

    var poisson = new GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson");
    poisson.print_summary();
  }
}
