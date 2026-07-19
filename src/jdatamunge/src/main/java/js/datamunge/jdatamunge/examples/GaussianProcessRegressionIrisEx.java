package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.GaussianProcessRegression;

public class GaussianProcessRegressionIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    String formula = "Petal.Length ~ Petal.Width";

    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols");
    System.out.println("formula: " + formula + "\n");

    var model = new GaussianProcessRegression(iris, formula);
    model.print_summary();

    model.plot_fit(iris).save("gpr_iris_fit.svg");
    model.plot_length_scale_profile().save("gpr_iris_length_scale_profile.svg");
    System.out.println("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

    var query = new DataFrame();
    query.add_numeric_column("Petal.Width", new DVector(new double[] {0.2, 1.3, 2.5, 10.0}));
    var detail = model.predict_frame(query, "confidence");
    System.out.println("\nPredictions with 95% confidence intervals:");
    System.out.println(detail.to_string());
    System.out.println("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)");
  }
}
