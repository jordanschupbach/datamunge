// CMD: just TARGET=bspline_ex run-java
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
import datamunge.DataFrame;
import datamunge.LM;
import java.util.Locale;

public class BsplineEx {
  public static void main(String[] args) {
    // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
    double[] x = new double[64];
    double[] y = new double[64];
    double[] z = new double[64];
    int k = 0;
    for (int iy = 0; iy < 8; iy++) {
      for (int ix = 0; ix < 8; ix++) {
        double xv = ix / 7.0;
        double yv = iy / 7.0;
        x[k] = xv;
        y[k] = yv;
        z[k] = Math.sin(xv) + yv * yv;
        k++;
      }
    }
    var data = new DataFrame();
    data.addNumericColumn("x", x);
    data.addNumericColumn("y", y);
    data.addNumericColumn("z", z);

    var surface = new LM(data, "z ~ bs(x, y)");
    System.out.println("Fitted z ~ bs(x, y)");
    surface.printSummary();

    var newPoints = new DataFrame();
    newPoints.addNumericColumn("x", new double[] {0.25, 0.75});
    newPoints.addNumericColumn("y", new double[] {0.50, 0.25});
    double[] pred = surface.predict(newPoints);
    System.out.println(String.format(Locale.ROOT, "Predictions: [%.6f, %.6f]", pred[0], pred[1]));
  }
}
