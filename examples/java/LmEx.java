// CMD: just TARGET=lm_ex run-java
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
import datamunge.DataFrame;
import datamunge.LM;
import java.util.List;

public class LmEx {
  public static void main(String[] args) {
    var cars = new DataFrame();
    cars.addNumericColumn(
        "hp", new double[] {110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0});
    cars.addNumericColumn(
        "wt", new double[] {2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44});
    cars.addStringColumn(
        "transmission",
        List.of(
            "manual", "manual", "manual", "automatic", "automatic",
            "automatic", "automatic", "automatic", "automatic", "automatic"));
    cars.addNumericColumn(
        "mpg", new double[] {21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2});

    System.out.println("Fitting: mpg ~ hp + wt + transmission\n");
    var model = new LM(cars, "mpg ~ hp + wt + transmission");
    // summary() returns the text printSummary() writes to native stdout, so going through
    // System.out keeps it in order with the surrounding println calls.
    System.out.print(model.summary());

    System.out.println("\nSequential ANOVA:\n" + model.anova());

    var newcars = new DataFrame();
    newcars.addNumericColumn("hp", new double[] {150.0, 90.0});
    newcars.addNumericColumn("wt", new double[] {3.0, 2.5});
    newcars.addStringColumn("transmission", List.of("manual", "automatic"));
    System.out.println(
        "\nPredictions with 95% confidence intervals:\n"
            + model.predictFrame(newcars, "confidence"));

    model.saveDiagnosticPlots("lm_ex_diagnostics");
    System.out.println("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
  }
}
