using System;

class Program {
  static void Main() {
    var hp = new double[] { 110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0 };
    var wt = new double[] { 2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44 };
    var transmission = new string[] { "manual", "manual", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic" };
    var mpg = new double[] { 21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2 };

    var cars = new DataFrame();
    cars.add_numeric_column("hp", new DVector(hp));
    cars.add_numeric_column("wt", new DVector(wt));
    cars.add_string_column("transmission", new SVector(transmission));
    cars.add_numeric_column("mpg", new DVector(mpg));

    Console.WriteLine("Fitting: mpg ~ hp + wt + transmission\n");
    var model = new LM(cars, "mpg ~ hp + wt + transmission");
    model.print_summary();

    Console.WriteLine("\nSequential ANOVA:");
    Console.WriteLine(model.anova().to_string());

    var newcars = new DataFrame();
    newcars.add_numeric_column("hp", new DVector(new double[] { 150.0, 90.0 }));
    newcars.add_numeric_column("wt", new DVector(new double[] { 3.0, 2.5 }));
    newcars.add_string_column("transmission", new SVector(new string[] { "manual", "automatic" }));

    var frame = model.predict_frame(newcars, "confidence");
    Console.WriteLine("\nPredictions with 95% confidence intervals:");
    Console.WriteLine(frame.to_string());

    model.save_diagnostic_plots("lm_ex_diagnostics");
    Console.WriteLine("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
  }
}
