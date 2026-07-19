using System;
using System.Collections.Generic;

class Program {
  static Random rng = new Random();

  static double Randn() {
    double u1 = rng.NextDouble();
    double u2 = rng.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
  }

  static void Main() {
    Console.WriteLine("=================== Random intercept on a real dataset (penguins) ===================");
    var penguins = DataFrame.penguins();
    var speciesModel = new LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
    speciesModel.print_summary();

    Console.WriteLine("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
    int nSchools = 30;
    var schoolIntercept = new List<double>();
    var schoolSlope = new List<double>();
    for (int i = 0; i < nSchools; i++) {
      schoolIntercept.Add(Randn() * 6.0);
      schoolSlope.Add(Randn() * 1.2);
    }

    double trueIntercept = 60.0;
    double trueSlope = 3.0;
    var school = new List<double>();
    var studyHours = new List<double>();
    var score = new List<double>();
    for (int s = 0; s < nSchools; s++) {
      int nStudents = 15 + (int)(rng.NextDouble() * 21);
      for (int j = 0; j < nStudents; j++) {
        double hours = rng.NextDouble() * 10.0;
        double noise = Randn() * 4.0;
        double sVal = trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + noise;
        school.Add(s);
        studyHours.Add(hours);
        score.Add(sVal);
      }
    }

    var df = new DataFrame();
    df.add_numeric_column("school", new DVector(school));
    df.add_numeric_column("study_hours", new DVector(studyHours));
    df.add_numeric_column("score", new DVector(score));

    var model = new LMM(df, "score ~ study_hours + (1 + study_hours | school)");
    model.print_summary();

    Console.WriteLine($"\nTrue generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

    Console.WriteLine("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
    var groupLabels = model.group_labels();
    for (uint idx = 0; idx < 3; idx++) {
      var re = model.random_effects_for_group(idx);
      Console.WriteLine($"school {groupLabels[(int)idx]}: intercept shift={re[0]}, slope shift={re[1]}");
    }

    Console.WriteLine("\n--- Prediction: population-level vs. school-adjusted ---");
    var newdataPopulation = new DataFrame();
    newdataPopulation.add_numeric_column("study_hours", new DVector(new double[] { 5.0 }));
    var newdataSchool0 = new DataFrame();
    newdataSchool0.add_numeric_column("study_hours", new DVector(new double[] { 5.0 }));
    newdataSchool0.add_numeric_column("school", new DVector(new double[] { 0.0 }));
    var predPop = model.predict(newdataPopulation);
    var predS0 = model.predict(newdataSchool0);
    Console.WriteLine($"5 study hours, unseen school:      {predPop[0]} (fixed effects only)");
    Console.WriteLine($"5 study hours, school 0 (known):    {predS0[0]} (fixed effects + school 0's BLUP)");
  }
}
