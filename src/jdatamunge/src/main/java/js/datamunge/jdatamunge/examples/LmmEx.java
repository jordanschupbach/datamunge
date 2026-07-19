package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.LMM;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LmmEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static Random rng = new Random();

  static double randn() {
    return rng.nextGaussian();
  }

  public static void run() {
    System.out.println("=================== Random intercept on a real dataset (penguins) ===================");
    var penguins = DataFrame.penguins();
    var speciesModel = new LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
    speciesModel.print_summary();

    System.out.println("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
    int nSchools = 30;
    double[] schoolIntercept = new double[nSchools];
    double[] schoolSlope = new double[nSchools];
    for (int i = 0; i < nSchools; i++) {
      schoolIntercept[i] = randn() * 6.0;
      schoolSlope[i] = randn() * 1.2;
    }

    double trueIntercept = 60.0;
    double trueSlope = 3.0;
    List<Double> school = new ArrayList<>();
    List<Double> studyHours = new ArrayList<>();
    List<Double> score = new ArrayList<>();
    for (int s = 0; s < nSchools; s++) {
      int nStudents = 15 + (int) (rng.nextDouble() * 21);
      for (int j = 0; j < nStudents; j++) {
        double hours = rng.nextDouble() * 10.0;
        double noise = randn() * 4.0;
        double sVal = trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + noise;
        school.add((double) s);
        studyHours.add(hours);
        score.add(sVal);
      }
    }

    var df = new DataFrame();
    df.add_numeric_column("school", new DVector(school));
    df.add_numeric_column("study_hours", new DVector(studyHours));
    df.add_numeric_column("score", new DVector(score));

    var model = new LMM(df, "score ~ study_hours + (1 + study_hours | school)");
    model.print_summary();

    System.out.println("\nTrue generating values: intercept=" + trueIntercept + ", slope=" + trueSlope + ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

    System.out.println("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
    var groupLabels = model.group_labels();
    for (int idx = 0; idx < 3; idx++) {
      var re = model.random_effects_for_group(idx);
      System.out.println("school " + groupLabels.get(idx) + ": intercept shift=" + re.get(0) + ", slope shift=" + re.get(1));
    }

    System.out.println("\n--- Prediction: population-level vs. school-adjusted ---");
    var newdataPopulation = new DataFrame();
    newdataPopulation.add_numeric_column("study_hours", new DVector(new double[] {5.0}));
    var newdataSchool0 = new DataFrame();
    newdataSchool0.add_numeric_column("study_hours", new DVector(new double[] {5.0}));
    newdataSchool0.add_numeric_column("school", new DVector(new double[] {0.0}));
    var predPop = model.predict(newdataPopulation);
    var predS0 = model.predict(newdataSchool0);
    System.out.println("5 study hours, unseen school:      " + predPop.get(0) + " (fixed effects only)");
    System.out.println("5 study hours, school 0 (known):    " + predS0.get(0) + " (fixed effects + school 0's BLUP)");
  }
}
