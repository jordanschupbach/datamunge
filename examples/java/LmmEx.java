// CMD: just TARGET=lmm_ex run-java
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
import datamunge.DataFrame;
import datamunge.LMM;
import datamunge.data.Data;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class LmmEx {
  // Java's own generator, so the simulated data (and the fit below) differ from other languages'.
  static final Random RNG = new Random(2024);

  // (0, 1]: avoids log(0) in the Box-Muller transform.
  static double uniform() {
    return 1 - RNG.nextDouble();
  }

  static double gauss() {
    return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  }

  public static void main(String[] args) {
    System.out.println(
        "=================== Random intercept on a real dataset (penguins) ===================");
    new LMM(Data.penguins(), "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
        .printSummary();

    System.out.println(
        "\n=================== Random intercept + slope on a simulated multi-school dataset"
            + " ===================");
    // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
    // and the return on an extra hour of study vary by school -- the textbook case for a
    // random-intercept-and-slope model.
    int nSchools = 30;
    double[] schoolIntercept = DoubleStream.generate(() -> gauss() * 6).limit(nSchools).toArray();
    double[] schoolSlope = DoubleStream.generate(() -> gauss() * 1.2).limit(nSchools).toArray();
    double trueIntercept = 60.0;
    double trueSlope = 3.0;

    var school = DoubleStream.builder();
    var studyHours = DoubleStream.builder();
    var score = DoubleStream.builder();
    for (int s = 0; s < nSchools; s++) {
      int nStudents = 15 + RNG.nextInt(21);
      for (int j = 0; j < nStudents; j++) {
        double hours = RNG.nextDouble() * 10;
        school.add(s);
        studyHours.add(hours);
        score.add(
            trueIntercept
                + schoolIntercept[s]
                + (trueSlope + schoolSlope[s]) * hours
                + gauss() * 4);
      }
    }
    var df = new DataFrame();
    df.addNumericColumn("school", school.build().toArray());
    df.addNumericColumn("study_hours", studyHours.build().toArray());
    df.addNumericColumn("score", score.build().toArray());

    var model = new LMM(df, "score ~ study_hours + (1 + study_hours | school)");
    model.printSummary();

    System.out.println(
        "\nTrue generating values: intercept="
            + trueIntercept
            + ", slope="
            + trueSlope
            + ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

    System.out.println("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
    List<String> labels = model.groupLabels();
    for (int idx = 0; idx < 3; idx++) {
      double[] shifts = model.randomEffectsForGroup(idx);
      System.out.println(
          "school "
              + labels.get(idx)
              + ": intercept shift="
              + shifts[0]
              + ", slope shift="
              + shifts[1]);
    }

    System.out.println("\n--- Prediction: population-level vs. school-adjusted ---");
    var population = new DataFrame();
    var school0 = new DataFrame();
    population.addNumericColumn("study_hours", new double[] {5.0});
    school0.addNumericColumn("study_hours", new double[] {5.0});
    school0.addNumericColumn("school", new double[] {0.0});
    System.out.println(
        "5 study hours, unseen school:      "
            + model.predict(population)[0]
            + " (fixed effects only)");
    System.out.println(
        "5 study hours, school 0 (known):    "
            + model.predict(school0)[0]
            + " (fixed effects + school 0's BLUP)");
  }
}
