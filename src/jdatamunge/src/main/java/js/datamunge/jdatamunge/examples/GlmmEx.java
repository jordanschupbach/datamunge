package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.SVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.GLMM;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GlmmEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static Random rng = new Random();

  static double randn() {
    return rng.nextGaussian();
  }

  public static void run() {
    System.out.println("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
    var penguins = DataFrame.penguins();
    List<Double> isMale = new ArrayList<>();
    List<Double> bodyMass = new ArrayList<>();
    List<String> island = new ArrayList<>();
    long n = penguins.nrows();
    for (long i = 0; i < n; i++) {
      if (!penguins.is_null("sex", i) && !penguins.is_null("body_mass_g", i) && !penguins.is_null("island", i)) {
        isMale.add(penguins.string_at("sex", i).equals("male") ? 1.0 : 0.0);
        bodyMass.add(penguins.numeric_at("body_mass_g", i));
        island.add(penguins.string_at("island", i));
      }
    }

    var sexDf = new DataFrame();
    sexDf.add_numeric_column("is_male", new DVector(isMale));
    sexDf.add_numeric_column("body_mass_g", new DVector(bodyMass));
    sexDf.add_string_column("island", new SVector(island));

    var sexModel = new GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial");
    sexModel.print_summary();

    System.out.println("\n=================== Poisson mixed model on simulated multi-site count data ===================");
    int nStores = 25;
    double[] storeEffect = new double[nStores];
    for (int i = 0; i < nStores; i++) storeEffect[i] = randn() * 0.4;

    double trueIntercept = 2.0;
    double trueSlope = 0.3;
    List<Double> store = new ArrayList<>();
    List<Double> promo = new ArrayList<>();
    List<Double> visits = new ArrayList<>();
    for (int s = 0; s < nStores; s++) {
      int nDays = 15 + (int) (rng.nextDouble() * 11);
      for (int d = 0; d < nDays; d++) {
        double promoIntensity = rng.nextDouble() * 3.0;
        double lam = Math.exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
        double lThresh = Math.exp(-lam);
        int k = 0;
        double p = 1.0;
        do {
          k++;
          p *= rng.nextDouble();
        } while (p > lThresh);
        store.add((double) s);
        promo.add(promoIntensity);
        visits.add((double) (k - 1));
      }
    }

    var df = new DataFrame();
    df.add_numeric_column("store", new DVector(store));
    df.add_numeric_column("promo", new DVector(promo));
    df.add_numeric_column("visits", new DVector(visits));

    var storeModel = new GLMM(df, "visits ~ promo + (1 | store)", "poisson");
    storeModel.print_summary();

    System.out.println("\nTrue generating values: intercept=" + trueIntercept + ", slope=" + trueSlope + ", random-intercept SD (log scale)=0.4");

    System.out.println("\n--- BLUPs for a few stores ---");
    var groupLabels = storeModel.group_labels();
    for (int idx = 0; idx < 3; idx++) {
      var re = storeModel.random_effects_for_group(idx);
      System.out.println("store " + groupLabels.get(idx) + ": intercept shift=" + re.get(0));
    }

    System.out.println("\n--- Prediction: population-level vs. store-adjusted ---");
    var newdataPopulation = new DataFrame();
    newdataPopulation.add_numeric_column("promo", new DVector(new double[] {1.5}));
    var newdataStore0 = new DataFrame();
    newdataStore0.add_numeric_column("promo", new DVector(new double[] {1.5}));
    newdataStore0.add_numeric_column("store", new DVector(new double[] {0.0}));
    var predPop = storeModel.predict(newdataPopulation);
    var predS0 = storeModel.predict(newdataStore0);
    System.out.println("promo=1.5, unseen store:   " + predPop.get(0) + " expected visits (fixed effects only)");
    System.out.println("promo=1.5, store 0 (known): " + predS0.get(0) + " expected visits (fixed effects + store 0's BLUP)");
  }
}
