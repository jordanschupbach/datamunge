// CMD: just TARGET=glmm_ex run-java
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
import datamunge.DataFrame;
import datamunge.GLMM;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class GlmmEx {
  // Java's own generator, so the simulated data (and the fit below) differ from other languages'.
  static final Random RNG = new Random(4242);

  // Box-Muller; 1 - u keeps log() away from 0.
  static double gauss() {
    double u1 = 1 - RNG.nextDouble();
    double u2 = RNG.nextDouble();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Knuth's Poisson sampler.
  static int rpois(double lambda) {
    double threshold = Math.exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
      k++;
      p *= RNG.nextDouble();
    } while (p > threshold);
    return k - 1;
  }

  public static void main(String[] args) {
    System.out.println(
        "=================== Binomial (logistic) mixed model on a real dataset (penguins)"
            + " ===================");
    // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
    // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
    // is not tied to location, so this is a well-behaved fit.
    var penguins = Data.penguins();
    var isMale = DoubleStream.builder();
    var bodyMass = DoubleStream.builder();
    List<String> island = new ArrayList<>();
    for (long i = 0; i < penguins.nrows(); i++) {
      if (penguins.isNull("sex", i)
          || penguins.isNull("body_mass_g", i)
          || penguins.isNull("island", i)) {
        continue;
      }
      isMale.add(penguins.stringAt("sex", i).equals("male") ? 1.0 : 0.0);
      bodyMass.add(penguins.numericAt("body_mass_g", i));
      island.add(penguins.stringAt("island", i));
    }
    var sexDf = new DataFrame();
    sexDf.addNumericColumn("is_male", isMale.build().toArray());
    sexDf.addNumericColumn("body_mass_g", bodyMass.build().toArray());
    sexDf.addStringColumn("island", island);
    new GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial").printSummary();

    System.out.println(
        "\n=================== Poisson mixed model on simulated multi-site count data"
            + " ===================");
    // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
    // the baseline traffic and the promo's effectiveness vary by store.
    int nStores = 25;
    double[] storeEffect = DoubleStream.generate(() -> gauss() * 0.4).limit(nStores).toArray();
    double trueIntercept = 2.0;
    double trueSlope = 0.3;

    var store = DoubleStream.builder();
    var promo = DoubleStream.builder();
    var visits = DoubleStream.builder();
    for (int s = 0; s < nStores; s++) {
      int nDays = 15 + RNG.nextInt(11);
      for (int day = 0; day < nDays; day++) {
        double promoIntensity = RNG.nextDouble() * 3;
        double lambda = Math.exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
        store.add(s);
        promo.add(promoIntensity);
        visits.add(rpois(lambda));
      }
    }
    var df = new DataFrame();
    df.addNumericColumn("store", store.build().toArray());
    df.addNumericColumn("promo", promo.build().toArray());
    df.addNumericColumn("visits", visits.build().toArray());
    var storeModel = new GLMM(df, "visits ~ promo + (1 | store)", "poisson");

    storeModel.printSummary();

    System.out.println(
        "\nTrue generating values: intercept="
            + trueIntercept
            + ", slope="
            + trueSlope
            + ", random-intercept SD (log scale)=0.4");

    System.out.println("\n--- BLUPs for a few stores ---");
    List<String> labels = storeModel.groupLabels();
    for (int idx = 0; idx < 3; idx++) {
      System.out.println(
          "store "
              + labels.get(idx)
              + ": intercept shift="
              + storeModel.randomEffectsForGroup(idx)[0]);
    }

    System.out.println("\n--- Prediction: population-level vs. store-adjusted ---");
    var population = new DataFrame();
    var store0 = new DataFrame();
    population.addNumericColumn("promo", new double[] {1.5});
    store0.addNumericColumn("promo", new double[] {1.5});
    store0.addNumericColumn("store", new double[] {0.0});
    System.out.println(
        "promo=1.5, unseen store:   "
            + storeModel.predict(population)[0]
            + " expected visits (fixed effects only)");
    System.out.println(
        "promo=1.5, store 0 (known): "
            + storeModel.predict(store0)[0]
            + " expected visits (fixed effects + store 0's BLUP)");
  }
}
