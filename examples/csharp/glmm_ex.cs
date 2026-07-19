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
    Console.WriteLine("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
    var penguins = DataFrame.penguins();
    var isMale = new List<double>();
    var bodyMass = new List<double>();
    var island = new List<string>();
    uint n = penguins.nrows();
    for (uint i = 0; i < n; i++) {
      if (!penguins.is_null("sex", i) && !penguins.is_null("body_mass_g", i) && !penguins.is_null("island", i)) {
        isMale.Add(penguins.string_at("sex", i) == "male" ? 1.0 : 0.0);
        bodyMass.Add(penguins.numeric_at("body_mass_g", i));
        island.Add(penguins.string_at("island", i));
      }
    }

    var sexDf = new DataFrame();
    sexDf.add_numeric_column("is_male", new DVector(isMale));
    sexDf.add_numeric_column("body_mass_g", new DVector(bodyMass));
    sexDf.add_string_column("island", new SVector(island));

    var sexModel = new GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial");
    sexModel.print_summary();

    Console.WriteLine("\n=================== Poisson mixed model on simulated multi-site count data ===================");
    int nStores = 25;
    var storeEffect = new List<double>();
    for (int i = 0; i < nStores; i++) storeEffect.Add(Randn() * 0.4);

    double trueIntercept = 2.0;
    double trueSlope = 0.3;
    var store = new List<double>();
    var promo = new List<double>();
    var visits = new List<double>();
    for (int s = 0; s < nStores; s++) {
      int nDays = 15 + (int)(rng.NextDouble() * 11);
      for (int d = 0; d < nDays; d++) {
        double promoIntensity = rng.NextDouble() * 3.0;
        double lam = Math.Exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
        double lThresh = Math.Exp(-lam);
        int k = 0;
        double p = 1.0;
        do {
          k += 1;
          p *= rng.NextDouble();
        } while (p > lThresh);
        store.Add(s);
        promo.Add(promoIntensity);
        visits.Add(k - 1);
      }
    }

    var df = new DataFrame();
    df.add_numeric_column("store", new DVector(store));
    df.add_numeric_column("promo", new DVector(promo));
    df.add_numeric_column("visits", new DVector(visits));

    var storeModel = new GLMM(df, "visits ~ promo + (1 | store)", "poisson");
    storeModel.print_summary();

    Console.WriteLine($"\nTrue generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD (log scale)=0.4");

    Console.WriteLine("\n--- BLUPs for a few stores ---");
    var groupLabels = storeModel.group_labels();
    for (uint idx = 0; idx < 3; idx++) {
      var re = storeModel.random_effects_for_group(idx);
      Console.WriteLine($"store {groupLabels[(int)idx]}: intercept shift={re[0]}");
    }

    Console.WriteLine("\n--- Prediction: population-level vs. store-adjusted ---");
    var newdataPopulation = new DataFrame();
    newdataPopulation.add_numeric_column("promo", new DVector(new double[] { 1.5 }));
    var newdataStore0 = new DataFrame();
    newdataStore0.add_numeric_column("promo", new DVector(new double[] { 1.5 }));
    newdataStore0.add_numeric_column("store", new DVector(new double[] { 0.0 }));
    var predPop = storeModel.predict(newdataPopulation);
    var predS0 = storeModel.predict(newdataStore0);
    Console.WriteLine($"promo=1.5, unseen store:   {predPop[0]} expected visits (fixed effects only)");
    Console.WriteLine($"promo=1.5, store 0 (known): {predS0[0]} expected visits (fixed effects + store 0's BLUP)");
  }
}
