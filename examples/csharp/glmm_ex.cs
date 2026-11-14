// CMD: just TARGET=glmm_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
using Datamunge;

var rng = new Random(4242);
double Uniform() => rng.NextDouble();
double Gauss()
{
    double u1 = 1 - Uniform(), u2 = Uniform();  // 1 - u keeps log() away from 0
    return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
}

Console.WriteLine("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
// Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
// confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
// is not tied to location, so this is a well-behaved fit.
{
    var penguins = Data.Penguins()!;
    List<double> isMale = [], bodyMass = [];
    List<string> island = [];
    for (ulong i = 0; i < penguins.Nrows(); i++)
    {
        if (new[] { "sex", "body_mass_g", "island" }.Any(c => penguins.IsNull(c, i))) continue;
        isMale.Add(penguins.StringAt("sex", i) == "male" ? 1.0 : 0.0);
        bodyMass.Add(penguins.NumericAt("body_mass_g", i));
        island.Add(penguins.StringAt("island", i));
    }
    var sexDf = new DataFrame();
    sexDf.AddNumericColumn("is_male", isMale);
    sexDf.AddNumericColumn("body_mass_g", bodyMass);
    sexDf.AddStringColumn("island", island);
    new GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial").PrintSummary();
}

Console.WriteLine("\n=================== Poisson mixed model on simulated multi-site count data ===================");
// 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
// the baseline traffic and the promo's effectiveness vary by store.
const int nStores = 25;
var storeEffect = Enumerable.Range(0, nStores).Select(_ => Gauss() * 0.4).ToArray();
const double trueIntercept = 2.0;
const double trueSlope = 0.3;

List<double> store = [], promo = [], visits = [];
for (var s = 0; s < nStores; s++)
{
    var nDays = 15 + rng.Next(0, 11);
    for (var day = 0; day < nDays; day++)
    {
        var promoIntensity = Uniform() * 3;
        var lam = Math.Exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
        // Knuth's Poisson sampler.
        var (threshold, k, p) = (Math.Exp(-lam), 0, 1.0);
        do
        {
            k++;
            p *= Uniform();
        } while (p > threshold);
        store.Add(s);
        promo.Add(promoIntensity);
        visits.Add(k - 1);
    }
}
var df = new DataFrame();
df.AddNumericColumn("store", store);
df.AddNumericColumn("promo", promo);
df.AddNumericColumn("visits", visits);
var storeModel = new GLMM(df, "visits ~ promo + (1 | store)", "poisson");

storeModel.PrintSummary();

Console.WriteLine($"\nTrue generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD (log scale)=0.4");

Console.WriteLine("\n--- BLUPs for a few stores ---");
var labels = storeModel.GroupLabels();
for (var idx = 0; idx < 3; idx++)
    Console.WriteLine($"store {labels[idx]}: intercept shift={storeModel.RandomEffectsForGroup((ulong)idx)[0]}");

Console.WriteLine("\n--- Prediction: population-level vs. store-adjusted ---");
var population = new DataFrame();
var store0 = new DataFrame();
population.AddNumericColumn("promo", [1.5]);
store0.AddNumericColumn("promo", [1.5]);
store0.AddNumericColumn("store", [0.0]);
Console.WriteLine($"promo=1.5, unseen store:   {storeModel.Predict(population)[0]} expected visits (fixed effects only)");
Console.WriteLine($"promo=1.5, store 0 (known): {storeModel.Predict(store0)[0]} expected visits (fixed effects + store 0's BLUP)");
