import math
import random

from pydatamunge import datamunge as dm

print("=================== Binomial (logistic) mixed model on a real dataset (penguins) "
      "===================")
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
penguins = dm.DataFrame.penguins()
is_male, body_mass, island = [], [], []
for i in range(penguins.nrows()):
    if penguins.is_null("sex", i) or penguins.is_null("body_mass_g", i) or penguins.is_null("island", i):
        continue
    is_male.append(1.0 if penguins.string_at("sex", i) == "male" else 0.0)
    body_mass.append(penguins.numeric_at("body_mass_g", i))
    island.append(penguins.string_at("island", i))

sex_df = dm.DataFrame()
sex_df.add_numeric_column("is_male", is_male)
sex_df.add_numeric_column("body_mass_g", body_mass)
sex_df.add_string_column("island", island)

sex_model = dm.GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")
sex_model.print_summary()

print("\n=================== Poisson mixed model on simulated multi-site count data "
      "===================")
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
rng = random.Random(4242)
n_stores = 25
store_effect = [rng.normalvariate(0.0, 0.4) for _ in range(n_stores)]  # true random-intercept SD (log scale)

true_intercept, true_slope = 2.0, 0.3
store, promo, visits = [], [], []
for s in range(n_stores):
    n_days = rng.randint(15, 25)
    for _ in range(n_days):
        promo_intensity = rng.uniform(0.0, 3.0)
        lam = math.exp(true_intercept + store_effect[s] + true_slope * promo_intensity)
        # Knuth's Poisson sampler.
        l_thresh, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= l_thresh:
                break
        store.append(float(s))
        promo.append(promo_intensity)
        visits.append(float(k - 1))

df = dm.DataFrame()
df.add_numeric_column("store", store)
df.add_numeric_column("promo", promo)
df.add_numeric_column("visits", visits)

store_model = dm.GLMM(df, "visits ~ promo + (1 | store)", "poisson")
store_model.print_summary()

print(f"\nTrue generating values: intercept={true_intercept}, slope={true_slope}, "
      "random-intercept SD (log scale)=0.4")

print("\n--- BLUPs for a few stores ---")
group_labels = store_model.group_labels()
for idx in (0, 1, 2):
    print(f"store {group_labels[idx]}: intercept shift={store_model.random_effects_for_group(idx)[0]}")

print("\n--- Prediction: population-level vs. store-adjusted ---")
newdata_population = dm.DataFrame()
newdata_population.add_numeric_column("promo", [1.5])
newdata_store0 = dm.DataFrame()
newdata_store0.add_numeric_column("promo", [1.5])
newdata_store0.add_numeric_column("store", [0.0])
print(f"promo=1.5, unseen store:   {store_model.predict(newdata_population)[0]} expected visits (fixed effects only)")
print(f"promo=1.5, store 0 (known): {store_model.predict(newdata_store0)[0]} expected visits "
      "(fixed effects + store 0's BLUP)")
