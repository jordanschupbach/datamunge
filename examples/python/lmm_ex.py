import random

from pydatamunge import datamunge as dm

print("=================== Random intercept on a real dataset (penguins) ===================")
penguins = dm.DataFrame.penguins()
species_model = dm.LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
species_model.print_summary()

print("\n=================== Random intercept + slope on a simulated multi-school dataset "
      "===================")
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
rng = random.Random(2024)
n_schools = 30
school_intercept = [rng.normalvariate(0.0, 6.0) for _ in range(n_schools)]  # true random-intercept SD
school_slope = [rng.normalvariate(0.0, 1.2) for _ in range(n_schools)]      # true random-slope SD

true_intercept, true_slope = 60.0, 3.0
school, study_hours, score = [], [], []
for s in range(n_schools):
    n_students = rng.randint(15, 35)
    for _ in range(n_students):
        hours = rng.uniform(0.0, 10.0)
        s_val = true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + rng.normalvariate(0.0, 4.0)
        school.append(float(s))
        study_hours.append(hours)
        score.append(s_val)

df = dm.DataFrame()
df.add_numeric_column("school", school)
df.add_numeric_column("study_hours", study_hours)
df.add_numeric_column("score", score)

model = dm.LMM(df, "score ~ study_hours + (1 + study_hours | school)")
model.print_summary()

print(f"\nTrue generating values: intercept={true_intercept}, slope={true_slope}, "
      "random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

print("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
group_labels = model.group_labels()
for idx in (0, 1, 2):
    re = model.random_effects_for_group(idx)
    print(f"school {group_labels[idx]}: intercept shift={re[0]}, slope shift={re[1]}")

print("\n--- Prediction: population-level vs. school-adjusted ---")
newdata_population = dm.DataFrame()
newdata_population.add_numeric_column("study_hours", [5.0])
newdata_school0 = dm.DataFrame()
newdata_school0.add_numeric_column("study_hours", [5.0])
newdata_school0.add_numeric_column("school", [0.0])
print(f"5 study hours, unseen school:      {model.predict(newdata_population)[0]} (fixed effects only)")
print(f"5 study hours, school 0 (known):    {model.predict(newdata_school0)[0]} (fixed effects + school 0's BLUP)")
