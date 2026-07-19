# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

cat("=================== Random intercept on a real dataset (penguins) ===================\n")
penguins <- DataFrame_penguins()
species_model <- LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
LMM_print_summary(species_model)

cat("\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n")
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
set.seed(2024)
n_schools <- 30
school_intercept <- rnorm(n_schools, 0.0, 6.0)  # true random-intercept SD
school_slope <- rnorm(n_schools, 0.0, 1.2)      # true random-slope SD

true_intercept <- 60.0
true_slope <- 3.0
school <- c()
study_hours <- c()
score <- c()
for (s in 1:n_schools) {
  n_students <- sample(15:35, 1)
  hours <- runif(n_students, 0.0, 10.0)
  s_val <- true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + rnorm(n_students, 0.0, 4.0)
  school <- c(school, rep(s - 1, n_students))
  study_hours <- c(study_hours, hours)
  score <- c(score, s_val)
}

df <- DataFrame_empty()
DataFrame_add_numeric_column(df, "school", school)
DataFrame_add_numeric_column(df, "study_hours", study_hours)
DataFrame_add_numeric_column(df, "score", score)

model <- LMM(df, "score ~ study_hours + (1 + study_hours | school)")
LMM_print_summary(model)

cat("\nTrue generating values: intercept=", true_intercept, ", slope=", true_slope,
    ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n", sep = "")

cat("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n")
group_labels <- LMM_group_labels(model)
for (idx in c(1, 2, 3)) {
  re <- LMM_random_effects_for_group(model, idx - 1)
  cat("school", group_labels[idx], ": intercept shift=", re[1], ", slope shift=", re[2], "\n")
}

cat("\n--- Prediction: population-level vs. school-adjusted ---\n")
newdata_population <- DataFrame_empty()
DataFrame_add_numeric_column(newdata_population, "study_hours", c(5.0))
newdata_school0 <- DataFrame_empty()
DataFrame_add_numeric_column(newdata_school0, "study_hours", c(5.0))
DataFrame_add_numeric_column(newdata_school0, "school", c(0.0))
cat("5 study hours, unseen school:      ", LMM_predict(model, newdata_population)[1], " (fixed effects only)\n", sep = "")
cat("5 study hours, school 0 (known):    ", LMM_predict(model, newdata_school0)[1],
    " (fixed effects + school 0's BLUP)\n", sep = "")
