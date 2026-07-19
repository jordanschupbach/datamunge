# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

cat("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n")
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
penguins <- DataFrame_penguins()
n <- DataFrame_nrows(penguins)
is_male <- c()
body_mass <- c()
island <- c()
for (i in 0:(n - 1)) {
  if (DataFrame_is_null(penguins, "sex", i) || DataFrame_is_null(penguins, "body_mass_g", i) ||
      DataFrame_is_null(penguins, "island", i)) next
  is_male <- c(is_male, if (DataFrame_string_at(penguins, "sex", i) == "male") 1.0 else 0.0)
  body_mass <- c(body_mass, DataFrame_numeric_at(penguins, "body_mass_g", i))
  island <- c(island, DataFrame_string_at(penguins, "island", i))
}

sex_df <- DataFrame_empty()
DataFrame_add_numeric_column(sex_df, "is_male", is_male)
DataFrame_add_numeric_column(sex_df, "body_mass_g", body_mass)
DataFrame_add_string_column_encoded(sex_df, "island", encode_strings(island))

sex_model <- GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")
GLMM_print_summary(sex_model)

cat("\n=================== Poisson mixed model on simulated multi-site count data ===================\n")
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
set.seed(4242)
n_stores <- 25
store_effect <- rnorm(n_stores, 0.0, 0.4)  # true random-intercept SD (log scale)

true_intercept <- 2.0
true_slope <- 0.3
store <- c()
promo <- c()
visits <- c()
for (s in 1:n_stores) {
  n_days <- sample(15:25, 1)
  promo_intensity <- runif(n_days, 0.0, 3.0)
  lam <- exp(true_intercept + store_effect[s] + true_slope * promo_intensity)
  k <- rpois(n_days, lam)
  store <- c(store, rep(s - 1, n_days))
  promo <- c(promo, promo_intensity)
  visits <- c(visits, as.numeric(k))
}

df <- DataFrame_empty()
DataFrame_add_numeric_column(df, "store", store)
DataFrame_add_numeric_column(df, "promo", promo)
DataFrame_add_numeric_column(df, "visits", visits)

store_model <- GLMM(df, "visits ~ promo + (1 | store)", "poisson")
GLMM_print_summary(store_model)

cat("\nTrue generating values: intercept=", true_intercept, ", slope=", true_slope,
    ", random-intercept SD (log scale)=0.4\n", sep = "")

cat("\n--- BLUPs for a few stores ---\n")
group_labels <- GLMM_group_labels(store_model)
for (idx in c(1, 2, 3)) {
  cat("store", group_labels[idx], ": intercept shift=", GLMM_random_effects_for_group(store_model, idx - 1)[1], "\n")
}

cat("\n--- Prediction: population-level vs. store-adjusted ---\n")
newdata_population <- DataFrame_empty()
DataFrame_add_numeric_column(newdata_population, "promo", c(1.5))
newdata_store0 <- DataFrame_empty()
DataFrame_add_numeric_column(newdata_store0, "promo", c(1.5))
DataFrame_add_numeric_column(newdata_store0, "store", c(0.0))
cat("promo=1.5, unseen store:   ", GLMM_predict(store_model, newdata_population)[1],
    " expected visits (fixed effects only)\n", sep = "")
cat("promo=1.5, store 0 (known): ", GLMM_predict(store_model, newdata_store0)[1],
    " expected visits (fixed effects + store 0's BLUP)\n", sep = "")
