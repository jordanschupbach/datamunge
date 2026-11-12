# CMD: nix develop .#r --command Rscript examples/r/glmm_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
library(datamunger)


cat("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n")
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
penguins <- DataFrame$penguins()
n <- penguins$nrows()
is_male <- c()
body_mass <- c()
island <- c()
for (i in 0:(n - 1)) {
  if (penguins$is_null("sex", i) || penguins$is_null("body_mass_g", i) ||
      penguins$is_null("island", i)) next
  is_male <- c(is_male, if (penguins$string_at("sex", i) == "male") 1.0 else 0.0)
  body_mass <- c(body_mass, penguins$numeric_at("body_mass_g", i))
  island <- c(island, penguins$string_at("island", i))
}

sex_df <- DataFrame$empty()
sex_df$add_numeric_column("is_male", is_male)
sex_df$add_numeric_column("body_mass_g", body_mass)
sex_df$add_string_column("island", island)

sex_model <- GLMM$new(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")
sex_model$print_summary()

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

df <- DataFrame$empty()
df$add_numeric_column("store", store)
df$add_numeric_column("promo", promo)
df$add_numeric_column("visits", visits)

store_model <- GLMM$new(df, "visits ~ promo + (1 | store)", "poisson")
store_model$print_summary()

cat("\nTrue generating values: intercept=", true_intercept, ", slope=", true_slope,
    ", random-intercept SD (log scale)=0.4\n", sep = "")

cat("\n--- BLUPs for a few stores ---\n")
group_labels <- store_model$group_labels()
for (idx in c(1, 2, 3)) {
  cat("store", group_labels[idx], ": intercept shift=", store_model$random_effects_for_group(idx - 1)[1], "\n")
}

cat("\n--- Prediction: population-level vs. store-adjusted ---\n")
newdata_population <- DataFrame$empty()
newdata_population$add_numeric_column("promo", c(1.5))
newdata_store0 <- DataFrame$empty()
newdata_store0$add_numeric_column("promo", c(1.5))
newdata_store0$add_numeric_column("store", c(0.0))
cat("promo=1.5, unseen store:   ", store_model$predict(newdata_population)[1],
    " expected visits (fixed effects only)\n", sep = "")
cat("promo=1.5, store 0 (known): ", store_model$predict(newdata_store0)[1],
    " expected visits (fixed effects + store 0's BLUP)\n", sep = "")
