1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

printf("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n");
% Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
% confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
% is not tied to location, so this is a well-behaved fit.
penguins = DataFrame_penguins();
is_male = [];
body_mass = [];
island = {};
n = DataFrame_nrows(penguins);
for i = 0:(n - 1)
  if !(DataFrame_is_null(penguins, "sex", i) || DataFrame_is_null(penguins, "body_mass_g", i) || DataFrame_is_null(penguins, "island", i))
    is_male(end + 1) = strcmp(DataFrame_string_at(penguins, "sex", i), "male");
    body_mass(end + 1) = DataFrame_numeric_at(penguins, "body_mass_g", i);
    island{end + 1} = DataFrame_string_at(penguins, "island", i);
  end
end

sex_df = DataFrame_empty();
DataFrame_add_numeric_column(sex_df, "is_male", dv(is_male));
DataFrame_add_numeric_column(sex_df, "body_mass_g", dv(body_mass));
DataFrame_add_string_column(sex_df, "island", sv(island));

sex_model = GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial");
GLMM_print_summary(sex_model);

printf("\n=================== Poisson mixed model on simulated multi-site count data ===================\n");
% 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
% the baseline traffic and the promo's effectiveness vary by store.
rand("seed", 4242);
randn("seed", 4242);
n_stores = 25;
store_effect = randn(1, n_stores) * 0.4;

true_intercept = 2.0;
true_slope = 0.3;
store = [];
promo = [];
visits = [];
for s = 0:(n_stores - 1)
  n_days = 15 + floor(rand() * 11);
  for d = 1:n_days
    promo_intensity = rand() * 3.0;
    lam = exp(true_intercept + store_effect(s + 1) + true_slope * promo_intensity);
    % Knuth's Poisson sampler.
    l_thresh = exp(-lam);
    k = 0;
    p = 1.0;
    while true
      k = k + 1;
      p = p * rand();
      if p <= l_thresh
        break
      end
    end
    store(end + 1) = s;
    promo(end + 1) = promo_intensity;
    visits(end + 1) = k - 1;
  end
end

df = DataFrame_empty();
DataFrame_add_numeric_column(df, "store", dv(store));
DataFrame_add_numeric_column(df, "promo", dv(promo));
DataFrame_add_numeric_column(df, "visits", dv(visits));

store_model = GLMM(df, "visits ~ promo + (1 | store)", "poisson");
GLMM_print_summary(store_model);

printf("\nTrue generating values: intercept=%g, slope=%g, random-intercept SD (log scale)=0.4\n", true_intercept, true_slope);

printf("\n--- BLUPs for a few stores ---\n");
group_labels = GLMM_group_labels(store_model);
for idx = 0:2
  re = GLMM_random_effects_for_group(store_model, idx);
  printf("store %s: intercept shift=%g\n", group_labels{idx + 1}, re{1});
end

printf("\n--- Prediction: population-level vs. store-adjusted ---\n");
newdata_population = DataFrame_empty();
DataFrame_add_numeric_column(newdata_population, "promo", dv([1.5]));
newdata_store0 = DataFrame_empty();
DataFrame_add_numeric_column(newdata_store0, "promo", dv([1.5]));
DataFrame_add_numeric_column(newdata_store0, "store", dv([0.0]));
pred_pop = GLMM_predict(store_model, newdata_population);
pred_s0 = GLMM_predict(store_model, newdata_store0);
printf("promo=1.5, unseen store:   %g expected visits (fixed effects only)\n", pred_pop{1});
printf("promo=1.5, store 0 (known): %g expected visits (fixed effects + store 0's BLUP)\n", pred_s0{1});
