% CMD: just TARGET=glmm_ex run-octave
% Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
1;

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  for p = 1:17
    s = sprintf ('%.*g', p, x);
    if (str2double (s) == x)
      return;
    end
  end
end

function g = gauss ()
  u1 = 1 - rand ();  % 1 - u keeps log() away from 0
  u2 = rand ();
  g = sqrt (-2 * log (u1)) * cos (2 * pi * u2);
end

rand ('twister', 4242);

printf ('=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n');
% Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
% confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
% is not tied to location, so this is a well-behaved fit.
penguins = datamunge.Data.penguins ();
is_male = [];
body_mass = [];
island = {};
for i = 0:penguins.nrows () - 1
  if (penguins.is_null ('sex', i) || penguins.is_null ('body_mass_g', i) || penguins.is_null ('island', i))
    continue;
  end
  is_male(end + 1) = strcmp (penguins.string_at ('sex', i), 'male');
  body_mass(end + 1) = penguins.numeric_at ('body_mass_g', i);
  island{end + 1} = penguins.string_at ('island', i);
end
sex_df = datamunge.DataFrame ();
sex_df.add_numeric_column ('is_male', is_male);
sex_df.add_numeric_column ('body_mass_g', body_mass);
sex_df.add_string_column ('island', island);
datamunge.GLMM (sex_df, 'is_male ~ body_mass_g + (1 | island)', 'binomial').print_summary ();

printf ('\n=================== Poisson mixed model on simulated multi-site count data ===================\n');
% 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
% the baseline traffic and the promo's effectiveness vary by store.
n_stores = 25;
store_effect = arrayfun (@(s) gauss () * 0.4, 1:n_stores);
true_intercept = 2.0;
true_slope = 0.3;

store = [];
promo = [];
visits = [];
for s = 0:n_stores - 1
  n_days = 15 + floor (rand () * 11);
  for day = 1:n_days
    promo_intensity = rand () * 3;
    lam = exp (true_intercept + store_effect(s + 1) + true_slope * promo_intensity);
    % Knuth's Poisson sampler.
    threshold = exp (-lam);
    k = 0;
    p = 1;
    do
      k++;
      p *= rand ();
    until (p <= threshold)
    store(end + 1) = s;
    promo(end + 1) = promo_intensity;
    visits(end + 1) = k - 1;
  end
end
df = datamunge.DataFrame ();
df.add_numeric_column ('store', store);
df.add_numeric_column ('promo', promo);
df.add_numeric_column ('visits', visits);
store_model = datamunge.GLMM (df, 'visits ~ promo + (1 | store)', 'poisson');

store_model.print_summary ();

printf ('\nTrue generating values: intercept=%.1f, slope=%.1f, random-intercept SD (log scale)=0.4\n', ...
        true_intercept, true_slope);

printf ('\n--- BLUPs for a few stores ---\n');
labels = store_model.group_labels ();
for idx = 0:2
  re = store_model.random_effects_for_group (idx);
  printf ('store %s: intercept shift=%s\n', labels{idx + 1}, num (re(1)));
end

printf ('\n--- Prediction: population-level vs. store-adjusted ---\n');
population = datamunge.DataFrame ();
store0 = datamunge.DataFrame ();
population.add_numeric_column ('promo', 1.5);
store0.add_numeric_column ('promo', 1.5);
store0.add_numeric_column ('store', 0);
predicted = store_model.predict (population);
printf ('promo=1.5, unseen store:   %s expected visits (fixed effects only)\n', num (predicted(1)));
predicted = store_model.predict (store0);
printf ('promo=1.5, store 0 (known): %s expected visits (fixed effects + store 0''s BLUP)\n', num (predicted(1)));
