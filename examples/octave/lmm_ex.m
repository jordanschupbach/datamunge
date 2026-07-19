1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

printf("=================== Random intercept on a real dataset (penguins) ===================\n");
penguins = DataFrame_penguins();
species_model = LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
LMM_print_summary(species_model);

printf("\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n");
% 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
% and the return on an extra hour of study vary by school -- the textbook case for a
% random-intercept-and-slope model.
rand("seed", 2024);
randn("seed", 2024);
n_schools = 30;
school_intercept = randn(1, n_schools) * 6.0;
school_slope = randn(1, n_schools) * 1.2;

true_intercept = 60.0;
true_slope = 3.0;
school = [];
study_hours = [];
score = [];
for s = 0:(n_schools - 1)
  n_students = 15 + floor(rand() * 21);
  for j = 1:n_students
    hours = rand() * 10.0;
    noise = randn() * 4.0;
    s_val = true_intercept + school_intercept(s + 1) + (true_slope + school_slope(s + 1)) * hours + noise;
    school(end + 1) = s;
    study_hours(end + 1) = hours;
    score(end + 1) = s_val;
  end
end

df = DataFrame_empty();
DataFrame_add_numeric_column(df, "school", dv(school));
DataFrame_add_numeric_column(df, "study_hours", dv(study_hours));
DataFrame_add_numeric_column(df, "score", dv(score));

model = LMM(df, "score ~ study_hours + (1 + study_hours | school)");
LMM_print_summary(model);

printf("\nTrue generating values: intercept=%g, slope=%g, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n", true_intercept, true_slope);

printf("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n");
group_labels = LMM_group_labels(model);
for idx = 0:2
  re = LMM_random_effects_for_group(model, idx);
  printf("school %s: intercept shift=%g, slope shift=%g\n", group_labels{idx + 1}, re{1}, re{2});
end

printf("\n--- Prediction: population-level vs. school-adjusted ---\n");
newdata_population = DataFrame_empty();
DataFrame_add_numeric_column(newdata_population, "study_hours", dv([5.0]));
newdata_school0 = DataFrame_empty();
DataFrame_add_numeric_column(newdata_school0, "study_hours", dv([5.0]));
DataFrame_add_numeric_column(newdata_school0, "school", dv([0.0]));
pred_pop = LMM_predict(model, newdata_population);
pred_s0 = LMM_predict(model, newdata_school0);
printf("5 study hours, unseen school:      %g (fixed effects only)\n", pred_pop{1});
printf("5 study hours, school 0 (known):    %g (fixed effects + school 0's BLUP)\n", pred_s0{1});
