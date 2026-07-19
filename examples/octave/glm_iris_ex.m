1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

iris = DataFrame_iris();

% Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
% the other two on these predictors, which sends logistic regression's coefficients toward
% +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
% binary split for a demo.
is_virginica = [];
petal_length = [];
petal_width = [];
n = DataFrame_nrows(iris);
for i = 0:(n - 1)
  species = DataFrame_string_at(iris, "Species", i);
  if strcmp(species, "versicolor") || strcmp(species, "virginica")
    is_virginica(end + 1) = strcmp(species, "virginica");
    petal_length(end + 1) = DataFrame_numeric_at(iris, "Petal.Length", i);
    petal_width(end + 1) = DataFrame_numeric_at(iris, "Petal.Width", i);
  end
end

sub = DataFrame_empty();
DataFrame_add_numeric_column(sub, "Petal.Length", dv(petal_length));
DataFrame_add_numeric_column(sub, "Petal.Width", dv(petal_width));
DataFrame_add_numeric_column(sub, "is_virginica", dv(is_virginica));

printf("=================== Logistic regression (binomial, logit link) ===================\n");
logit = GLM(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
GLM_print_summary(logit);

fitted = GLM_fitted_values(logit);
correct = 0;
for i = 1:numel(is_virginica)
  if (fitted{i} >= 0.5) == (is_virginica(i) >= 0.5)
    correct = correct + 1;
  end
end
printf("\nResubstitution accuracy at 0.5 threshold: %g%%\n", 100.0 * correct / numel(is_virginica));

GLM_save_diagnostic_plots(logit, "glm_logistic_iris");
printf("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n");

% Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
% classic sigmoid shape of a fitted logistic regression, with a 95%% confidence band.
width_mean = mean(petal_width);
grid_n = 100;
pl_min = min(petal_length) - 0.3;
pl_max = max(petal_length) + 0.3;
grid_x = pl_min + (pl_max - pl_min) * (0:(grid_n - 1)) / (grid_n - 1);
grid = DataFrame_empty();
DataFrame_add_numeric_column(grid, "Petal.Length", dv(grid_x));
DataFrame_add_numeric_column(grid, "Petal.Width", dv(repmat(width_mean, 1, grid_n)));
curve_frame = GLM_predict_frame(logit, grid, "confidence");
printf("\nPredicted-probability curve (first 5 rows):\n");
printf("%s\n", DataFrame_to_string(curve_frame, 5));

% Poisson regression, for contrast: same IRLS engine, different family/link.
printf("\n=================== Poisson regression (log link) ===================\n");
count = [];
sepal_width = [];
all_petal_length = [];
for i = 0:(n - 1)
  count(end + 1) = round(DataFrame_numeric_at(iris, "Sepal.Length", i));
  sepal_width(end + 1) = DataFrame_numeric_at(iris, "Sepal.Width", i);
  all_petal_length(end + 1) = DataFrame_numeric_at(iris, "Petal.Length", i);
end
count_data = DataFrame_empty();
DataFrame_add_numeric_column(count_data, "Sepal.Width", dv(sepal_width));
DataFrame_add_numeric_column(count_data, "Petal.Length", dv(all_petal_length));
DataFrame_add_numeric_column(count_data, "count", dv(count));

poisson = GLM(count_data, "count ~ Sepal.Width + Petal.Length", "poisson");
GLM_print_summary(poisson);
