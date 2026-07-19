1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

FORMULA = "Petal.Length ~ Petal.Width";

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n", DataFrame_nrows(iris), DataFrame_ncols(iris));
printf("formula: %s\n\n", FORMULA);

% Both the length scale and the noise ratio are auto-selected by maximizing the exact log
% marginal likelihood.
model = GaussianProcessRegression(iris, FORMULA);
GaussianProcessRegression_print_summary(model);

Plot_save(GaussianProcessRegression_plot_fit(model, iris), "gpr_iris_fit.svg");
Plot_save(GaussianProcessRegression_plot_length_scale_profile(model), "gpr_iris_length_scale_profile.svg");
printf("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n");

% Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
% interval at every point -- including far outside the training data, where it should widen
% substantially as the model's uncertainty grows.
query = DataFrame_empty();
DataFrame_add_numeric_column(query, "Petal.Width", dv([0.2, 1.3, 2.5, 10.0]));
detail = GaussianProcessRegression_predict_frame(model, query, "confidence");
printf("\nPredictions with 95%% confidence intervals:\n");
printf("%s\n", DataFrame_to_string(detail));
printf("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n");
