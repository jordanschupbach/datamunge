% CMD: just TARGET=gaussian_process_regression_iris_ex run-octave
% Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.

formula = 'Petal.Length ~ Petal.Width';

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n', iris.nrows (), iris.ncols ());
printf ('formula: %s\n\n', formula);

% Both the length scale and the noise ratio are auto-selected by maximizing the exact log
% marginal likelihood.
model = datamunge.GaussianProcessRegression (iris, formula);
model.print_summary ();

model.plot_fit (iris).save ('gpr_iris_fit.svg');
model.plot_length_scale_profile ().save ('gpr_iris_length_scale_profile.svg');
printf ('\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n');

% Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
% interval at every point -- including far outside the training data, where it should widen
% substantially as the model's uncertainty grows.
query = datamunge.DataFrame ();
query.add_numeric_column ('Petal.Width', [0.2 1.3 2.5 10.0]);
detail = model.predict_frame (query, 'confidence');
printf ('\nPredictions with 95%% confidence intervals:\n');
printf ('%s\n', detail.to_string ());
printf ('(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n');
