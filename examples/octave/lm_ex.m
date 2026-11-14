% CMD: just TARGET=lm_ex run-octave
% Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.

cars = datamunge.DataFrame ();
cars.add_numeric_column ('hp', [110 110 93 110 175 105 245 62 95 123]);
cars.add_numeric_column ('wt', [2.62 2.875 2.32 3.215 3.44 3.46 3.57 3.19 3.15 3.44]);
cars.add_string_column ('transmission', {'manual', 'manual', 'manual', 'automatic', 'automatic', ...
                                         'automatic', 'automatic', 'automatic', 'automatic', 'automatic'});
cars.add_numeric_column ('mpg', [21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2]);

printf ('Fitting: mpg ~ hp + wt + transmission\n\n');
model = datamunge.LM (cars, 'mpg ~ hp + wt + transmission');
model.print_summary ();

printf ('\nSequential ANOVA:\n%s\n', model.anova ().to_string ());

newcars = datamunge.DataFrame ();
newcars.add_numeric_column ('hp', [150 90]);
newcars.add_numeric_column ('wt', [3.0 2.5]);
newcars.add_string_column ('transmission', {'manual', 'automatic'});
printf ('\nPredictions with 95%% confidence intervals:\n%s\n', ...
        model.predict_frame (newcars, 'confidence').to_string ());

model.save_diagnostic_plots ('lm_ex_diagnostics');
printf ('\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n');
