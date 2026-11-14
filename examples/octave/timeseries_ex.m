% CMD: just TARGET=timeseries_ex run-octave
% Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).

1;

% Restart Octave's generator from a fixed seed.
function seed (n)
  rand ('twister', n);
end

% (0, 1]: avoids log(0) in the Box-Muller transform.
function u = uniform ()
  u = 1 - rand ();
end

function g = gauss ()
  u1 = uniform ();
  u2 = uniform ();
  g = sqrt (-2 * log (u1)) * cos (2 * pi * u2);
end

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  for d = 1:17
    s = sprintf ('%.*g', d, x);
    if str2double (s) == x
      break;
    end
  end
end

function s = joined (v)
  s = strjoin (arrayfun (@num, v, 'UniformOutput', false), ', ');
end

printf ('=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n');
seed (42);
y = zeros (1, 150);
level = 100.0;
prev_shock = 0.0;
for i = 1:150
  shock = gauss ();
  level += 0.3 + 0.4 * prev_shock + shock;  % integrated ARMA(0,1) increments with drift
  y(i) = level;
  prev_shock = shock;
end

options = datamunge.ARIMAOptions ();
options.p = 1;
options.d = 1;
options.q = 1;
options.de_population_size = 80;
options.de_max_generations = 400;
model = datamunge.ARIMA (y, options);

ar = model.ar_coefficients ();
ma = model.ma_coefficients ();
printf ('AR coefficient: %s\n', num (ar(1)));
printf ('MA coefficient: %s\n', num (ma(1)));
printf ('sigma^2: %s, AIC: %s, BIC: %s\n', num (model.sigma2 ()), num (model.aic ()), num (model.bic ()));

intervals = model.forecast_with_intervals (6);
printf ('6-step forecast: %s\n', joined (intervals{1}));
printf ('forecast std. errors: %s\n', joined (intervals{2}));

printf ('\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n');
seed (7);
s = zeros (1, 120);
prev = 0.0;
for i = 0:119
  prev = 0.5 * prev + gauss ();
  s(i + 1) = 20 + 0.2 * i + 5 * sin (2 * pi * i / 12) + prev;
end

options2 = datamunge.ARIMAOptions ();
options2.p = 1;
options2.seasonal_p = 1;
options2.seasonal_d = 1;
options2.seasonal_period = 12;
options2.de_population_size = 100;
options2.de_max_generations = 500;
model2 = datamunge.ARIMA (s, options2);
ar = model2.ar_coefficients ();
sar = model2.seasonal_ar_coefficients ();
printf ('AR coefficient: %s, seasonal AR coefficient: %s\n', num (ar(1)), num (sar(1)));
printf ('12-step forecast: %s\n', joined (model2.forecast (12)));

printf ('\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n');
seasonal_shape = [0.8 0.75 0.9 0.95 1.0 1.05 1.1 1.05 1.0 1.1 1.3 1.6];
seed (11);
y2 = zeros (1, 48);
for i = 0:47
  y2(i + 1) = (100 + 2 * i) * seasonal_shape(mod (i, 12) + 1) + gauss () * 3;
end

es_options = datamunge.ExponentialSmoothingOptions ();
es_options.trend = datamunge.TrendType.Additive;
es_options.seasonal = datamunge.SeasonalType.Multiplicative;
es_options.seasonal_period = 12;
es_options.de_population_size = 60;
es_options.de_max_generations = 300;
es_model = datamunge.ExponentialSmoothing (y2, es_options);
printf ('alpha=%s beta=%s gamma=%s\n', num (es_model.alpha ()), num (es_model.beta ()), num (es_model.gamma ()));
printf ('sigma^2: %s, AIC: %s\n', num (es_model.sigma2 ()), num (es_model.aic ()));
printf ('12-month forecast: %s\n', joined (es_model.forecast (12)));

printf ("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n");
flat = [50.2 49.8 50.5 49.6 50.1 50.3 49.9 50.0 50.4 49.7];

ses = datamunge.ExponentialSmoothing (flat, datamunge.ExponentialSmoothingOptions ());
printf ('SES alpha= %s\n', num (ses.alpha ()));
printf ('SES 5-step forecast: %s\n', joined (ses.forecast (5)));

holt_options = datamunge.ExponentialSmoothingOptions ();
holt_options.trend = datamunge.TrendType.Additive;
holt = datamunge.ExponentialSmoothing (flat, holt_options);
printf ('Holt alpha=%s beta=%s\n', num (holt.alpha ()), num (holt.beta ()));
printf ('Holt 5-step forecast: %s\n', joined (holt.forecast (5)));
