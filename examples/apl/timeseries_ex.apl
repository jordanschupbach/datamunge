⍝ CMD: just TARGET=timeseries_ex run-apl
⍝ Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
⍝ The simulated noise comes from the library's seeded generator (dm∆set_seed, dm∆rnorm), so the series
⍝ (and the fitted numbers) differ from the Python example's, which uses Python's random module.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ NUM X: X with up to 10 significant digits ('-' for '¯', 'e' exponent).
∇Z←NUM X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
∇
⍝ LIST V: a Python-style list "[a, b, c]".
∇Z←LIST V
 Z←'[',(¯2↓∊(NUM¨V),¨⊂', '),']'
∇

PUT '=================== ARIMA(1,1,1) on a simulated random walk with drift ==================='
dm∆set_seed 42
SHOCKS←dm∆rnorm 0 1 150
⍝ integrated ARMA(0,1) increments with drift
Y←100++\0.3+(0.4×0,¯1↓SHOCKS)+SHOCKS

OPTIONS←dm∆ARIMAOptions∆new ⍬
OPTIONS dm∆ARIMAOptions∆p∆set 1
OPTIONS dm∆ARIMAOptions∆d∆set 1
OPTIONS dm∆ARIMAOptions∆q∆set 1
OPTIONS dm∆ARIMAOptions∆de_population_size∆set 80
OPTIONS dm∆ARIMAOptions∆de_max_generations∆set 400
MODEL←dm∆ARIMA∆new Y OPTIONS

PUT 'AR coefficient: ',NUM ↑dm∆ARIMA∆ar_coefficients MODEL
PUT 'MA coefficient: ',NUM ↑dm∆ARIMA∆ma_coefficients MODEL
PUT 'sigma^2: ',(NUM dm∆ARIMA∆sigma2 MODEL),', AIC: ',(NUM dm∆ARIMA∆aic MODEL),', BIC: ',NUM dm∆ARIMA∆bic MODEL

⍝ std::pair<vector, vector> is a 2-vector of vectors.
(POINT SE)←MODEL dm∆ARIMA∆forecast_with_intervals 6
PUT '6-step forecast: ',LIST POINT
PUT 'forecast std. errors: ',LIST SE
dm∆Delete OPTIONS MODEL

PUT NL,'=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================='
dm∆set_seed 7
E←dm∆rnorm 0 1 120
I←¯1+⍳120
⍝ AR(1) noise prev[i] = 0.5×prev[i-1] + e[i], as a weighted sum of the shocks so far.
PREV←((0.5*0⌈I∘.-I)×I∘.≥I)+.×E
S←20+(0.2×I)+(5×1○(○2)×I÷12)+PREV

OPTIONS←dm∆ARIMAOptions∆new ⍬
OPTIONS dm∆ARIMAOptions∆p∆set 1
OPTIONS dm∆ARIMAOptions∆seasonal_p∆set 1
OPTIONS dm∆ARIMAOptions∆seasonal_d∆set 1
OPTIONS dm∆ARIMAOptions∆seasonal_period∆set 12
OPTIONS dm∆ARIMAOptions∆de_population_size∆set 100
OPTIONS dm∆ARIMAOptions∆de_max_generations∆set 500
MODEL←dm∆ARIMA∆new S OPTIONS
PUT 'AR coefficient: ',(NUM ↑dm∆ARIMA∆ar_coefficients MODEL),' , seasonal AR coefficient: ',NUM ↑dm∆ARIMA∆seasonal_ar_coefficients MODEL
PUT '12-step forecast: ',LIST MODEL dm∆ARIMA∆forecast 12
dm∆Delete OPTIONS MODEL

PUT NL,'=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================='
SEASONAL_SHAPE←0.8 0.75 0.9 0.95 1 1.05 1.1 1.05 1 1.1 1.3 1.6
dm∆set_seed 11
I←¯1+⍳48
Y2←((100+2×I)×SEASONAL_SHAPE[1+12|I])+3×dm∆rnorm 0 1 48

OPTIONS←dm∆ExponentialSmoothingOptions∆new ⍬
OPTIONS dm∆ExponentialSmoothingOptions∆trend∆set dm∆TrendType∆Additive
OPTIONS dm∆ExponentialSmoothingOptions∆seasonal∆set dm∆SeasonalType∆Multiplicative
OPTIONS dm∆ExponentialSmoothingOptions∆seasonal_period∆set 12
OPTIONS dm∆ExponentialSmoothingOptions∆de_population_size∆set 60
OPTIONS dm∆ExponentialSmoothingOptions∆de_max_generations∆set 300
MODEL←dm∆ExponentialSmoothing∆new Y2 OPTIONS
PUT 'alpha=',(NUM dm∆ExponentialSmoothing∆alpha MODEL),' beta=',(NUM dm∆ExponentialSmoothing∆beta MODEL),' gamma=',NUM dm∆ExponentialSmoothing∆gamma MODEL
PUT 'sigma^2: ',(NUM dm∆ExponentialSmoothing∆sigma2 MODEL),', AIC: ',NUM dm∆ExponentialSmoothing∆aic MODEL
PUT '12-month forecast: ',LIST MODEL dm∆ExponentialSmoothing∆forecast 12
dm∆Delete OPTIONS MODEL

PUT NL,'=================== Simple exponential smoothing vs. Holt''s linear trend ==================='
FLAT←50.2 49.8 50.5 49.6 50.1 50.3 49.9 50 50.4 49.7

SES_OPTIONS←dm∆ExponentialSmoothingOptions∆new ⍬
SES←dm∆ExponentialSmoothing∆new FLAT SES_OPTIONS
PUT 'SES alpha= ',NUM dm∆ExponentialSmoothing∆alpha SES
PUT 'SES 5-step forecast: ',LIST SES dm∆ExponentialSmoothing∆forecast 5

HOLT_OPTIONS←dm∆ExponentialSmoothingOptions∆new ⍬
HOLT_OPTIONS dm∆ExponentialSmoothingOptions∆trend∆set dm∆TrendType∆Additive
HOLT←dm∆ExponentialSmoothing∆new FLAT HOLT_OPTIONS
PUT 'Holt alpha=',(NUM dm∆ExponentialSmoothing∆alpha HOLT),' beta=',NUM dm∆ExponentialSmoothing∆beta HOLT
PUT 'Holt 5-step forecast: ',LIST HOLT dm∆ExponentialSmoothing∆forecast 5
dm∆Delete SES_OPTIONS SES HOLT_OPTIONS HOLT
