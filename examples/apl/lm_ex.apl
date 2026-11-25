⍝ CMD: just TARGET=lm_ex run-apl
⍝ Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
CARS←dm∆DataFrame∆new ⍬
CARS dm∆DataFrame∆add_numeric_column 'hp' (110 110 93 110 175 105 245 62 95 123)
CARS dm∆DataFrame∆add_numeric_column 'wt' (2.62 2.875 2.32 3.215 3.44 3.46 3.57 3.19 3.15 3.44)
CARS dm∆DataFrame∆add_string_column 'transmission' ('manual' 'manual' 'manual' 'automatic' 'automatic' 'automatic' 'automatic' 'automatic' 'automatic' 'automatic')
CARS dm∆DataFrame∆add_numeric_column 'mpg' (21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2)

PUT 'Fitting: mpg ~ hp + wt + transmission',NL
MODEL←dm∆LM∆new CARS 'mpg ~ hp + wt + transmission'
dm∆LM∆print_summary MODEL

PUT NL,'Sequential ANOVA:'
ANOVA←dm∆LM∆anova MODEL
PUT ANOVA dm∆DataFrame∆to_string ⍬

NEWCARS←dm∆DataFrame∆new ⍬
NEWCARS dm∆DataFrame∆add_numeric_column 'hp' (150 90)
NEWCARS dm∆DataFrame∆add_numeric_column 'wt' (3 2.5)
NEWCARS dm∆DataFrame∆add_string_column 'transmission' ('manual' 'automatic')

FRAME←MODEL dm∆LM∆predict_frame NEWCARS 'confidence'
PUT NL,'Predictions with 95% confidence intervals:'
PUT FRAME dm∆DataFrame∆to_string ⍬

MODEL dm∆LM∆save_diagnostic_plots 'lm_ex_diagnostics'
PUT NL,'Saved diagnostic plots as lm_ex_diagnostics_*.svg'
dm∆Delete CARS MODEL ANOVA NEWCARS FRAME
