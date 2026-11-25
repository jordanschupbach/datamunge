⍝ CMD: just TARGET=hypothesis_tests_ex run-apl
⍝ Classical hypothesis tests on iris: t/Wilcoxon, ANOVA/Kruskal-Wallis, correlation, F, normality,
⍝ chi-squared/Fisher and proportion/binomial tests.
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
⍝ LABEL REPORT R: one line describing the HypothesisTestResult R, which it then releases.
∇LABEL REPORT R;LINE;ALT
 LINE←LABEL,': statistic=',NUM dm∆HypothesisTestResult∆statistic R
 LINE←LINE,(0≠dm∆HypothesisTestResult∆parameter1 R)/', df1=',NUM dm∆HypothesisTestResult∆parameter1 R
 LINE←LINE,(0≠dm∆HypothesisTestResult∆parameter2 R)/', df2=',NUM dm∆HypothesisTestResult∆parameter2 R
 ALT←dm∆HypothesisTestResult∆alternative R
 ⍝ dm∆Alternative is the (name value) table of the enum: TwoSided, Less, Greater.
 ALT←⊃('two.sided' 'less' 'greater')[(∊(dm∆Alternative)[;2])⍳ALT]
 LINE←LINE,', p=',(NUM dm∆HypothesisTestResult∆p_value R),' (',ALT,')'
 →(~dm∆HypothesisTestResult∆has_conf_int R)/DONE
 LINE←LINE,', CI=[',(NUM dm∆HypothesisTestResult∆conf_int_lower R),', ',(NUM dm∆HypothesisTestResult∆conf_int_upper R),']'
DONE:PUT LINE,' -- ',dm∆HypothesisTestResult∆method R
 dm∆Delete R
∇

IRIS←dm∆data∆iris
SPECIES←{IRIS dm∆DataFrame∆string_at 'Species' ⍵}¨¯1+⍳dm∆DataFrame∆nrows IRIS
PETAL←IRIS dm∆DataFrame∆pull_numeric 'Petal.Length'
SEPAL←IRIS dm∆DataFrame∆pull_numeric 'Sepal.Length'
SETOSA_PETAL←(SPECIES∊⊂'setosa')/PETAL
VERSICOLOR_PETAL←(SPECIES∊⊂'versicolor')/PETAL
VIRGINICA_PETAL←(SPECIES∊⊂'virginica')/PETAL

PUT '=================== t-tests: petal length, setosa vs. versicolor ==================='
'Welch two-sample t-test' REPORT dm∆t_test_two_sample SETOSA_PETAL VERSICOLOR_PETAL
'Wilcoxon rank-sum test' REPORT dm∆wilcoxon_rank_sum_test SETOSA_PETAL VERSICOLOR_PETAL

PUT NL,'=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================='
ALL_PETAL←SETOSA_PETAL,VERSICOLOR_PETAL,VIRGINICA_PETAL
SIZES←(≢SETOSA_PETAL),(≢VERSICOLOR_PETAL),≢VIRGINICA_PETAL
'One-way ANOVA' REPORT dm∆one_way_anova ALL_PETAL SIZES
'Kruskal-Wallis' REPORT dm∆kruskal_wallis_test ALL_PETAL SIZES

PUT NL,'=================== Correlation: sepal length vs. petal length ==================='
'Pearson correlation' REPORT dm∆pearson_correlation_test SEPAL PETAL
'Spearman correlation' REPORT dm∆spearman_correlation_test SEPAL PETAL

PUT NL,'=================== F-test: petal length variance, setosa vs. virginica ==================='
'F test' REPORT dm∆f_test_variance SETOSA_PETAL VIRGINICA_PETAL

PUT NL,'=================== Normality: is sepal length normally distributed within setosa? ==================='
SETOSA_SEPAL←(SPECIES∊⊂'setosa')/SEPAL
'Shapiro-Francia' REPORT dm∆shapiro_francia_test SETOSA_SEPAL
'KS vs. fitted normal' REPORT dm∆ks_test_one_sample_normal SETOSA_SEPAL 5.006 0.3525

PUT NL,'=================== Chi-squared / Fisher: is petal length "long" independent of species? ==================='
MEDIAN_ALL←ALL_PETAL[⍋ALL_PETAL][1+⌊(≢ALL_PETAL)÷2]
SETOSA_LONG←+/SETOSA_PETAL>MEDIAN_ALL
SETOSA_SHORT←(≢SETOSA_PETAL)-SETOSA_LONG
VERSICOLOR_LONG←+/VERSICOLOR_PETAL>MEDIAN_ALL
VERSICOLOR_SHORT←(≢VERSICOLOR_PETAL)-VERSICOLOR_LONG
PUT 'table: setosa=[',(⍕SETOSA_LONG),',',(⍕SETOSA_SHORT),'] versicolor=[',(⍕VERSICOLOR_LONG),',',(⍕VERSICOLOR_SHORT),']'
TABLE←SETOSA_LONG,SETOSA_SHORT,VERSICOLOR_LONG,VERSICOLOR_SHORT
'Chi-squared independence' REPORT dm∆chi_squared_test_independence TABLE 2 2
'Fisher''s exact test' REPORT dm∆fisher_exact_test_2x2 SETOSA_LONG SETOSA_SHORT VERSICOLOR_LONG VERSICOLOR_SHORT

PUT NL,'=================== Proportion / binomial: fraction of "long" petals overall ==================='
LONG_COUNT←+/ALL_PETAL>MEDIAN_ALL
'One-sample proportion test (vs 0.5)' REPORT dm∆proportion_test_one_sample LONG_COUNT (≢ALL_PETAL) 0.5
'Exact binomial test (vs 0.5)' REPORT dm∆binomial_test LONG_COUNT (≢ALL_PETAL) 0.5
dm∆Delete IRIS
