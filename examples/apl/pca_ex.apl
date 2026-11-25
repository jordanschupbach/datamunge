⍝ CMD: just TARGET=pca_ex run-apl
⍝ PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ D FMT X: the number X with D decimals, C style ('-' for '¯', '0.5' for '.5').
∇Z←D FMT X
 Z←D⍕X
 Z←(∨\Z≠' ')/Z
 Z←((Z[1]='.')/'0'),Z
 Z←(1↑Z),(('¯.'≡2↑Z)/'0'),1↓Z
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
IRIS←dm∆data∆iris
FEATURES←'Sepal.Length' 'Sepal.Width' 'Petal.Length' 'Petal.Width'
⍝ species labels, for the grouped (colored-by-species) scores plot
SPECIES←{IRIS dm∆DataFrame∆string_at 'Species' ⍵}¨¯1+⍳dm∆DataFrame∆nrows IRIS

PUT '=================== PCA on iris (scaled) ==================='
PCA←dm∆PCA∆new IRIS FEATURES 1 1
dm∆PCA∆print_summary PCA

PUT NL,'PC1 loadings (which original features drive it):'
NAMES←dm∆PCA∆feature_names PCA
LOADINGS←PCA dm∆PCA∆component_loadings 0
∇NAMES SHOW LOADINGS;I
 I←0
LOOP:→((⍴NAMES)<I←I+1)/0
 PUT '  ',(⊃NAMES[I]),': ',4 FMT LOADINGS[I]
 →LOOP
∇
NAMES SHOW LOADINGS

PUT NL,'Training scores as a DataFrame:'
SCORES←dm∆PCA∆scores_frame PCA
PUT SCORES dm∆DataFrame∆to_string ⍬

SCATTER←PCA dm∆PCA∆plot_scores_grouped ⊂SPECIES
SCATTER dm∆Plot∆save_svg 'pca_iris_scores.svg'
PUT NL,'Scores scatter (colored by species) saved as pca_iris_scores.svg'

SCREE←dm∆PCA∆plot_scree PCA
SCREE dm∆Plot∆save_svg 'pca_iris_scree.svg'
PUT 'Scree plot saved as pca_iris_scree.svg'

PUT NL,'=================== PCA on iris (unscaled) ==================='
UNSCALED←dm∆PCA∆new IRIS FEATURES 1 0
SCALED_RATIO←dm∆PCA∆explained_variance_ratio PCA
UNSCALED_RATIO←dm∆PCA∆explained_variance_ratio UNSCALED
PUT 'PC1 explains ',(2 FMT 100×UNSCALED_RATIO[1]),'% of variance (vs. ',(2 FMT 100×SCALED_RATIO[1]),'% scaled) -- Sepal.Length''s larger raw variance dominates the unscaled covariance matrix.'
dm∆Delete IRIS PCA SCORES SCATTER SCREE UNSCALED
