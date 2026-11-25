⍝ CMD: just TARGET=naive_bayes_penguins_ex run-apl
⍝ Naive Bayes classifier of penguin species from numeric and categorical predictors.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ R X: the number X like Python's repr of a float ('-' for '¯', '46.0' for 46).
∇Z←R X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
 Z←Z,(~∨/Z∊'.e')/'.0'
∇
⍝ Note: like the Python version, this doesn't call drop_nulls() first -- NaiveBayesClassifier
⍝ drops incomplete rows for its own formula columns internally when fitting, matching every
⍝ other formula-based model in this library.
PENGUINS←dm∆data∆penguins
PUT 'penguins: ',(⍕dm∆DataFrame∆nrows PENGUINS),' rows x ',(⍕dm∆DataFrame∆ncols PENGUINS),' cols',NL

⍝ A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
⍝ predictors in one formula -- each modeled independently given the class, per the naive
⍝ Bayes assumption.
MODEL←dm∆NaiveBayesClassifier∆new PENGUINS 'species ~ bill_length_mm + bill_depth_mm + island + sex'
dm∆NaiveBayesClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆NaiveBayesClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Misclassified rows:'
PREDICTIONS←MODEL dm∆NaiveBayesClassifier∆predict PENGUINS
SPECIES←PENGUINS dm∆DataFrame∆pull_string 'species'
ISLAND←PENGUINS dm∆DataFrame∆pull_string 'island'
SEX←PENGUINS dm∆DataFrame∆pull_string 'sex'
BILL_LENGTH←PENGUINS dm∆DataFrame∆pull_numeric 'bill_length_mm'
BILL_DEPTH←PENGUINS dm∆DataFrame∆pull_numeric 'bill_depth_mm'
COMPLETE←∧⌿⊃((⊂PENGUINS) dm∆DataFrame∆pull_string_valid¨'species' 'island' 'sex'),(⊂PENGUINS) dm∆DataFrame∆pull_numeric_valid¨'bill_length_mm' 'bill_depth_mm'
WRONG←COMPLETE∧~PREDICTIONS≡¨SPECIES
∇SHOWWRONG;I
 I←0
LOOP:→(I≥⍴WRONG)/0
 →(~WRONG[I+1])/NEXT
 PUT '  row ',(⍕I),': bill_length=',(R BILL_LENGTH[I+1]),' bill_depth=',(R BILL_DEPTH[I+1]),' island=',(⊃ISLAND[I+1]),' sex=',(⊃SEX[I+1]),'  actual=',(⊃SPECIES[I+1]),'  predicted=',⊃PREDICTIONS[I+1]
NEXT:I←I+1
 →LOOP
∇
SHOWWRONG
PUT (⍕+/WRONG),' misclassified (of ',(⍕dm∆DataFrame∆nrows PENGUINS),' rows, some incomplete)'

⍝ plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
⍝ two-predictor model (bill measurements alone) just for visualization.
BILL_ONLY←dm∆NaiveBayesClassifier∆new PENGUINS 'species ~ bill_length_mm + bill_depth_mm'
PUT NL,'bill-measurements-only model training accuracy: ',(R 100×dm∆NaiveBayesClassifier∆training_accuracy BILL_ONLY),'%'
PLOT←BILL_ONLY dm∆NaiveBayesClassifier∆plot_classification PENGUINS 'bill_length_mm' 'bill_depth_mm'
PLOT dm∆Plot∆save 'naive_bayes_penguins_classification.svg'
REGIONS←BILL_ONLY dm∆NaiveBayesClassifier∆plot_decision_regions 'bill_length_mm' 'bill_depth_mm'
REGIONS dm∆Plot∆save 'naive_bayes_penguins_decision_regions.svg'
PUT 'Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg'
dm∆Delete PENGUINS MODEL CM BILL_ONLY PLOT REGIONS
