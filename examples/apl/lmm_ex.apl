⍝ CMD: just TARGET=lmm_ex run-apl
⍝ Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: up to 10 significant digits ('-' for '¯', 'e' exponent).
∇Z←G X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
∇
PUT '=================== Random intercept on a real dataset (penguins) ==================='
PENGUINS←dm∆data∆penguins
SPECIES_MODEL←dm∆LMM∆new PENGUINS 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)'
dm∆LMM∆print_summary SPECIES_MODEL

PUT NL,'=================== Random intercept + slope on a simulated multi-school dataset ==================='
⍝ 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
⍝ and the return on an extra hour of study vary by school -- the textbook case for a
⍝ random-intercept-and-slope model. The data come from datamunge's own seeded RNG
⍝ (dm∆set_seed, dm∆rnorm, dm∆runif), so they differ from the other ports' simulated data.
dm∆set_seed 2024
N_SCHOOLS←30
SCHOOL_INTERCEPT←dm∆rnorm 0 6 N_SCHOOLS   ⍝ true random-intercept SD
SCHOOL_SLOPE←dm∆rnorm 0 1.2 N_SCHOOLS     ⍝ true random-slope SD
TRUE_INTERCEPT←60
TRUE_SLOPE←3
∇SIMULATE;S;N;HOURS
 SCHOOL←STUDY_HOURS←SCORE←⍬
 S←0
LOOP:N←15+⌊↑dm∆runif 0 21 1   ⍝ 15..35 students
 HOURS←dm∆runif 0 10 N
 SCHOOL←SCHOOL,N⍴S
 STUDY_HOURS←STUDY_HOURS,HOURS
 SCORE←SCORE,TRUE_INTERCEPT+SCHOOL_INTERCEPT[S+1]+((TRUE_SLOPE+SCHOOL_SLOPE[S+1])×HOURS)+dm∆rnorm 0 4 N
 S←S+1
 →(S<N_SCHOOLS)/LOOP
∇
SIMULATE

DF←dm∆DataFrame∆new ⍬
DF dm∆DataFrame∆add_numeric_column 'school' (1.0×SCHOOL)
DF dm∆DataFrame∆add_numeric_column 'study_hours' STUDY_HOURS
DF dm∆DataFrame∆add_numeric_column 'score' SCORE

MODEL←dm∆LMM∆new DF 'score ~ study_hours + (1 + study_hours | school)'
dm∆LMM∆print_summary MODEL

PUT NL,'True generating values: intercept=',(G TRUE_INTERCEPT),'.0, slope=',(G TRUE_SLOPE),'.0, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0'

PUT NL,'--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---'
GROUP_LABELS←dm∆LMM∆group_labels MODEL
∇BLUPS;IDX;RE
 IDX←0
LOOP:RE←MODEL dm∆LMM∆random_effects_for_group IDX
 PUT 'school ',(⍕⊃GROUP_LABELS[IDX+1]),': intercept shift=',(G RE[1]),', slope shift=',G RE[2]
 IDX←IDX+1
 →(IDX<3)/LOOP
∇
BLUPS

PUT NL,'--- Prediction: population-level vs. school-adjusted ---'
NEWDATA_POPULATION←dm∆DataFrame∆new ⍬
NEWDATA_POPULATION dm∆DataFrame∆add_numeric_column 'study_hours' (,5.0)
NEWDATA_SCHOOL0←dm∆DataFrame∆new ⍬
NEWDATA_SCHOOL0 dm∆DataFrame∆add_numeric_column 'study_hours' (,5.0)
NEWDATA_SCHOOL0 dm∆DataFrame∆add_numeric_column 'school' (,0.0)
PUT '5 study hours, unseen school:      ',(G ↑MODEL dm∆LMM∆predict NEWDATA_POPULATION),' (fixed effects only)'
PUT '5 study hours, school 0 (known):    ',(G ↑MODEL dm∆LMM∆predict NEWDATA_SCHOOL0),' (fixed effects + school 0''s BLUP)'
dm∆Delete PENGUINS SPECIES_MODEL DF MODEL NEWDATA_POPULATION NEWDATA_SCHOOL0
