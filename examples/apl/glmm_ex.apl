⍝ CMD: just TARGET=glmm_ex run-apl
⍝ Generalized linear mixed models: binomial (penguin sex by island) and Poisson (simulated store visits).
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
PUT '=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================='
⍝ Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
⍝ confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
⍝ is not tied to location, so this is a well-behaved fit.
PENGUINS←dm∆data∆penguins
KEEP←(PENGUINS dm∆DataFrame∆pull_string_valid 'sex')∧(PENGUINS dm∆DataFrame∆pull_numeric_valid 'body_mass_g')∧PENGUINS dm∆DataFrame∆pull_string_valid 'island'
SEX_DF←dm∆DataFrame∆new ⍬
SEX_DF dm∆DataFrame∆add_numeric_column 'is_male' (1.0×(KEEP/PENGUINS dm∆DataFrame∆pull_string 'sex')∊⊂'male')
SEX_DF dm∆DataFrame∆add_numeric_column 'body_mass_g' (KEEP/PENGUINS dm∆DataFrame∆pull_numeric 'body_mass_g')
SEX_DF dm∆DataFrame∆add_string_column 'island' (KEEP/PENGUINS dm∆DataFrame∆pull_string 'island')

SEX_MODEL←dm∆GLMM∆new SEX_DF 'is_male ~ body_mass_g + (1 | island)' 'binomial'
dm∆GLMM∆print_summary SEX_MODEL

PUT NL,'=================== Poisson mixed model on simulated multi-site count data ==================='
⍝ 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
⍝ the baseline traffic and the promo's effectiveness vary by store. The data come from
⍝ datamunge's own seeded RNG (dm∆set_seed, dm∆rnorm, dm∆runif), so they differ from the
⍝ other ports' simulated data.
dm∆set_seed 4242
N_STORES←25
STORE_EFFECT←dm∆rnorm 0 0.4 N_STORES   ⍝ true random-intercept SD (log scale)
TRUE_INTERCEPT←2
TRUE_SLOPE←0.3
⍝ POISSON LAM: one Poisson(LAM) draw by Knuth's sampler (count the running products of
⍝ uniforms that stay above exp(-LAM)).
∇Z←POISSON LAM;L;P;PRODS
 L←*-LAM
 Z←0
 P←1
MORE:PRODS←P××\dm∆runif 0 1 64
 Z←Z++/PRODS>L
 P←¯1↑PRODS
 →(P>L)/MORE
∇
∇SIMULATE;S;N;D;X;LAM
 STORE←PROMO←VISITS←⍬
 S←0
LOOP:N←15+⌊↑dm∆runif 0 11 1   ⍝ 15..25 days
 D←0
DAY:X←↑dm∆runif 0 3 1
 LAM←*TRUE_INTERCEPT+STORE_EFFECT[S+1]+TRUE_SLOPE×X
 STORE←STORE,S
 PROMO←PROMO,X
 VISITS←VISITS,POISSON LAM
 D←D+1
 →(D<N)/DAY
 S←S+1
 →(S<N_STORES)/LOOP
∇
SIMULATE

DF←dm∆DataFrame∆new ⍬
DF dm∆DataFrame∆add_numeric_column 'store' (1.0×STORE)
DF dm∆DataFrame∆add_numeric_column 'promo' PROMO
DF dm∆DataFrame∆add_numeric_column 'visits' (1.0×VISITS)

STORE_MODEL←dm∆GLMM∆new DF 'visits ~ promo + (1 | store)' 'poisson'
dm∆GLMM∆print_summary STORE_MODEL

PUT NL,'True generating values: intercept=',(G TRUE_INTERCEPT),'.0, slope=',(G TRUE_SLOPE),', random-intercept SD (log scale)=0.4'

PUT NL,'--- BLUPs for a few stores ---'
GROUP_LABELS←dm∆GLMM∆group_labels STORE_MODEL
∇BLUPS;IDX
 IDX←0
LOOP:PUT 'store ',(⍕⊃GROUP_LABELS[IDX+1]),': intercept shift=',G ↑STORE_MODEL dm∆GLMM∆random_effects_for_group IDX
 IDX←IDX+1
 →(IDX<3)/LOOP
∇
BLUPS

PUT NL,'--- Prediction: population-level vs. store-adjusted ---'
NEWDATA_POPULATION←dm∆DataFrame∆new ⍬
NEWDATA_POPULATION dm∆DataFrame∆add_numeric_column 'promo' (,1.5)
NEWDATA_STORE0←dm∆DataFrame∆new ⍬
NEWDATA_STORE0 dm∆DataFrame∆add_numeric_column 'promo' (,1.5)
NEWDATA_STORE0 dm∆DataFrame∆add_numeric_column 'store' (,0.0)
PUT 'promo=1.5, unseen store:   ',(G ↑STORE_MODEL dm∆GLMM∆predict NEWDATA_POPULATION),' expected visits (fixed effects only)'
PUT 'promo=1.5, store 0 (known): ',(G ↑STORE_MODEL dm∆GLMM∆predict NEWDATA_STORE0),' expected visits (fixed effects + store 0''s BLUP)'
dm∆Delete PENGUINS SEX_DF SEX_MODEL DF STORE_MODEL NEWDATA_POPULATION NEWDATA_STORE0
