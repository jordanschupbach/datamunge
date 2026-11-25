⍝ CMD: just TARGET=optim_advanced_ex run-apl
⍝ Second-order and structured optimizers driven by APL objectives: Newton, TrustRegionNewton,
⍝ LevenbergMarquardt and BayesianOptimization (APL implementations of the C++ function interfaces).
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: the number X with up to 15 significant digits ('-' for '¯', 'e' exponent).
∇Z←G X;⎕PP
 ⎕PP←15
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
∇
⍝ The list X as [a, b].
∇Z←LIST X
 Z←'[',(⊃{⍺,', ',⍵}/G¨,X),']'
∇

⍝ ---- APL implementations of the director-based function interfaces (dm∆X∆subclass): the
⍝ ---- optimizer calls Impl∆evaluate/gradient/hessian/residuals/jacobian from C++.

⍝ QuadraticBowl, a HessianFunction: f(x) = (x - TARGET)ᵀ A (x - TARGET), with its exact
⍝ (constant) Hessian -- the natural example for Newton's method. A is SPD, so the bowl is
⍝ well-conditioned and Newton converges in one step.
TARGET←3 ¯1.5
A←2 2⍴4 0.5 0.5 2
∇Z←H Bowl∆evaluate X;D
 D←X-TARGET
 Z←D+.×A+.×D
∇
∇Z←H Bowl∆gradient X
 Z←2×A+.×X-TARGET
∇
⍝ A std::vector<std::vector<double>> result is a vector of rows.
∇Z←H Bowl∆hessian X
 Z←⊂[2]2×A
∇

⍝ ExpDecay, a ResidualFunction: residuals of A·exp(-k·t) + c against TS/YS -- the classic
⍝ nonlinear least-squares curve fit, and the textbook use case for LevenbergMarquardt.
∇Z←H ExpDecay∆residuals P;AA;KK;CC
 (AA KK CC)←P
 Z←(AA×*-KK×TS)+CC-YS
∇
∇Z←H ExpDecay∆jacobian P;AA;KK;CC;E
 (AA KK CC)←P
 E←*-KK×TS
 ⍝ One row per residual: d/dA, d/dk, d/dc.
 Z←⊂[1]⊃E (-AA×TS×E) ((⍴TS)⍴1)
∇

⍝ Shifted1DBowl, an ArbitraryFunction: a 1-D bowl with its minimum near x=1.7 and no gradient
⍝ -- exactly the black-box setting BayesianOptimization targets.
∇Z←H Shifted∆evaluate X
 Z←((X[1]-1.7)*2)+0.1×1○10×X[1]
∇

PUT '=================== HessianFunction: Newton''s method on a quadratic bowl ==================='
F←'Bowl' dm∆HessianFunction∆subclass ⍬
OPTIONS←dm∆NewtonOptions∆new ⍬
OPTIONS dm∆NewtonOptions∆max_iterations∆set 50
OPTIONS dm∆NewtonOptions∆tolerance∆set 1E¯10
OPT←dm∆Newton∆new OPTIONS
⍝ optimize returns the objective value, then the updated coordinates.
(VALUE X)←OPT dm∆Newton∆optimize F (0 0)
PUT 'Newton:            f=',(G VALUE),' x=',(LIST X),' (true minimum: f=0 at ',(LIST TARGET),')'
dm∆Delete OPTIONS OPT

PUT NL,'=================== HessianFunction: TrustRegionNewton on the same bowl ==================='
OPTIONS←dm∆TrustRegionNewtonOptions∆new ⍬
OPTIONS dm∆TrustRegionNewtonOptions∆max_iterations∆set 100
OPTIONS dm∆TrustRegionNewtonOptions∆tolerance∆set 1E¯10
OPT←dm∆TrustRegionNewton∆new OPTIONS
(VALUE X)←OPT dm∆TrustRegionNewton∆optimize F (5 5)
PUT 'TrustRegionNewton: f=',(G VALUE),' x=',(LIST X),' (true minimum: f=0 at ',(LIST TARGET),')'
dm∆Delete OPTIONS OPT F

PUT NL,'=================== ResidualFunction: LevenbergMarquardt curve fit ==================='
TRUE←5 0.7 1
TS←0.5×¯1+⍳9
YS←(TRUE[1]×*-TRUE[2]×TS)+TRUE[3]
F←'ExpDecay' dm∆ResidualFunction∆subclass ⍬
OPTIONS←dm∆LevenbergMarquardtOptions∆new ⍬
OPTIONS dm∆LevenbergMarquardtOptions∆max_iterations∆set 200
OPTIONS dm∆LevenbergMarquardtOptions∆tolerance∆set 1E¯12
OPT←dm∆LevenbergMarquardt∆new OPTIONS
(VALUE P)←OPT dm∆LevenbergMarquardt∆optimize F (1 0.1 0)
PUT 'LevenbergMarquardt: sum-sq-residual=',(G VALUE),' params(A,k,c)=',LIST P
PUT '                    true params(A,k,c)=',LIST TRUE
dm∆Delete OPTIONS OPT F

PUT NL,'=================== BayesianOptimization with RBFGaussianProcessSurrogate ==================='
F←'Shifted' dm∆ArbitraryFunction∆subclass ⍬
SURROGATE←dm∆RBFGaussianProcessSurrogate∆new 1 1E¯6
OPTIONS←dm∆BayesianOptimizationOptions∆new ⍬
OPTIONS dm∆BayesianOptimizationOptions∆initial_samples∆set 10
OPTIONS dm∆BayesianOptimizationOptions∆max_iterations∆set 60
OPTIONS dm∆BayesianOptimizationOptions∆seed∆set 42
OPT←dm∆BayesianOptimization∆new OPTIONS
(VALUE X)←OPT dm∆BayesianOptimization∆optimize F (,0) (,¯3) (,5) SURROGATE
PUT 'BayesianOptimization: f=',(G VALUE),' x=',(LIST X),' (target minimum near x=1.7)'
dm∆Delete F SURROGATE OPTIONS OPT
