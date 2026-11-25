⍝ CMD: just TARGET=autodiff_ex run-apl
⍝ Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var) and second order (HyperDual).
⍝ Like the other bindings, this uses the bound named-method API: the C++ template drivers
⍝ (derivative, gradient_forward, jacobian_forward, hessian, ...) don't cross the binding boundary.
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
 Z←'[',(⊃{⍺,', ',⍵}/G¨X),']'
∇
⍝ K X keeps the handle X in KEEP (released by FREE) and returns it: every arithmetic method
⍝ returns a new handle, and GNU APL has no finalizers.
KEEP←⍬
∇Z←K X
 KEEP←KEEP,⊂X
 Z←X
∇
∇FREE
 dm∆Delete KEEP
 KEEP←⍬
∇

⍝ The Rosenbrock function (1-x0)² + 100(x1-x0²)² on Duals, HyperDuals and Vars (on TAPE).
∇Z←ROSENBROCK_DUAL B;X0;X1;A;BB
 (X0 X1)←B
 A←K (K dm∆Dual∆new 1 0) dm∆Dual∆subtract X0
 BB←K X1 dm∆Dual∆subtract K X0 dm∆Dual∆multiply X0
 Z←K (K A dm∆Dual∆multiply A) dm∆Dual∆add K (K BB dm∆Dual∆multiply BB) dm∆Dual∆multiply_scalar 100
∇
∇Z←ROSENBROCK_HYPERDUAL B;X0;X1;A;BB
 (X0 X1)←B
 A←K (K dm∆HyperDual∆new 1 0 0 0) dm∆HyperDual∆subtract X0
 BB←K X1 dm∆HyperDual∆subtract K X0 dm∆HyperDual∆multiply X0
 Z←K (K A dm∆HyperDual∆multiply A) dm∆HyperDual∆add K (K BB dm∆HyperDual∆multiply BB) dm∆HyperDual∆multiply_scalar 100
∇
∇Z←ROSENBROCK_VAR B;X0;X1;A;BB
 (X0 X1)←B
 A←K (K dm∆Var∆new TAPE 1) dm∆Var∆subtract X0
 BB←K X1 dm∆Var∆subtract K X0 dm∆Var∆multiply X0
 Z←K (K A dm∆Var∆multiply A) dm∆Var∆add K (K BB dm∆Var∆multiply BB) dm∆Var∆multiply_scalar 100
∇
∇Z←ROSENBROCK_VALUE B;X0;X1
 (X0 X1)←B
 Z←((1-X0)*2)+100×(X1-X0×X0)*2
∇

PUT '=================== Forward mode: scalar derivative ==================='
⍝ Seed derivative = 1 to read df/dx directly.
X0←K dm∆Dual∆new 2 1
F←K (K (K X0 dm∆Dual∆multiply X0) dm∆Dual∆multiply X0) dm∆Dual∆subtract K X0 dm∆Dual∆multiply_scalar 2
PUT 'f(x) = x^3 - 2x, f''(2) = ',(G dm∆Dual∆derivative F),' (exact: 10)'
FREE

PUT NL,'=================== Reverse mode: build a graph by hand ==================='
TAPE←dm∆Tape∆new ⍬
A←K dm∆Var∆new TAPE 2
B←K dm∆Var∆new TAPE 3
Y←K (K A dm∆Var∆multiply B) dm∆Var∆add K dm∆Var∆sin A
PUT 'y = a*b + sin(a) at a=2, b=3 -> y = ',G dm∆Var∆value Y
ADJOINT←TAPE dm∆Tape∆backward Y
PUT 'dy/da = ',(G ADJOINT[1+dm∆Var∆index A]),' (exact: b + cos(a))'
PUT 'dy/db = ',(G ADJOINT[1+dm∆Var∆index B]),' (exact: a)'
FREE
dm∆Delete TAPE

PUT NL,'=================== Forward vs reverse mode agree on the Rosenbrock function ==================='
(PX PY)←0 0
⍝ Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
GX←dm∆Dual∆derivative ROSENBROCK_DUAL (K dm∆Dual∆new PX 1) (K dm∆Dual∆new PY 0)
GY←dm∆Dual∆derivative ROSENBROCK_DUAL (K dm∆Dual∆new PX 0) (K dm∆Dual∆new PY 1)
⍝ Reverse mode: one Tape/Var pass computes every partial at once.
TAPE←dm∆Tape∆new ⍬
VX←K dm∆Var∆new TAPE PX
VY←K dm∆Var∆new TAPE PY
GRAD←TAPE dm∆Tape∆backward ROSENBROCK_VAR VX VY
PUT 'f(0,0) = ',G ROSENBROCK_VALUE PX PY
PUT 'gradient (forward mode): ',LIST GX GY
PUT 'gradient (reverse mode): ',LIST GRAD[1+dm∆Var∆index¨VX VY]
FREE
dm∆Delete TAPE

PUT NL,'=================== Jacobian of a vector-valued function ==================='
⍝ f(x,y) = [x^2, xy, y^3] at (2,3): one Dual pass per input direction fills one column.
∇Z←JACOBIAN_COLUMN SEED;X;Y
 X←K dm∆Dual∆new 2 (SEED[1])
 Y←K dm∆Dual∆new 3 (SEED[2])
 Z←dm∆Dual∆derivative¨(K X dm∆Dual∆multiply X) (K X dm∆Dual∆multiply Y) (K (K Y dm∆Dual∆multiply Y) dm∆Dual∆multiply Y)
∇
JAC←⍉⊃JACOBIAN_COLUMN¨(1 0) (0 1)
PUT 'f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:'
PUT¨(⊂'  '),¨LIST¨⊂[2]JAC
FREE

PUT NL,'=================== Hessian via second-order forward mode (HyperDual) ==================='
⍝ One HyperDual pass per (i, j) pair reads off d²f/dxi dxj directly from eps1eps2; at the
⍝ Rosenbrock function's minimum (1,1).
∇Z←I HESSIAN_ENTRY J;E1;E2
 E1←I=1 2
 E2←J=1 2
 Z←dm∆HyperDual∆eps1eps2 ROSENBROCK_HYPERDUAL (K dm∆HyperDual∆new 1 (E1[1]) (E2[1]) 0) (K dm∆HyperDual∆new 1 (E1[2]) (E2[2]) 0)
∇
H←(⍳2)∘.HESSIAN_ENTRY⍳2
PUT 'Hessian of the Rosenbrock function at its minimum (1,1):'
PUT¨(⊂'  '),¨LIST¨⊂[2]H
FREE

PUT NL,'=================== Gradient descent driven by reverse-mode gradients ==================='
POINT←¯1.2 1   ⍝ the classic Rosenbrock starting point
RATE←0.001
NSTEPS←2000
∇DESCEND;STEP;VX;VY;VF;GRAD;LOSS
 STEP←0
LOOP:TAPE←dm∆Tape∆new ⍬
 VX←K dm∆Var∆new TAPE (POINT[1])
 VY←K dm∆Var∆new TAPE (POINT[2])
 VF←ROSENBROCK_VAR VX VY
 GRAD←TAPE dm∆Tape∆backward VF
 LOSS←dm∆Var∆value VF
 POINT←POINT-RATE×GRAD[1+dm∆Var∆index¨VX VY]
 FREE
 dm∆Delete TAPE
 →(~STEP∊0,NSTEPS-1)/NEXT
 PUT 'step ',(⍕STEP),': loss = ',(G LOSS),', x = ',LIST POINT
NEXT:STEP←STEP+1
 →(STEP<NSTEPS)/LOOP
∇
DESCEND
PUT '(true minimum is at [1, 1] with loss 0)'
