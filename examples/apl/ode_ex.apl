⍝ CMD: just TARGET=ode_ex run-apl
⍝ ODEs: a live APL right-hand side, the built-in named systems, a harmonic-oscillator energy check and a Lorenz phase plot.
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
⍝ G X: up to 6 significant digits, roughly C's %g ('-' for '¯', 'e' exponent).
∇Z←G X;⎕PP
 ⎕PP←6
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
∇
⍝ N LJ S: S left-justified to N columns.
∇Z←N LJ S
 Z←N↑S
∇
⍝ The list X as [a, b, c].
∇Z←LIST X
 Z←'[',(⊃{⍺,', ',⍵}/G¨X),']'
∇
⍝ Every state of a solution as a matrix: one row per step.
∇Z←STATES SOL
 Z←⊃(⊂SOL) dm∆ODESolution∆state_at¨¯1+⍳dm∆ODESolution∆size SOL
∇
∇Z←TIMES SOL
 Z←(⊂SOL) dm∆ODESolution∆time_at¨¯1+⍳dm∆ODESolution∆size SOL
∇
∇Z←LAST SOL
 Z←SOL dm∆ODESolution∆state_at ¯1+dm∆ODESolution∆size SOL
∇

⍝ A live, user-supplied RHS: an APL implementation of the abstract C++ RHS (dm∆RHS∆subclass calls
⍝ Decay∆evaluate for its virtual evaluate). dy/dt = -K×y.
K←1
∇Z←H Decay∆evaluate B;T;Y
 (T Y)←B
 Z←,-K×Y
∇

⍝ Solves dy/dt = -y with one method and prints its row.
∇RUN B;NAME;METHOD;ORDER;STEP;TOL;OPTIONS;SOLVER;SOL;Y
 (NAME METHOD ORDER STEP TOL)←B
 OPTIONS←dm∆ODEOptions∆new ⍬
 OPTIONS dm∆ODEOptions∆method∆set METHOD
 OPTIONS dm∆ODEOptions∆multistep_order∆set ORDER
 OPTIONS dm∆ODEOptions∆step_size∆set STEP
 →(0=TOL)/SOLVE
 OPTIONS dm∆ODEOptions∆abs_tol∆set TOL
 OPTIONS dm∆ODEOptions∆rel_tol∆set TOL
SOLVE:SOLVER←dm∆ODESolver∆new OPTIONS
 SOL←SOLVER dm∆ODESolver∆solve DECAY (,1) 0 TEND
 Y←↑LAST SOL
 PUT (28 LJ NAME),(14 LJ 6 FMT Y),(14 LJ 6 FMT |Y-EXACT),(8 LJ ⍕dm∆ODESolution∆steps_taken SOL),⍕dm∆ODESolution∆function_evaluations SOL
 dm∆Delete OPTIONS SOLVER SOL
∇

⍝ Prints one built-in system's step count and final state.
∇BUILTIN B;NAME;PARAMS;Y0;SOL
 (NAME PARAMS Y0)←B
 SOL←SOLVER dm∆ODESolver∆solve_builtin NAME PARAMS Y0 0 1
 PUT (22 LJ NAME),' steps=',(6 LJ ⍕dm∆ODESolution∆steps_taken SOL),' final state=',LIST LAST SOL
 dm∆Delete SOL
∇

⍝ An RK4 solver with step size B.
∇Z←RK4 B;OPTIONS
 OPTIONS←dm∆ODEOptions∆new ⍬
 OPTIONS dm∆ODEOptions∆method∆set dm∆StepMethod∆RK4
 OPTIONS dm∆ODEOptions∆step_size∆set B
 Z←dm∆ODESolver∆new OPTIONS
 dm∆Delete OPTIONS
∇

PUT '=================== dy/dt = -y, y(0) = 1 (live custom RHS) ==================='
DECAY←'Decay' dm∆RHS∆subclass ⍬
TEND←2
EXACT←*-TEND
PUT 'exact y(2.0) = ',6 FMT EXACT
PUT (28 LJ 'method'),(14 LJ 'y(2)'),(14 LJ 'abs.error'),(8 LJ 'steps'),'evals'
RUN 'Euler' dm∆StepMethod∆Euler 4 0.01 0
RUN 'Midpoint/RK2' dm∆StepMethod∆Midpoint 4 0.01 0
RUN 'RK4' dm∆StepMethod∆RK4 4 0.01 0
RUN 'RK45 adaptive' dm∆StepMethod∆RK45 4 0.1 1E¯8
RUN 'Adams-Bashforth' dm∆StepMethod∆AdamsBashforth 4 0.01 0
RUN 'Adams-Bashforth-Moulton' dm∆StepMethod∆AdamsMoulton 4 0.01 0

PUT NL,'=================== Every named built-in system (no director needed) ==================='
SOLVER←dm∆ODESolver∆new ⍬
BUILTIN 'exponential_decay' (,1) (,1)
BUILTIN 'logistic_growth' (1 1) (,0.5)
BUILTIN 'harmonic_oscillator' (,1) (1 0)
BUILTIN 'van_der_pol' (,1) (2 0)
BUILTIN 'lorenz' (10 28 (8÷3)) (1 1 1)
dm∆Delete SOLVER

PUT NL,'=================== Harmonic oscillator (energy conservation) ==================='
SOLVER←RK4 0.01
SOL←SOLVER dm∆ODESolver∆solve_builtin 'harmonic_oscillator' (,1) (1 0) 0 20
(X V)←LAST SOL
PUT 'x(20) = ',(6 FMT X),' (cos(20) = ',(6 FMT 2○20),')'
PUT 'energy x^2+v^2 = ',(6 FMT (X×X)+V×V),' (should stay near 1.0)'
T←TIMES SOL
Y←STATES SOL
PLOT←dm∆RPlot∆plot T (Y[;1]) 'l' 'x(t)'
R1←PLOT dm∆RPlot∆lines T (Y[;2]) 'v(t)'
R2←PLOT dm∆RPlot∆title 'Harmonic Oscillator'
R3←PLOT dm∆RPlot∆x_label 't'
R4←PLOT dm∆RPlot∆y_label 'state'
PLOT dm∆Plot∆save_svg 'ode_harmonic_oscillator_apl.svg'
PUT 'wrote ode_harmonic_oscillator_apl.svg'
dm∆Delete SOLVER SOL PLOT R1 R2 R3 R4

PUT NL,'=================== Lorenz attractor (phase plane) ==================='
SOLVER←RK4 0.005
SOL←SOLVER dm∆ODESolver∆solve_builtin 'lorenz' (10 28 (8÷3)) (1 1 1) 0 25
PUT 'steps_taken = ',⍕dm∆ODESolution∆steps_taken SOL
Y←STATES SOL
PLOT←dm∆RPlot∆plot (Y[;1]) (Y[;3]) 'l' 'trajectory'
R1←PLOT dm∆RPlot∆title 'Lorenz Attractor (x-z phase plane)'
R2←PLOT dm∆RPlot∆x_label 'x'
R3←PLOT dm∆RPlot∆y_label 'z'
PLOT dm∆Plot∆save_svg 'ode_lorenz_phase_plane_apl.svg'
PUT 'wrote ode_lorenz_phase_plane_apl.svg'
dm∆Delete SOLVER SOL PLOT R1 R2 R3 DECAY
