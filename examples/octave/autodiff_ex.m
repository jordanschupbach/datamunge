% CMD: just TARGET=autodiff_ex run-octave
% Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).

% Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
% (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
% bound. This port demonstrates the same ideas with the bound named-method API.
% v.index () is the Var's 0-based position in the adjoint vector t.backward returns.

1;

function r = rosenbrock_dual (x0, x1)
  a = datamunge.Dual (1.0, 0.0).subtract (x0);
  b = x1.subtract (x0.multiply (x0));
  r = a.multiply (a).add (b.multiply (b).multiply_scalar (100.0));
end

function r = rosenbrock_value (x0, x1)
  r = (1.0 - x0)^2 + 100.0 * (x1 - x0 * x0)^2;
end

% Build the Rosenbrock graph on tape; returns the output Var.
function r = rosenbrock_var (tape, x0, x1)
  a = datamunge.Var (tape, 1.0).subtract (x0);
  b = x1.subtract (x0.multiply (x0));
  r = a.multiply (a).add (b.multiply (b).multiply_scalar (100.0));
end

function r = rosenbrock_hyperdual (x0, x1)
  a = datamunge.HyperDual (1.0, 0.0, 0.0, 0.0).subtract (x0);
  b = x1.subtract (x0.multiply (x0));
  r = a.multiply (a).add (b.multiply (b).multiply_scalar (100.0));
end

function r = adjoint_of (adjoints, v)
  r = adjoints(v.index () + 1);
end

function outputs = vector_fn (x, y)
  outputs = {x.multiply(x), x.multiply(y), y.multiply(y).multiply(y)};
end

% Shortest round-trip decimal text for x, with a trailing '.0' on whole numbers.
function s = num (x)
  if (x == fix (x) && abs (x) < 1e15)
    s = sprintf ('%d.0', x);
    return;
  end
  for p = 1:17
    s = sprintf ('%.*g', p, x);
    if (str2double (s) == x)
      return;
    end
  end
end

printf ('=================== Forward mode: scalar derivative ===================\n');
x0 = datamunge.Dual (2.0, 1.0);  % seed derivative = 1 to read df/dx directly
f = x0.multiply (x0).multiply (x0).subtract (x0.multiply_scalar (2.0));  % x^3 - 2x
printf ('f(x) = x^3 - 2x, f''(2) = %s (exact: 10)\n', num (f.derivative ()));

printf ('\n=================== Reverse mode: build a graph by hand ===================\n');
tape = datamunge.Tape ();
a = datamunge.Var (tape, 2.0);
b = datamunge.Var (tape, 3.0);
y = a.multiply (b).add (a.sin ());
printf ('y = a*b + sin(a) at a=2, b=3 -> y = %s\n', num (y.value ()));
adjoint = tape.backward (y);
printf ('dy/da = %s (exact: b + cos(a))\n', num (adjoint_of (adjoint, a)));
printf ('dy/db = %s (exact: a)\n', num (adjoint_of (adjoint, b)));

printf ('\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n');
px = 0.0;
py = 0.0;
% Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx = rosenbrock_dual (datamunge.Dual (px, 1.0), datamunge.Dual (py, 0.0)).derivative ();
gy = rosenbrock_dual (datamunge.Dual (px, 0.0), datamunge.Dual (py, 1.0)).derivative ();
% Reverse mode: one Tape/Var pass computes every partial at once.
tape = datamunge.Tape ();
vx = datamunge.Var (tape, px);
vy = datamunge.Var (tape, py);
grad = tape.backward (rosenbrock_var (tape, vx, vy));
printf ('f(0,0) = %s\n', num (rosenbrock_value (px, py)));
printf ('gradient (forward mode): [%s, %s]\n', num (gx), num (gy));
printf ('gradient (reverse mode): [%s, %s]\n', num (adjoint_of (grad, vx)), num (adjoint_of (grad, vy)));

printf ('\n=================== Jacobian of a vector-valued function ===================\n');
% f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
jac = zeros (3, 2);
seeds = [1.0 0.0; 0.0 1.0];  % row col = (sx, sy)
for col = 1:2
  outputs = vector_fn (datamunge.Dual (2.0, seeds(col, 1)), datamunge.Dual (3.0, seeds(col, 2)));
  for row = 1:numel (outputs)
    jac(row, col) = outputs{row}.derivative ();
  end
end
printf ('f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n');
for row = 1:3
  printf ('  [%s, %s]\n', num (jac(row, 1)), num (jac(row, 2)));
end

printf ('\n=================== Hessian via second-order forward mode (HyperDual) ===================\n');
% One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx = 1.0;  % the Rosenbrock function's minimum
my = 1.0;
seeds = [1.0 0.0; 0.0 1.0];  % row i = (x, y) seed
h = zeros (2, 2);
for i = 1:2
  for j = 1:2
    e1 = seeds(i, :);
    e2 = seeds(j, :);
    h(i, j) = rosenbrock_hyperdual (datamunge.HyperDual (mx, e1(1), e2(1), 0.0), ...
                                    datamunge.HyperDual (my, e1(2), e2(2), 0.0)).eps1eps2 ();
  end
end
printf ('Hessian of the Rosenbrock function at its minimum (1,1):\n');
for i = 1:2
  printf ('  [%s, %s]\n', num (h(i, 1)), num (h(i, 2)));
end

printf ('\n=================== Gradient descent driven by reverse-mode gradients ===================\n');
point = [-1.2 1.0];  % the classic Rosenbrock start
learning_rate = 0.001;
n_steps = 2000;
for step = 0:n_steps-1
  tape = datamunge.Tape ();
  vx = datamunge.Var (tape, point(1));
  vy = datamunge.Var (tape, point(2));
  vf = rosenbrock_var (tape, vx, vy);
  grad = tape.backward (vf);
  loss = vf.value ();
  point(1) -= learning_rate * adjoint_of (grad, vx);
  point(2) -= learning_rate * adjoint_of (grad, vy);
  if (step == 0 || step == n_steps - 1)
    printf ('step %d: loss = %s, x = [%s, %s]\n', step, num (loss), num (point(1)), num (point(2)));
  end
end
printf ('(true minimum is at [1, 1] with loss 0)\n');
