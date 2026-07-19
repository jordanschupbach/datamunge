1;

datamunge;

function r = rosenbrock_dual(x0, x1)
  datamunge;
  a = Dual_subtract(Dual(1.0, 0.0), x0);
  b = Dual_subtract(x1, Dual_multiply(x0, x0));
  r = Dual_add(Dual_multiply(a, a), Dual_multiply_scalar(Dual_multiply(b, b), 100.0));
endfunction

function v = rosenbrock_value(x0, x1)
  a = 1.0 - x0;
  b = x1 - x0 * x0;
  v = a * a + 100.0 * b * b;
endfunction

printf("=================== Forward mode: scalar derivative ===================\n");
x0 = Dual(2.0, 1.0);  % seed derivative = 1 to read df/dx directly
f = Dual_subtract(Dual_multiply(Dual_multiply(x0, x0), x0), Dual_multiply_scalar(x0, 2.0));  % x^3 - 2x
printf("f(x) = x^3 - 2x, f'(2) = %g (exact: 10)\n", Dual_derivative(f));

printf("\n=================== Reverse mode: build a graph by hand ===================\n");
tape = Tape();
a = Var(tape, 2.0);
b = Var(tape, 3.0);
y = Var_add(Var_multiply(a, b), Var_sin(a));
printf("y = a*b + sin(a) at a=2, b=3 -> y = %g\n", Var_value(y));
adjoint = Tape_backward(tape, y);
printf("dy/da = %g (exact: b + cos(a))\n", adjoint{Var_index(a) + 1});
printf("dy/db = %g (exact: a)\n", adjoint{Var_index(b) + 1});

printf("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n");
px = 0.0;
py = 0.0;

% Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
gx = Dual_derivative(rosenbrock_dual(Dual(px, 1.0), Dual(py, 0.0)));
gy = Dual_derivative(rosenbrock_dual(Dual(px, 0.0), Dual(py, 1.0)));

% Reverse mode: one Tape/Var pass computes every partial at once.
tape2 = Tape();
vx = Var(tape2, px);
vy = Var(tape2, py);
va = Var_subtract(Var(tape2, 1.0), vx);
vb = Var_subtract(vy, Var_multiply(vx, vx));
vf = Var_add(Var_multiply(va, va), Var_multiply_scalar(Var_multiply(vb, vb), 100.0));
grad_rev = Tape_backward(tape2, vf);

printf("f(0,0) = %g\n", rosenbrock_value(px, py));
printf("gradient (forward mode): [%g, %g]\n", gx, gy);
printf("gradient (reverse mode): [%g, %g]\n", grad_rev{Var_index(vx) + 1}, grad_rev{Var_index(vy) + 1});

printf("\n=================== Jacobian of a vector-valued function ===================\n");
% f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
vx_val = 2.0;
vy_val = 3.0;

jac = zeros(3, 2);
cols = {[0, 1.0, 0.0], [1, 0.0, 1.0]};
for ci = 1:2
  c = cols{ci};
  col = c(1);
  seed_x = c(2);
  seed_y = c(3);
  ox = Dual_multiply(Dual(vx_val, seed_x), Dual(vx_val, seed_x));
  oy = Dual_multiply(Dual(vx_val, seed_x), Dual(vy_val, seed_y));
  oz = Dual_multiply(Dual_multiply(Dual(vy_val, seed_y), Dual(vy_val, seed_y)), Dual(vy_val, seed_y));
  jac(1, col + 1) = Dual_derivative(ox);
  jac(2, col + 1) = Dual_derivative(oy);
  jac(3, col + 1) = Dual_derivative(oz);
end
printf("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n");
for row = 1:3
  printf("  [%g, %g]\n", jac(row, 1), jac(row, 2));
end

printf("\n=================== Hessian via second-order forward mode (HyperDual) ===================\n");
% One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
mx = 1.0;
my = 1.0;  % the Rosenbrock function's minimum

function r = rosenbrock_hyperdual(x0, x1)
  datamunge;
  a = HyperDual_subtract(HyperDual(1.0, 0.0, 0.0, 0.0), x0);
  b = HyperDual_subtract(x1, HyperDual_multiply(x0, x0));
  r = HyperDual_add(HyperDual_multiply(a, a), HyperDual_multiply_scalar(HyperDual_multiply(b, b), 100.0));
endfunction

seeds = {[1.0, 0.0], [0.0, 1.0]};
h = zeros(2, 2);
for i = 1:2
  for j = 1:2
    e1 = seeds{i};
    e2 = seeds{j};
    hx = HyperDual(mx, e1(1), e2(1), 0.0);
    hy = HyperDual(my, e1(2), e2(2), 0.0);
    h(i, j) = HyperDual_eps1eps2(rosenbrock_hyperdual(hx, hy));
  end
end
printf("Hessian of the Rosenbrock function at its minimum (1,1):\n");
for row = 1:2
  printf("  [%g, %g]\n", h(row, 1), h(row, 2));
end

printf("\n=================== Gradient descent driven by reverse-mode gradients ===================\n");
point = [-1.2, 1.0];  % the classic Rosenbrock starting point
learning_rate = 0.001;
n_steps = 2000;
for step = 0:(n_steps - 1)
  t = Tape();
  vx2 = Var(t, point(1));
  vy2 = Var(t, point(2));
  va2 = Var_subtract(Var(t, 1.0), vx2);
  vb2 = Var_subtract(vy2, Var_multiply(vx2, vx2));
  vf2 = Var_add(Var_multiply(va2, va2), Var_multiply_scalar(Var_multiply(vb2, vb2), 100.0));
  grad = Tape_backward(t, vf2);
  loss = Var_value(vf2);
  point(1) = point(1) - learning_rate * grad{Var_index(vx2) + 1};
  point(2) = point(2) - learning_rate * grad{Var_index(vy2) + 1};
  if step == 0 || step == n_steps - 1
    printf("step %d: loss = %g, x = [%g, %g]\n", step, loss, point(1), point(2));
  end
end
printf("(true minimum is at [1, 1] with loss 0)\n");
