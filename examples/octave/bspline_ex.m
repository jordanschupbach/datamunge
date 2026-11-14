% CMD: just TARGET=bspline_ex run-octave
% Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.

% Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
x = [];
y = [];
z = [];
for iy = 0:7
  for ix = 0:7
    xv = ix / 7.0;
    yv = iy / 7.0;
    x(end+1) = xv;
    y(end+1) = yv;
    z(end+1) = sin (xv) + yv * yv;
  end
end
data = datamunge.DataFrame ();
data.add_numeric_column ('x', x);
data.add_numeric_column ('y', y);
data.add_numeric_column ('z', z);

surface = datamunge.LM (data, 'z ~ bs(x, y)');
printf ('Fitted z ~ bs(x, y)\n');
surface.print_summary ();

new_points = datamunge.DataFrame ();
new_points.add_numeric_column ('x', [0.25 0.75]);
new_points.add_numeric_column ('y', [0.50 0.25]);
pred = surface.predict (new_points);
printf ('Predictions: [%.6f, %.6f]\n', pred(1), pred(2));
