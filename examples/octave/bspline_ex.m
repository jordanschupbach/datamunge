1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

% Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
x = [];
y = [];
z = [];
for iy = 0:7
  for ix = 0:7
    xv = ix / 7.0;
    yv = iy / 7.0;
    x(end + 1) = xv;
    y(end + 1) = yv;
    z(end + 1) = sin(xv) + yv * yv;
  end
end

data = DataFrame_empty();
DataFrame_add_numeric_column(data, "x", dv(x));
DataFrame_add_numeric_column(data, "y", dv(y));
DataFrame_add_numeric_column(data, "z", dv(z));

surface = LM(data, "z ~ bs(x, y)");
printf("Fitted z ~ bs(x, y)\n");
LM_print_summary(surface);

new_points = DataFrame_empty();
DataFrame_add_numeric_column(new_points, "x", dv([0.25, 0.75]));
DataFrame_add_numeric_column(new_points, "y", dv([0.50, 0.25]));
pred = LM_predict(surface, new_points);
% Returned std::vector<double> arrives as a native Octave cell array (1-based {i}).
printf("Predictions: [%.6f, %.6f]\n", pred{1}, pred{2});
