% CMD: just TARGET=tensor_ex run-octave
% Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.

1;

function s = shape_string (tensor)
  s = ['[' strjoin(arrayfun (@num2str, tensor.shape (), 'UniformOutput', false), ', ') ']'];
end

printf ('=================== Construction ===================\n');
printf ('zeros([2,3]): %s\n', datamunge.Tensor.zeros ([2 3]).to_string ());
printf ('eye(3): %s\n', datamunge.Tensor.eye (3).to_string ());

r = datamunge.Tensor.arange (0.0, 12.0, 1.0).reshape ([3 4]);
printf ('arange(0,12).reshape([3,4]): %s\n', r.to_string ());

printf ('\n=================== Shape ops ===================\n');
printf ('transpose -> shape %s\n', shape_string (r.transpose_ ()));
printf ('slice(axis=1, start=1, stop=3): %s\n', r.slice (1, 1, 3).to_string ());

printf ('\n=================== Broadcasting arithmetic ===================\n');
col = datamunge.Tensor.from_values ([3 1], [1.0 2.0 3.0]);
row = datamunge.Tensor.from_values ([1 4], [10.0 20.0 30.0 40.0]);
printf ('(3,1) + (1,4) -> %s\n', col.add (row).to_string ());

printf ('\n=================== Reductions ===================\n');
printf ('r.sum() = %g, r.mean() = %g\n', r.sum (), r.mean ());
printf ('column means (axis=0): %s\n', r.mean_axis (0).to_string ());

printf ('\n=================== Linear algebra ===================\n');
a = datamunge.Tensor.from_values ([2 3], [1.0 2.0 3.0 4.0 5.0 6.0]);
b = datamunge.Tensor.from_values ([3 2], [7.0 8.0 9.0 10.0 11.0 12.0]);
printf ('matmul(2x3, 3x2) -> %s\n', a.matmul (b).to_string ());
v1 = datamunge.Tensor.from_values (3, [1.0 2.0 3.0]);
v2 = datamunge.Tensor.from_values (3, [4.0 5.0 6.0]);
printf ('dot([1,2,3], [4,5,6]) = %g\n', v1.dot (v2));
printf ('outer(v1, v2) -> %s\n', v1.outer (v2).to_string ());

printf ('\n=================== Comparisons & masks ===================\n');
mask = r.greater_equal (datamunge.Tensor.full ([3 4], 6.0));
printf ('r >= 6 -> %s\n', mask.to_string ());
printf ('count(r >= 6) = %g\n', mask.sum ());

printf ('\n=================== A real dataset as a Tensor ===================\n');
iris = datamunge.Data.iris ();
n = iris.nrows ();
columns = {'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'};
flat = zeros (1, n * numel (columns));
k = 0;
for i = 0:n - 1
  for j = 1:numel (columns)
    k += 1;
    flat(k) = iris.numeric_at (columns{j}, i);
  end
end
x = datamunge.Tensor.from_values ([n 4], flat);
printf ('iris feature tensor shape: %s\n', shape_string (x));

feature_means = x.mean_axis (0);
centered = x.subtract (feature_means.reshape ([1 4]));
scatter = centered.transpose_ ().matmul (centered);
printf ('feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %s\n', feature_means.to_string ());
printf ('(X-mean)^T (X-mean) [4x4 scatter matrix]: %s\n', scatter.to_string (16));

species = cell (1, n);
for i = 0:n - 1
  species{i + 1} = iris.string_at ('Species', i);
end
species_tensor = datamunge.Tensor.from_string_values (n, species);
setosa_mask = species_tensor.equal (datamunge.Tensor.from_string_values (1, {'setosa'}));
printf ('setosa count = %g (of %d rows)\n', setosa_mask.sum (), n);
