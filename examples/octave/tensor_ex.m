1;

datamunge;

function v = sv(t)
  datamunge;
  v = SizeVector();
  for i = 1:numel(t)
    SizeVector_push_back(v, t(i));
  end
endfunction

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = strv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

function s = shape_str(shp)
  parts = {};
  for i = 1:numel(shp)
    parts{end + 1} = num2str(shp{i});
  end
  s = strjoin(parts, ", ");
endfunction

printf("=================== Construction ===================\n");
z = Tensor_zeros(sv([2, 3]));
printf("zeros([2,3]): %s\n", Tensor_to_string(z));

eyet = Tensor_eye(3);
printf("eye(3): %s\n", Tensor_to_string(eyet));

r = Tensor_reshape(Tensor_arange(0.0, 12.0, 1.0), sv([3, 4]));
printf("arange(0,12).reshape([3,4]): %s\n", Tensor_to_string(r));

printf("\n=================== Shape ops ===================\n");
rt = Tensor_transpose(r);
printf("transpose -> shape [%s]\n", shape_str(Tensor_shape(rt)));
sliced = Tensor_slice(r, 1, 1, 3);
printf("slice(axis=1, start=1, stop=3): %s\n", Tensor_to_string(sliced));

printf("\n=================== Broadcasting arithmetic ===================\n");
col = Tensor_from_values(sv([3, 1]), dv([1, 2, 3]));
row = Tensor_from_values(sv([1, 4]), dv([10, 20, 30, 40]));
broadcast_sum = Tensor_add(col, row);
printf("(3,1) + (1,4) -> %s\n", Tensor_to_string(broadcast_sum));

printf("\n=================== Reductions ===================\n");
printf("r.sum() = %g, r.mean() = %g\n", Tensor_sum(r), Tensor_mean(r));
col_means = Tensor_mean_axis(r, 0);
printf("column means (axis=0): %s\n", Tensor_to_string(col_means));

printf("\n=================== Linear algebra ===================\n");
a = Tensor_from_values(sv([2, 3]), dv([1, 2, 3, 4, 5, 6]));
b = Tensor_from_values(sv([3, 2]), dv([7, 8, 9, 10, 11, 12]));
printf("matmul(2x3, 3x2) -> %s\n", Tensor_to_string(Tensor_matmul(a, b)));

v1 = Tensor_from_values(sv([3]), dv([1, 2, 3]));
v2 = Tensor_from_values(sv([3]), dv([4, 5, 6]));
printf("dot([1,2,3], [4,5,6]) = %g\n", Tensor_dot(v1, v2));
printf("outer(v1, v2) -> %s\n", Tensor_to_string(Tensor_outer(v1, v2)));

printf("\n=================== Comparisons & masks ===================\n");
mask = Tensor_greater_equal(r, Tensor_full(sv([3, 4]), 6.0));
printf("r >= 6 -> %s\n", Tensor_to_string(mask));
printf("count(r >= 6) = %g\n", Tensor_sum(mask));

printf("\n=================== A real dataset as a Tensor ===================\n");
iris = DataFrame_iris();
n = DataFrame_nrows(iris);
flat = [];
for i = 0:(n - 1)
  flat(end + 1) = DataFrame_numeric_at(iris, "Sepal.Length", i);
  flat(end + 1) = DataFrame_numeric_at(iris, "Sepal.Width", i);
  flat(end + 1) = DataFrame_numeric_at(iris, "Petal.Length", i);
  flat(end + 1) = DataFrame_numeric_at(iris, "Petal.Width", i);
end
x = Tensor_from_values(sv([n, 4]), dv(flat));
printf("iris feature tensor shape: [%s]\n", shape_str(Tensor_shape(x)));

feature_means = Tensor_mean_axis(x, 0);
centered = Tensor_subtract(x, Tensor_reshape(feature_means, sv([1, 4])));
scatter = Tensor_matmul(Tensor_transpose(centered), centered);
printf("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %s\n", Tensor_to_string(feature_means));
printf("(X-mean)^T (X-mean) [4x4 scatter matrix]: %s\n", Tensor_to_string(scatter, 16));

species = {};
for i = 0:(n - 1)
  species{end + 1} = DataFrame_string_at(iris, "Species", i);
end
species_tensor = Tensor_from_string_values(sv([n]), strv(species));
setosa_mask = Tensor_equal(species_tensor, Tensor_from_string_values(sv([1]), strv({"setosa"})));
printf("setosa count = %g (of %d rows)\n", Tensor_sum(setosa_mask), n);
