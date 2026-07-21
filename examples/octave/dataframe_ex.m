1;

datamunge;

% Octave's SWIG bindings use flat ClassName_method(obj, ...) function calls (no obj.method()
% sugar), and plain Octave arrays/cell arrays never auto-convert to any vector<T> parameter --
% build a real DVector/SVector/IVector via the object constructor + push_back.
function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

function v = iv(t)
  datamunge;
  v = IVector();
  for i = 1:numel(t)
    IVector_push_back(v, t(i));
  end
endfunction

sales = DataFrame_empty();
DataFrame_add_string_column(sales, "region", sv({"west", "west", "east", "south", "south", "south"}));
DataFrame_add_string_column(sales, "product", sv({"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"}));
DataFrame_add_numeric_column(sales, "sales", dv([10, 10, 14, 8, 0, 11]), iv([1, 1, 1, 1, 0, 1]));
DataFrame_add_string_column(sales, "quarter", sv({"Q1", "Q1", "Q1", "Q2", "Q2", ""}), iv([1, 1, 1, 1, 1, 0]));

printf("raw data\n");
printf("%s\n\n", DataFrame_to_string(sales));

cleaned = DataFrame_drop_duplicates(sales, sv({"region", "product", "sales", "quarter"}));
DataFrame_fill_null_string(cleaned, "quarter", "unknown");
DataFrame_fill_null_numeric(cleaned, "sales", 0.0);
printf("after drop_duplicates + fill_null\n");
printf("%s\n\n", DataFrame_to_string(cleaned));

selected = DataFrame_sort_by(DataFrame_select(cleaned, sv({"region", "sales", "quarter"})), "sales", false);
printf("selected + sorted\n");
printf("%s\n\n", DataFrame_to_string(selected));

grouped = DataFrame_sort_by(DataFrame_group_by_sum(cleaned, sv({"region"}), sv({"sales"})), "sales", false);
printf("group_by_sum(region)\n");
printf("%s\n\n", DataFrame_to_string(grouped));

targets = DataFrame_empty();
DataFrame_add_string_column(targets, "region", sv({"west", "east", "south"}));
DataFrame_add_numeric_column(targets, "target", dv([18, 12, 25]));
joined = DataFrame_join(grouped, targets, "region", "region", "left");
printf("joined with targets\n");
printf("%s\n\n", DataFrame_to_string(joined));

shape = DataFrame_shape(cleaned);
printf("shape = (%d, %d)\n", shape{1}, shape{2});
printf("sales count = %g\n", DataFrame_numeric_count(cleaned, "sales"));
printf("sales nulls = %g\n", DataFrame_numeric_null_count(cleaned, "sales"));
printf("sales sum = %g\n", DataFrame_numeric_sum(cleaned, "sales"));
printf("sales mean = %g\n", DataFrame_numeric_mean(cleaned, "sales"));
