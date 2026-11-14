% CMD: just TARGET=dataframe_ex run-octave
% DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.

sales = datamunge.DataFrame ();
sales.add_string_column ('region', {'west', 'west', 'east', 'south', 'south', 'south'});
sales.add_string_column ('product', {'widget', 'widget', 'widget', 'gizmo', 'gizmo', 'gizmo'});
sales.add_numeric_column ('sales', [10 10 14 8 0 11], [1 1 1 1 0 1]);
sales.add_string_column ('quarter', {'Q1', 'Q1', 'Q1', 'Q2', 'Q2', ''}, [1 1 1 1 1 0]);

printf ('raw data\n%s\n\n', sales.to_string ());

cleaned = sales.drop_duplicates ({'region', 'product', 'sales', 'quarter'});
cleaned.fill_null_string ('quarter', 'unknown');
cleaned.fill_null_numeric ('sales', 0);
printf ('after drop_duplicates + fill_null\n%s\n\n', cleaned.to_string ());

selected = cleaned.select ({'region', 'sales', 'quarter'}).sort_by ('sales', false);
printf ('selected + sorted\n%s\n\n', selected.to_string ());

grouped = cleaned.group_by_sum ({'region'}, {'sales'}).sort_by ('sales', false);
printf ('group_by_sum(region)\n%s\n\n', grouped.to_string ());

targets = datamunge.DataFrame ();
targets.add_string_column ('region', {'west', 'east', 'south'});
targets.add_numeric_column ('target', [18 12 25]);
printf ('joined with targets\n%s\n\n', grouped.join (targets, 'region', 'region', 'left').to_string ());

shape = cleaned.shape ();
printf ('shape = (%d, %d)\n', shape(1), shape(2));
printf ('sales count = %d\n', cleaned.numeric_count ('sales'));
printf ('sales nulls = %d\n', cleaned.numeric_null_count ('sales'));
printf ('sales sum = %g\n', cleaned.numeric_sum ('sales'));
printf ('sales mean = %g\n', cleaned.numeric_mean ('sales'));
