require "octruby"

module DataFrameEx
  module_function

  def encode_strings(values)
    "#{values.length}\u001e#{values.join("\u001f")}"
  end

  def dvector(values)
    out = Datamunge::DVector.new(values.length)
    values.each_with_index { |value, index| out[index] = value.to_f }
    out
  end

  def ivector(values)
    out = Datamunge::IVector.new(values.length)
    values.each_with_index { |value, index| out[index] = value.to_i }
    out
  end
end

sales = Datamunge::DataFrame.new
sales.add_string_column_encoded("region", DataFrameEx.encode_strings(%w[west west east south south south]))
sales.add_string_column_encoded("product", DataFrameEx.encode_strings(%w[widget widget widget gizmo gizmo gizmo]))
sales.add_numeric_column("sales", DataFrameEx.dvector([10, 10, 14, 8, 0, 11]), DataFrameEx.ivector([1, 1, 1, 1, 0, 1]))
sales.add_string_column_encoded("quarter", DataFrameEx.encode_strings(["Q1", "Q1", "Q1", "Q2", "Q2", ""]), DataFrameEx.ivector([1, 1, 1, 1, 1, 0]))

puts "raw data"
puts sales.to_string
puts

cleaned = sales.drop_duplicates_encoded(DataFrameEx.encode_strings(%w[region product sales quarter]))
cleaned.fill_null_string("quarter", "unknown")
cleaned.fill_null_numeric("sales", 0.0)
puts "after drop_duplicates + fill_null"
puts cleaned.to_string
puts

selected = cleaned.select_encoded(DataFrameEx.encode_strings(%w[region sales quarter])).sort_by("sales", false)
puts "selected + sorted"
puts selected.to_string
puts

grouped = cleaned.group_by_sum_encoded(DataFrameEx.encode_strings(["region"]), DataFrameEx.encode_strings(["sales"])).sort_by("sales", false)
puts "group_by_sum(region)"
puts grouped.to_string
puts

targets = Datamunge::DataFrame.new
targets.add_string_column_encoded("region", DataFrameEx.encode_strings(%w[west east south]))
targets.add_numeric_column("target", DataFrameEx.dvector([18, 12, 25]))
joined = grouped.join(targets, "region", "region", true)
puts "joined with targets"
puts joined.to_string
puts

shape = cleaned.shape
puts "shape = (#{shape[0]}, #{shape[1]})"
puts "sales count = #{cleaned.numeric_count('sales')}"
puts "sales nulls = #{cleaned.numeric_null_count('sales')}"
puts "sales sum = #{cleaned.numeric_sum('sales')}"
puts "sales mean = #{cleaned.numeric_mean('sales')}"
