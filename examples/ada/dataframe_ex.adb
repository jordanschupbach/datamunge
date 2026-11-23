--  CMD: just TARGET=dataframe_ex run-ada
--  DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Dataframe_Ex is
   LF : constant Character := ASCII.LF;
   Sales   : Data_Frame := New_Data_Frame;
   Targets : Data_Frame := New_Data_Frame;
   Cleaned, Picked, Selected, Summed, Grouped, Joined : Data_Frame;
   Shape : Long_Long_Integer_Vector;
begin
   Sales.Add_String_Column ("region", ["west", "west", "east", "south", "south", "south"]);
   Sales.Add_String_Column ("product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]);
   Sales.Add_Numeric_Column ("sales", [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]);
   Sales.Add_String_Column ("quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0]);

   Put_Line ("raw data" & LF & Sales.Image & LF);

   Cleaned := Sales.Drop_Duplicates (["region", "product", "sales", "quarter"]);
   Cleaned.Fill_Null_String ("quarter", "unknown");
   Cleaned.Fill_Null_Numeric ("sales", 0.0);
   Put_Line ("after drop_duplicates + fill_null" & LF & Cleaned.Image & LF);

   --  Each step returns a new frame to free, so chains keep the intermediates.
   Picked := Cleaned.Select_Kw (["region", "sales", "quarter"]);
   Selected := Picked.Sort_By ("sales", False);
   Put_Line ("selected + sorted" & LF & Selected.Image & LF);

   Summed := Cleaned.Group_By_Sum (["region"], ["sales"]);
   Grouped := Summed.Sort_By ("sales", False);
   Put_Line ("group_by_sum(region)" & LF & Grouped.Image & LF);

   Targets.Add_String_Column ("region", ["west", "east", "south"]);
   Targets.Add_Numeric_Column ("target", [18.0, 12.0, 25.0]);
   Joined := Grouped.Join (Targets, "region", "region", "left");
   Put_Line ("joined with targets" & LF & Joined.Image & LF);

   Shape := Cleaned.Shape;
   Put_Line ("shape = (" & Img (Shape (1)) & ", " & Img (Shape (2)) & ")");
   Put_Line ("sales count = " & Img (Cleaned.Numeric_Count ("sales")));
   Put_Line ("sales nulls = " & Img (Cleaned.Numeric_Null_Count ("sales")));
   Put_Line ("sales sum = " & Fmt (Cleaned.Numeric_Sum ("sales")));
   Put_Line ("sales mean = " & Fmt (Cleaned.Numeric_Mean ("sales")));

   Free (Joined);
   Free (Grouped);
   Free (Summed);
   Free (Selected);
   Free (Picked);
   Free (Cleaned);
   Free (Targets);
   Free (Sales);
end Dataframe_Ex;
