--  CMD: just TARGET=lm_ex run-ada
--  Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;

procedure Lm_Ex is
   Cars    : Data_Frame := New_Data_Frame;
   Newcars : Data_Frame := New_Data_Frame;
   Model   : Lm;
   Table   : Data_Frame;
begin
   Cars.Add_Numeric_Column ("hp", [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]);
   Cars.Add_Numeric_Column ("wt", [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]);
   Cars.Add_String_Column ("transmission", ["manual", "manual", "manual", "automatic", "automatic",
                                            "automatic", "automatic", "automatic", "automatic", "automatic"]);
   Cars.Add_Numeric_Column ("mpg", [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]);

   Put_Line ("Fitting: mpg ~ hp + wt + transmission" & ASCII.LF);
   Model := New_Lm (Cars, "mpg ~ hp + wt + transmission");
   Model.Print_Summary;

   Table := Model.Anova;
   Put_Line (ASCII.LF & "Sequential ANOVA:" & ASCII.LF & Table.Image);
   Free (Table);

   Newcars.Add_Numeric_Column ("hp", [150.0, 90.0]);
   Newcars.Add_Numeric_Column ("wt", [3.0, 2.5]);
   Newcars.Add_String_Column ("transmission", ["manual", "automatic"]);
   Table := Model.Predict_Frame (Newcars, "confidence");
   Put_Line (ASCII.LF & "Predictions with 95% confidence intervals:" & ASCII.LF & Table.Image);
   Free (Table);

   Model.Save_Diagnostic_Plots ("lm_ex_diagnostics");
   Put_Line (ASCII.LF & "Saved diagnostic plots as lm_ex_diagnostics_*.svg");

   Free (Model);
   Free (Newcars);
   Free (Cars);
end Lm_Ex;
