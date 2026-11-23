--  CMD: just TARGET=lmm_ex run-ada
--  Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;
with Interfaces;

procedure Lmm_Ex is
   use type Interfaces.Integer_32;

   Rng : Dot_Net_Random := New_Dot_Net_Random (2024);

   --  (0, 1]: avoids log(0) in the Box-Muller transform.
   function Uniform return Long_Float is (1.0 - Rng.Next_Double);

   function Gauss return Long_Float is
      U1 : constant Long_Float := Uniform;  --  C# evaluates the log's Uniform() first
      R  : constant Long_Float := C_Sqrt (-2.0 * C_Log (U1));
      U2 : constant Long_Float := Uniform;
   begin
      return R * C_Cos (2.0 * Pi * U2);
   end Gauss;

   N_Schools      : constant := 30;
   True_Intercept : constant Long_Float := 60.0;
   True_Slope     : constant Long_Float := 3.0;

   type School_Values is array (0 .. N_Schools - 1) of Long_Float;

   Penguins       : Data_Frame;
   Penguin_Model  : Lmm;
   School_Intercept, School_Slope : School_Values;
   School, Study_Hours, Score, Re : Long_Float_Vector;
   Df             : Data_Frame := New_Data_Frame;
   Population     : Data_Frame := New_Data_Frame;
   School0        : Data_Frame := New_Data_Frame;
   Model          : Lmm;
   Labels         : String_Vector;
   N_Students     : Interfaces.Integer_32;
   Hours          : Long_Float;
begin
   Put_Line ("=================== Random intercept on a real dataset (penguins) ===================");
   Penguins := Data.Penguins;
   Penguin_Model := New_Lmm (Penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
   Penguin_Model.Print_Summary;
   Free (Penguin_Model);
   Free (Penguins);

   Put_Line (ASCII.LF
             & "=================== Random intercept + slope on a simulated multi-school dataset ===================");
   --  30 schools, ~25 students each: score depends on study_hours, but both the baseline score
   --  and the return on an extra hour of study vary by school -- the textbook case for a
   --  random-intercept-and-slope model.
   for S in School_Intercept'Range loop
      School_Intercept (S) := Gauss * 6.0;
   end loop;
   for S in School_Slope'Range loop
      School_Slope (S) := Gauss * 1.2;
   end loop;

   for S in 0 .. N_Schools - 1 loop
      N_Students := 15 + Rng.Next (0, 21);
      for Student in 1 .. N_Students loop
         Hours := Rng.Next_Double * 10.0;
         School.Append (Long_Float (S));
         Study_Hours.Append (Hours);
         Score.Append (True_Intercept + School_Intercept (S) + (True_Slope + School_Slope (S)) * Hours
                       + Gauss * 4.0);
      end loop;
   end loop;
   Df.Add_Numeric_Column ("school", School);
   Df.Add_Numeric_Column ("study_hours", Study_Hours);
   Df.Add_Numeric_Column ("score", Score);

   Model := New_Lmm (Df, "score ~ study_hours + (1 + study_hours | school)");
   Model.Print_Summary;

   Put_Line (ASCII.LF & "True generating values: intercept=" & Fmt (True_Intercept) & ", slope=" & Fmt (True_Slope)
             & ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

   Put_Line (ASCII.LF & "--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
   Labels := Model.Group_Labels;
   for Idx in 0 .. 2 loop
      Re := Model.Random_Effects_For_Group (Long_Long_Integer (Idx));
      Put_Line ("school " & Labels (Idx + 1) & ": intercept shift=" & Fmt (Re (1))
                & ", slope shift=" & Fmt (Re (2)));
   end loop;

   Put_Line (ASCII.LF & "--- Prediction: population-level vs. school-adjusted ---");
   Population.Add_Numeric_Column ("study_hours", [5.0]);
   School0.Add_Numeric_Column ("study_hours", [5.0]);
   School0.Add_Numeric_Column ("school", [0.0]);
   Put_Line ("5 study hours, unseen school:      " & Fmt (Model.Predict (Population) (1)) & " (fixed effects only)");
   Put_Line ("5 study hours, school 0 (known):    " & Fmt (Model.Predict (School0) (1))
             & " (fixed effects + school 0's BLUP)");

   Free (Model);
   Free (School0);
   Free (Population);
   Free (Df);
end Lmm_Ex;
