--  CMD: just TARGET=glm_iris_ex run-ada
--  Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Glm_Iris_Ex is
   Grid_N : constant := 100;

   Iris       : Data_Frame := Data.Iris;
   Sub        : Data_Frame := New_Data_Frame;
   Grid       : Data_Frame := New_Data_Frame;
   Count_Data : Data_Frame := New_Data_Frame;
   Curve_Frame : Data_Frame;
   Logit, Poisson : Glm;
   Is_Virginica, Petal_Length, Petal_Width, Fitted : Long_Float_Vector;
   Grid_Length, Grid_Width, Sepal_Width_All, Petal_Length_All, Count_All : Long_Float_Vector;
   Correct    : Natural := 0;
   Width_Mean, Pl_Min, Pl_Max : Long_Float;
begin
   --  Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
   --  the other two on these predictors, which sends logistic regression's coefficients toward
   --  +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
   --  binary split for a demo.
   for I in 0 .. Iris.Nrows - 1 loop
      declare
         Species : constant String := Iris.String_At ("Species", I);
      begin
         if Species = "versicolor" or else Species = "virginica" then
            Is_Virginica.Append ((if Species = "virginica" then 1.0 else 0.0));
            Petal_Length.Append (Iris.Numeric_At ("Petal.Length", I));
            Petal_Width.Append (Iris.Numeric_At ("Petal.Width", I));
         end if;
      end;
   end loop;

   Sub.Add_Numeric_Column ("Petal.Length", Petal_Length);
   Sub.Add_Numeric_Column ("Petal.Width", Petal_Width);
   Sub.Add_Numeric_Column ("is_virginica", Is_Virginica);

   Put_Line ("=================== Logistic regression (binomial, logit link) ===================");
   Logit := New_Glm (Sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
   Logit.Print_Summary;

   Fitted := Logit.Fitted_Values;
   for I in 1 .. Is_Virginica.Last_Index loop
      if (Fitted (I) >= 0.5) = (Is_Virginica (I) >= 0.5) then
         Correct := Correct + 1;
      end if;
   end loop;
   Put_Line (ASCII.LF & "Resubstitution accuracy at 0.5 threshold: "
             & Fmt (100.0 * Long_Float (Correct) / Long_Float (Is_Virginica.Length)) & "%");

   Logit.Save_Diagnostic_Plots ("glm_logistic_iris");
   Put_Line (ASCII.LF
             & "Saved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

   --  Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
   --  classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
   Width_Mean := 0.0;
   for W of Petal_Width loop
      Width_Mean := Width_Mean + W;
   end loop;
   Width_Mean := Width_Mean / Long_Float (Petal_Width.Length);
   Pl_Min := Petal_Length (1);
   Pl_Max := Petal_Length (1);
   for L of Petal_Length loop
      Pl_Min := Long_Float'Min (Pl_Min, L);
      Pl_Max := Long_Float'Max (Pl_Max, L);
   end loop;
   Pl_Min := Pl_Min - 0.3;
   Pl_Max := Pl_Max + 0.3;
   for I in 0 .. Grid_N - 1 loop
      Grid_Length.Append (Pl_Min + (Pl_Max - Pl_Min) * Long_Float (I) / Long_Float (Grid_N - 1));
      Grid_Width.Append (Width_Mean);
   end loop;
   Grid.Add_Numeric_Column ("Petal.Length", Grid_Length);
   Grid.Add_Numeric_Column ("Petal.Width", Grid_Width);
   Curve_Frame := Logit.Predict_Frame (Grid, "confidence");
   Put_Line (ASCII.LF & "Predicted-probability curve (first 5 rows):");
   Put_Line (Curve_Frame.Image (5));
   Free (Curve_Frame);

   --  Poisson regression, for contrast: same IRLS engine, different family/link.
   Put_Line (ASCII.LF & "=================== Poisson regression (log link) ===================");
   for I in 0 .. Iris.Nrows - 1 loop
      Sepal_Width_All.Append (Iris.Numeric_At ("Sepal.Width", I));
      Petal_Length_All.Append (Iris.Numeric_At ("Petal.Length", I));
      Count_All.Append (Long_Float'Floor (Iris.Numeric_At ("Sepal.Length", I) + 0.5));
   end loop;
   Count_Data.Add_Numeric_Column ("Sepal.Width", Sepal_Width_All);
   Count_Data.Add_Numeric_Column ("Petal.Length", Petal_Length_All);
   Count_Data.Add_Numeric_Column ("count", Count_All);
   Poisson := New_Glm (Count_Data, "count ~ Sepal.Width + Petal.Length", "poisson");
   Poisson.Print_Summary;

   Free (Poisson);
   Free (Logit);
   Free (Count_Data);
   Free (Grid);
   Free (Sub);
   Free (Iris);
end Glm_Iris_Ex;
