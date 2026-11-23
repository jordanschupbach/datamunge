--  CMD: just TARGET=gaussian_process_regression_iris_ex run-ada
--  Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Gaussian_Process_Regression_Iris_Ex is
   --  Saves a plot as SVG and frees it.
   procedure Save_And_Free (P : in out R_Plot; Path : String) is
   begin
      P.Save (Path);
      Free (P);
   end Save_And_Free;

   Formula : constant String := "Petal.Length ~ Petal.Width";
   Iris    : Data_Frame := Data.Iris;
   Query   : Data_Frame := New_Data_Frame;
   Model   : Gaussian_Process_Regression;
   Detail  : Data_Frame;
   P       : R_Plot;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols");
   Put_Line ("formula: " & Formula & ASCII.LF);

   --  Both the length scale and the noise ratio are auto-selected by maximizing the exact log
   --  marginal likelihood.
   Model := New_Gaussian_Process_Regression (Iris, Formula);
   Model.Print_Summary;

   P := Model.Plot_Fit (Iris);
   Save_And_Free (P, "gpr_iris_fit.svg");
   P := Model.Plot_Length_Scale_Profile;
   Save_And_Free (P, "gpr_iris_length_scale_profile.svg");
   Put_Line (ASCII.LF & "Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

   --  Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
   --  interval at every point -- including far outside the training data, where it should widen
   --  substantially as the model's uncertainty grows.
   Query.Add_Numeric_Column ("Petal.Width", [0.2, 1.3, 2.5, 10.0]);
   Detail := Model.Predict_Frame (Query, "confidence");
   Put_Line (ASCII.LF & "Predictions with 95% confidence intervals:");
   Put_Line (Detail.Image);
   Put_Line ("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its"
             & ASCII.LF & " interval is than the in-range predictions.)");

   Free (Detail);
   Free (Model);
   Free (Query);
   Free (Iris);
end Gaussian_Process_Regression_Iris_Ex;
