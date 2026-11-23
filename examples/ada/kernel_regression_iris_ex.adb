--  CMD: just TARGET=kernel_regression_iris_ex run-ada
--  Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Kernel_Regression_Iris_Ex is
   Formula : constant String := "Petal.Length ~ Petal.Width";

   --  Saves a plot a model returned by value, then frees it.
   procedure Save_And_Free (Plot : R_Plot; Path : String) is
      P : R_Plot := Plot;
   begin
      P.Save (Path);
      Free (P);
   end Save_And_Free;

   Iris                : Data_Frame := Data.Iris;
   Model, Small, Large : Kernel_Regression;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols");
   Put_Line ("formula: " & Formula & ASCII.LF);

   Model := New_Kernel_Regression (Iris, Formula);  --  bandwidth auto-selected via leave-one-out CV
   Model.Print_Summary;

   Save_And_Free (Model.Plot_Fit (Iris), "kernel_regression_iris_fit.svg");
   Save_And_Free (Model.Plot_Cv_Curve, "kernel_regression_iris_cv.svg");
   Put_Line (ASCII.LF & "Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

   --  A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
   --  bias-variance story. New_Kernel_Regression (data, formula, kernel, bandwidth, n_bandwidth, standardize)
   Small := New_Kernel_Regression (Iris, Formula, "gaussian", 0.05);
   Put_Line (ASCII.LF & "bandwidth=0.05 (too small): LOO R-squared=" & Fmt (Small.R_Squared)
             & "  LOO RMSE=" & Fmt (Small.Rmse));
   Save_And_Free (Small.Plot_Fit (Iris), "kernel_regression_iris_fit_small_bandwidth.svg");

   Large := New_Kernel_Regression (Iris, Formula, "gaussian", 5.0);
   Put_Line ("bandwidth=5.0 (too large):  LOO R-squared=" & Fmt (Large.R_Squared)
             & "  LOO RMSE=" & Fmt (Large.Rmse));
   Save_And_Free (Large.Plot_Fit (Iris), "kernel_regression_iris_fit_large_bandwidth.svg");

   Put_Line ("bandwidth=" & Fmt (Model.Bandwidth) & " (CV-selected): LOO R-squared=" & Fmt (Model.R_Squared)
             & "  LOO RMSE=" & Fmt (Model.Rmse));
   Put_Line (ASCII.LF
             & "Saved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");

   Free (Large);
   Free (Small);
   Free (Model);
   Free (Iris);
end Kernel_Regression_Iris_Ex;
