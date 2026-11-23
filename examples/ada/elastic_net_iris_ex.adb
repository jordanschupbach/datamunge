--  CMD: just TARGET=elastic_net_iris_ex run-ada
--  Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Elastic_Net_Iris_Ex is
   Formula : constant String :=
     "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

   Iris    : Data_Frame := Data.Iris;
   Ridge_Model   : Ridge;
   Lasso_Model   : Lasso;
   Elastic : Elastic_Net;

   --  Saves a model's coefficient-path and CV-curve figures, and frees them.
   procedure Save_Figures (Path, Cv : in out R_Plot; Name, Label : String) is
   begin
      Path.Save ("elastic_net_" & Name & "_path.svg");
      Cv.Save ("elastic_net_" & Name & "_cv.svg");
      Put_Line ("  " & Label & " elastic_net_" & Name & "_path.svg, elastic_net_" & Name & "_cv.svg");
      Free (Path);
      Free (Cv);
   end Save_Figures;

   Path, Cv : R_Plot;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols");
   Put_Line ("formula: " & Formula & ASCII.LF);

   Put_Line ("=================== Ridge ===================");
   Ridge_Model := New_Ridge (Iris, Formula);
   Ridge_Model.Print_Summary;

   Put_Line (ASCII.LF & "=================== Lasso ===================");
   Lasso_Model := New_Lasso (Iris, Formula);
   Lasso_Model.Print_Summary;

   Put_Line (ASCII.LF & "================= Elastic Net =================");
   --  New_Elastic_Net (Data, Formula, Alpha, Lambda, N_Lambda, Cv_Folds, Standardize, Seed)
   Elastic := New_Elastic_Net (Iris, Formula, 0.5);
   Elastic.Print_Summary;

   Put_Line (ASCII.LF & "Saved figures showing how each model's coefficients respond to the regularization strength, "
             & "and the cross-validation curve used to pick it:");

   Path := Ridge_Model.Plot_Coefficient_Path;
   Cv := Ridge_Model.Plot_Cv_Curve;
   Save_Figures (Path, Cv, "ridge", "ridge:      ");
   Path := Lasso_Model.Plot_Coefficient_Path;
   Cv := Lasso_Model.Plot_Cv_Curve;
   Save_Figures (Path, Cv, "lasso", "lasso:      ");
   Path := Elastic.Plot_Coefficient_Path;
   Cv := Elastic.Plot_Cv_Curve;
   Save_Figures (Path, Cv, "elasticnet", "elastic net:");

   Free (Elastic);
   Free (Lasso_Model);
   Free (Ridge_Model);
   Free (Iris);
end Elastic_Net_Iris_Ex;
