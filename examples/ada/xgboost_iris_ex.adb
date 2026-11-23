--  CMD: just TARGET=xgboost_iris_ex run-ada
--  XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Xgboost_Iris_Ex is
   --  Saves a plot a model returned by value, then frees it.
   procedure Save_And_Free (Plot : R_Plot; Path : String) is
      P : R_Plot := Plot;
   begin
      P.Save (Path);
      Free (P);
   end Save_And_Free;

   --  The last element of a vector (C#'s v[^1]).
   function Last (V : Long_Float_Vector) return Long_Float is (V (V.Last_Index));

   Formula       : constant String := "Species ~ Petal.Length + Petal.Width";
   Iris          : Data_Frame := Data.Iris;
   Table         : Data_Frame;
   Model, Heavy  : Xg_Boost_Classifier;
   Predictions   : String_Vector;
   N             : Long_Long_Integer;
   Misclassified : Natural := 0;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Xg_Boost_Classifier (Iris, Formula);
   Model.Print_Summary;

   Table := Model.Confusion_Matrix;
   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):" & ASCII.LF & Table.Image);
   Free (Table);

   Put_Line (ASCII.LF & "Misclassified rows:");
   Predictions := Model.Predict (Iris);
   N := Iris.Nrows;
   for I in 0 .. N - 1 loop
      declare
         Actual    : constant String := Iris.String_At ("Species", I);
         Predicted : constant String := Predictions (Positive (I + 1));
      begin
         if Predicted /= Actual then
            Misclassified := Misclassified + 1;
            Put_Line ("  row " & Img (I) & ": Petal.Length=" & Fmt (Iris.Numeric_At ("Petal.Length", I))
                      & " Petal.Width=" & Fmt (Iris.Numeric_At ("Petal.Width", I))
                      & "  actual=" & Actual & "  predicted=" & Predicted);
         end if;
      end;
   end loop;
   Put_Line (Img (Long_Long_Integer (Misclassified)) & " of " & Img (N) & " misclassified ("
             & Fmt (100.0 * Long_Float (Misclassified) / Long_Float (N)) & "%)");

   Save_And_Free (Model.Plot_Training_Deviance, "xgboost_iris_training_deviance.svg");
   Save_And_Free (Model.Plot_Decision_Regions ("Petal.Length", "Petal.Width"), "xgboost_iris_decision_regions.svg");
   Put_Line (ASCII.LF & "Saved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

   --  XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
   --  grows the same deep trees but keeps every leaf weight small, producing a much smoother
   --  decision boundary than the lightly-regularized default.
   --  New_Xg_Boost_Classifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
   --                           min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
   Heavy := New_Xg_Boost_Classifier (Iris, Formula, 100, 0.3, 6, 50.0);
   Put_Line (ASCII.LF & "lambda=1 (default):   training accuracy=" & Fmt (100.0 * Model.Training_Accuracy)
             & "%  deviance=" & Fmt (Last (Model.Training_Deviance)));
   Put_Line ("lambda=50 (heavy L2): training accuracy=" & Fmt (100.0 * Heavy.Training_Accuracy)
             & "%  deviance=" & Fmt (Last (Heavy.Training_Deviance)));
   Save_And_Free (Heavy.Plot_Decision_Regions ("Petal.Length", "Petal.Width"),
                  "xgboost_iris_decision_regions_heavy_lambda.svg");
   Put_Line ("Saved xgboost_iris_decision_regions_heavy_lambda.svg");

   Free (Heavy);
   Free (Model);
   Free (Iris);
end Xgboost_Iris_Ex;
