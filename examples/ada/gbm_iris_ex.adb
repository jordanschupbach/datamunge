--  CMD: just TARGET=gbm_iris_ex run-ada
--  Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Gbm_Iris_Ex is
   --  Saves a plot as SVG and frees it.
   procedure Save_And_Free (P : in out R_Plot; Path : String) is
   begin
      P.Save (Path);
      Free (P);
   end Save_And_Free;

   Iris         : Data_Frame := Data.Iris;
   Model        : Gbm_Classifier;
   Few          : Gbm_Classifier;
   Table        : Data_Frame;
   Predictions  : String_Vector;
   P            : R_Plot;
   N            : Long_Long_Integer;
   Misclassified : Natural := 0;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Gbm_Classifier (Iris, "Species ~ Petal.Length + Petal.Width");
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

   P := Model.Plot_Training_Deviance;
   Save_And_Free (P, "gbm_iris_training_deviance.svg");
   P := Model.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Save_And_Free (P, "gbm_iris_decision_regions.svg");
   Put_Line (ASCII.LF & "Saved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

   --  A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
   --  at the training loss, gradually sharpening the decision boundary.
   --  New_Gbm_Classifier (Data, Formula, N_Trees, Learning_Rate, Max_Depth, Min_Samples_Split,
   --    Min_Samples_Leaf, Subsample, Seed)
   Few := New_Gbm_Classifier (Iris, "Species ~ Petal.Length + Petal.Width", 5);
   Put_Line (ASCII.LF & "5-round ensemble:   training accuracy=" & Fmt (100.0 * Few.Training_Accuracy)
             & "%  deviance=" & Fmt (Few.Training_Deviance.Last_Element));
   Put_Line ("100-round ensemble: training accuracy=" & Fmt (100.0 * Model.Training_Accuracy)
             & "%  deviance=" & Fmt (Model.Training_Deviance.Last_Element));
   P := Few.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Save_And_Free (P, "gbm_iris_decision_regions_5rounds.svg");
   Put_Line ("Saved gbm_iris_decision_regions_5rounds.svg");

   Free (Few);
   Free (Model);
   Free (Iris);
end Gbm_Iris_Ex;
