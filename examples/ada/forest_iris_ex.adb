--  CMD: just TARGET=forest_iris_ex run-ada
--  Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Forest_Iris_Ex is
   --  Saves a plot as SVG and frees it.
   procedure Save_And_Free (P : in out R_Plot; Path : String) is
   begin
      P.Save (Path);
      Free (P);
   end Save_And_Free;

   Iris         : Data_Frame := Data.Iris;
   Model        : Random_Forest_Classifier;
   Small_Forest : Random_Forest_Classifier;
   Table        : Data_Frame;
   Predictions  : String_Vector;
   P            : R_Plot;
   N            : Long_Long_Integer;
   Misclassified : Natural := 0;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Random_Forest_Classifier (Iris, "Species ~ Petal.Length + Petal.Width");
   Model.Print_Summary;

   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):");
   Table := Model.Confusion_Matrix;
   Put_Line (Table.Image);
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

   P := Model.Plot_Classification (Iris, "Petal.Length", "Petal.Width");
   Save_And_Free (P, "forest_iris_classification.svg");
   P := Model.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Save_And_Free (P, "forest_iris_decision_regions.svg");
   Put_Line (ASCII.LF & "Saved forest_iris_classification.svg and forest_iris_decision_regions.svg");

   --  A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
   --  are added -- the defining random forest effect a single decision tree cannot demonstrate.
   --  New_Random_Forest_Classifier (Data, Formula, N_Trees, Max_Depth, Min_Samples_Split,
   --    Min_Samples_Leaf, Max_Features, Criterion, Bootstrap, Sample_Fraction, Seed)
   Small_Forest := New_Random_Forest_Classifier (Iris, "Species ~ Petal.Length + Petal.Width", 5);
   Put_Line (ASCII.LF & "5-tree forest:   training accuracy=" & Fmt (Small_Forest.Training_Accuracy * 100.0)
             & "%  OOB accuracy=" & Fmt (Small_Forest.Oob_Accuracy * 100.0) & "%");
   Put_Line ("100-tree forest: training accuracy=" & Fmt (Model.Training_Accuracy * 100.0)
             & "%  OOB accuracy=" & Fmt (Model.Oob_Accuracy * 100.0) & "%");
   P := Small_Forest.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Save_And_Free (P, "forest_iris_decision_regions_5trees.svg");
   Put_Line ("Saved forest_iris_decision_regions_5trees.svg");

   Free (Small_Forest);
   Free (Model);
   Free (Iris);
end Forest_Iris_Ex;
