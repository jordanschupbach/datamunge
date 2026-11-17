"""Binding-level ownership, prediction and SVG checks for the README classifier."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pydatamunge import datamunge as dm


class MLPClassifierSearchTests(unittest.TestCase):
    def test_model_and_history_own_their_inputs(self):
        iris = dm.data.iris()
        fit = dm.MLPClassifierSearch(iris, "Species", "Petal.Length", "Petal.Width")
        self.assertEqual((fit.training_rows(), fit.validation_rows(), fit.test_rows()), (90, 30, 30))
        self.assertGreaterEqual(fit.best_hidden_nodes(), 2)
        self.assertLessEqual(fit.best_hidden_nodes(), 24)
        self.assertGreaterEqual(fit.test_accuracy(), .85)
        history = fit.trials()
        del iris
        self.assertIn("Test accuracy", fit.summary())
        del fit
        self.assertEqual(history.nrows(), 12)
        self.assertEqual(history.string_at("phase", 4), "bayesian")
        self.assertGreaterEqual(history.numeric_sum("cached"), 1)

    def test_predictors_only_and_native_plot_export(self):
        iris = dm.data.iris()
        fit = dm.MLPClassifierSearch(iris, "Species", "Petal.Length", "Petal.Width",
                                    2, 6, 3, 2, 40, 42)
        predictors = iris.select(["Petal.Length", "Petal.Width"])
        self.assertEqual(fit.predict(predictors), fit.predict(iris))
        self.assertEqual(len(fit.predict(predictors)), 150)
        plot = fit.plot_decision_regions(30)
        del fit
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "decision.svg"
            plot.save(str(target))
            root = ET.parse(target).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            text = target.read_text()
            for value in ("Petal.Length", "Petal.Width", "setosa", "versicolor", "virginica"):
                self.assertIn(value, text)

    def test_help_and_invalid_search_controls(self):
        self.assertIn("Bayesian", dm.MLPClassifierSearch.__doc__)
        with self.assertRaises(ValueError):
            dm.MLPClassifierSearch(dm.data.iris(), "Species", "Petal.Length", "Petal.Width", 2, 2)


if __name__ == "__main__":
    unittest.main()
