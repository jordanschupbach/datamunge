import pytest


def test_import_and_hello():
    from pydatamunge import datamunge

    assert hasattr(datamunge, "hello")
    datamunge.hello()


def test_std_pair_templates():
    from pydatamunge import datamunge

    p = datamunge.DPair(1.25, 2.5)
    assert pytest.approx(p.first) == 1.25
    assert pytest.approx(p.second) == 2.5


def test_std_vector_templates():
    from pydatamunge import datamunge

    v = datamunge.DVector()
    v.push_back(3.0)
    v.push_back(4.5)

    assert len(v) == 2
    assert pytest.approx(v[0]) == 3.0
    assert pytest.approx(v[1]) == 4.5


def test_can_pass_callback_into_cpp():
    from pydatamunge import datamunge

    class TimesTwo(datamunge.Callback):
        def call(self, x):
            return x * 2.0

    cb = TimesTwo()
    assert pytest.approx(datamunge.call_with_callback(3.0, cb)) == 6.0

    v = datamunge.make_dvector(1.0, 2.0, 3.0)
    out = datamunge.map_dvector_with_callback(v, cb)
    assert pytest.approx(datamunge.sum_dvector(out)) == 12.0


def test_lm_fits_and_predicts():
    from pydatamunge import datamunge

    df = datamunge.DataFrame()
    df.add_numeric_column("x", datamunge.DVector([1.0, 2.0, 3.0, 4.0, 5.0]))
    df.add_numeric_column("y", datamunge.DVector([3.0, 5.0, 7.0, 9.0, 11.0]))

    model = datamunge.LM(df, "y ~ x")
    coefficients = list(model.coefficients())
    assert pytest.approx(coefficients[0], abs=1e-9) == 1.0
    assert pytest.approx(coefficients[1], abs=1e-9) == 2.0
    assert pytest.approx(model.r_squared(), abs=1e-9) == 1.0

    newdata = datamunge.DataFrame()
    newdata.add_numeric_column("x", datamunge.DVector([6.0]))
    predictions = list(model.predict(newdata))
    assert pytest.approx(predictions[0], abs=1e-9) == 13.0


def test_pca_matches_known_iris_variance_explained():
    from pydatamunge import datamunge

    iris = datamunge.DataFrame.iris()
    features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

    pca = datamunge.PCA(iris, features, True, True)
    assert pca.observations() == 150
    assert pca.num_components() == 4

    ratio = list(pca.explained_variance_ratio())
    assert pytest.approx(ratio[0], abs=1e-3) == 0.7296
    assert pytest.approx(ratio[1], abs=1e-3) == 0.2285
    assert pytest.approx(sum(ratio), abs=1e-9) == 1.0

    scores_df = pca.scores_frame()
    assert scores_df.nrows() == 150
    assert list(scores_df.columns()) == ["PC1", "PC2", "PC3", "PC4"]


def test_mds_embeds_iris_with_high_goodness_of_fit():
    from pydatamunge import datamunge

    iris = datamunge.DataFrame.iris()
    features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

    mds = datamunge.MDS(iris, features, 2, "euclidean")
    assert mds.observations() == 150
    assert mds.n_components() == 2
    assert mds.goodness_of_fit() > 0.9

    embedding_df = mds.embedding_frame()
    assert embedding_df.nrows() == 150
    assert list(embedding_df.columns()) == ["Dim1", "Dim2"]


def test_pca_and_mds_plot_grouped_by_species():
    from pydatamunge import datamunge

    iris = datamunge.DataFrame.iris()
    features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

    pca = datamunge.PCA(iris, features, True, True)
    species = [iris.string_at("Species", i) for i in pca.kept_row_indices()]
    scatter = pca.plot_scores_grouped(species)
    assert scatter is not None

    mds = datamunge.MDS(iris, features, 2, "euclidean")
    mds_scatter = mds.plot_embedding_grouped(species)
    assert mds_scatter is not None


def test_ode_solver_rk4_matches_exponential_decay_closed_form():
    from pydatamunge import datamunge

    import math

    options = datamunge.ODEOptions()
    options.method = datamunge.StepMethod_RK4
    options.step_size = 0.01
    solver = datamunge.ODESolver(options)

    sol = solver.solve_builtin("exponential_decay", datamunge.DVector([1.0]), datamunge.DVector([1.0]), 0.0, 2.0)
    last = sol.state_at(sol.size() - 1)
    assert pytest.approx(last[0], abs=1e-6) == math.exp(-2.0)
    assert sol.time_at(0) == 0.0
    assert pytest.approx(sol.time_at(sol.size() - 1)) == 2.0


def test_ode_solver_live_custom_rhs_via_director():
    from pydatamunge import datamunge

    import math

    class ExponentialDecay(datamunge.RHS):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def evaluate(self, t, y):
            return datamunge.DVector([-self.k * y[0]])

    rhs = ExponentialDecay(1.0)
    solver = datamunge.ODESolver()
    sol = solver.solve(rhs, datamunge.DVector([1.0]), 0.0, 2.0)
    last = sol.state_at(sol.size() - 1)
    assert pytest.approx(last[0], abs=1e-6) == math.exp(-2.0)


def test_ode_solver_covers_every_named_builtin_system():
    from pydatamunge import datamunge

    solver = datamunge.ODESolver()
    systems = [
        ("exponential_decay", datamunge.DVector([1.0]), datamunge.DVector([1.0])),
        ("logistic_growth", datamunge.DVector([1.0, 1.0]), datamunge.DVector([0.5])),
        ("harmonic_oscillator", datamunge.DVector([1.0]), datamunge.DVector([1.0, 0.0])),
        ("van_der_pol", datamunge.DVector([1.0]), datamunge.DVector([2.0, 0.0])),
        ("lorenz", datamunge.DVector([10.0, 28.0, 8.0 / 3.0]), datamunge.DVector([1.0, 1.0, 1.0])),
    ]
    for system, params, y0 in systems:
        sol = solver.solve_builtin(system, params, y0, 0.0, 1.0)
        assert sol.size() > 0


# No test exercises solve_builtin's unknown-system error path: any C++ exception thrown across
# the SWIG-Python boundary (confirmed here, and separately for R) aborts the whole interpreter
# process rather than raising a catchable Python exception, since no %exception block exists in
# this binding. This is pre-existing and repo-wide, not specific to ODESolver; see DataFrame.join
# with an invalid join_type for the same crash. Fixing it needs a global %exception translation
# layer, out of scope here.
