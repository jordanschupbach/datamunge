"""Generated algebra renderers delegate to the C++ serializers and return owned text."""
import gc
import pytest
from pydatamunge import datamunge as dm


def test_symbolic_tex_owns_source_and_preserves_grouping():
    x = dm.Expr.variable("x")
    expression = x.negate().power(dm.Expr.constant(2))
    source = expression.to_tex()
    del expression, x
    gc.collect()
    assert source == r"\left(-x\right)^{2}"
    assert isinstance(source, str)
    assert dm.Expr.variable("rate_1").to_tex() == r"\text{rate\_1}"
    with pytest.raises(ValueError):
        dm.Expr.constant(1).to_tex(0)


def test_polynomial_and_rational_tex_are_native_strings():
    p = dm.Polynomial([-1, 0, 1])
    assert p.to_tex("h") == r"{h}^{2} - 1"
    assert dm.Polynomial([1.23456789]).to_tex("x", 10) == "1.23456789"
    fraction = dm.RationalFunction(dm.Polynomial([1, 1]), dm.Polynomial([-1, 1]))
    assert fraction.to_tex() == r"\frac{x + 1}{x - 1}"
    with pytest.raises(ValueError):
        p.to_tex("")


def test_calculus_matches_known_integral_and_shifted_taylor():
    x = dm.Expr.variable("x")
    assert dm.definite_integral(x.power(dm.Expr.constant(2)), "x", 0, 3) == pytest.approx(9)
    taylor = dm.taylor_series(x.exp(), "x", 1, 4)
    assert taylor.evaluate(0) == pytest.approx(2.718281828459045)
    assert "{h}^{4}" in taylor.to_tex("h")
