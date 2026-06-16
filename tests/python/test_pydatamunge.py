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
