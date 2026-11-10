"""Real native frozen-graph format, feed, ownership and detached-output binding tests."""
import gc
import pytest
from pydatamunge import nn as n

FIXTURE = [10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64]


def test_bytes_file_and_native_metadata_survive_source_collection(tmp_path):
    graph = n.FrozenGraph(n.ByteVector(FIXTURE))
    assert n.FrozenGraph(bytes(FIXTURE)).constant('c').item() == 2
    assert list(graph.node_names()) == ['c']
    assert graph.operation('c:0') == 'Const'
    assert graph.constant('c').dim() == 0 and graph.constant('c').item() == 2
    path = tmp_path / 'graph.pb'
    path.write_bytes(bytes(FIXTURE))
    loaded = n.FrozenGraph.load(str(path))
    path.unlink()
    gc.collect()
    assert loaded.constant('c').item() == 2


def test_named_detached_outputs_keep_order_and_outlive_model_and_feeds():
    graph = n.FrozenGraph(n.ByteVector(FIXTURE))
    feed = n.full([], 7).requires_grad_()
    result = graph.evaluate([('c', feed)], ['c:0', 'c'])
    assert [row[0] for row in result] == ['c:0', 'c']
    first, second = result[0][1], result[1][1]
    constant = graph.constant('c')
    del result, graph, feed
    gc.collect()
    assert first.item() == second.item() == 7 and constant.item() == 2
    assert not first.requires_grad() and not constant.requires_grad()


def test_malformed_bytes_bounds_ports_and_duplicate_feeds_reject():
    with pytest.raises(RuntimeError):
        n.FrozenGraph(n.ByteVector([10, 255]))
    with pytest.raises(RuntimeError):
        n.FrozenGraph(n.ByteVector(FIXTURE), 2)
    graph = n.FrozenGraph(n.ByteVector(FIXTURE))
    with pytest.raises(RuntimeError):
        graph.operation('c:1')
    with pytest.raises(RuntimeError):
        graph.constant('unknown')
    with pytest.raises(RuntimeError):
        graph.evaluate([('c', n.full([], 1)), ('c:0', n.full([], 2))], ['c'])
    assert not list(graph.evaluate([], []))
