"""Example data preparation through SWIG; no external network in unit tests."""
from pathlib import Path
import os
import struct

import pytest

from pydatamunge import nn
from pydatamunge.nn_components import dataset_assets as assets, memory_components
from pydatamunge.nn_workflows import data_root


def idx(shape, values):
    return bytes([0, 0, 8, len(shape)]) + struct.pack('>' + 'I' * len(shape), *shape) + bytes(values)


def test_shared_default_cache_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv('DATAMUNGE_DATA_DIR', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    expected = tmp_path / 'cache/datamunge/datasets'
    assert Path(assets.DataRoot()) == data_root() == expected
    assert not expected.exists()  # Looking up a path must not download anything.
    monkeypatch.setenv('DATAMUNGE_DATA_DIR', str(tmp_path / 'chosen'))
    assert data_root() == tmp_path / 'chosen'


def test_cached_mnist_split_works_offline_without_other_split(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    (tmp_path / 't10k-images-idx3-ubyte').write_bytes(idx([2, 28, 28], [0] * 784 + [255] * 784))
    (tmp_path / 't10k-labels-idx1-ubyte').write_bytes(idx([2], [3, 7]))
    loaded = assets.LoadMnist(str(tmp_path), 'test')
    assert list(loaded.labels.values()) == [3, 7]
    assert loaded.images.shape() == (2, 28, 28, 1)
    assert list(loaded.images.values()) == [0] * 784 + [1] * 784
    assert not (tmp_path / 'train-images-idx3-ubyte.gz').exists()


def test_cached_cifar_split_is_preserved_offline(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    original = bytes([4]) + bytes([128]) * 3072
    file = tmp_path / 'test_batch.bin'
    file.write_bytes(original)
    loaded = assets.LoadCifar10(str(tmp_path), 'test')
    assert list(loaded.labels.values()) == [4]
    assert file.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['test_batch.bin']


@pytest.mark.parametrize('prepare', [assets.EnsureMnist, assets.EnsureCifar, assets.EnsureSentences])
def test_missing_offline_data_has_actionable_error(tmp_path, monkeypatch, prepare):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    with pytest.raises(RuntimeError, match='automatic download disabled.*DATAMUNGE_DATA_OFFLINE'):
        prepare(str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_corrupt_download_is_never_installed_and_path_is_not_shell_code(tmp_path, monkeypatch):
    mirror = tmp_path / 'mirror/mnist'
    mirror.mkdir(parents=True)
    (mirror / 'train-images-idx3-ubyte.gz').write_bytes(b'bad archive')
    monkeypatch.setenv('DATAMUNGE_DATA_MIRROR', (tmp_path / 'mirror').as_uri())
    monkeypatch.delenv('DATAMUNGE_DATA_OFFLINE', raising=False)
    destination = tmp_path / "spaces ';$(touch SHOULD_NOT_EXIST)"
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        assets.EnsureMnist(str(destination))
    assert not list(destination.iterdir())  # Includes cleanup of staging directories.
    assert not Path('SHOULD_NOT_EXIST').exists()
    # An interrupted attempt can be retried, and fails for the checksum again.
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        assets.EnsureMnist(str(destination))


def test_derived_and_memory_cached_assets_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    saliency = tmp_path / 'saliency'
    saliency.mkdir()
    (saliency / 'saliency_data.npz').write_bytes(b'caller supplied')
    assets.EnsureSaliency(str(saliency))
    assert (saliency / 'saliency_data.npz').read_bytes() == b'caller supplied'
    sentence = tmp_path / 'sentence'
    sentence.mkdir()
    (sentence / 'sentence.npz').write_bytes(b'caller supplied')
    assets.EnsureSentences(str(sentence))
    assert (sentence / 'sentence.npz').read_bytes() == b'caller supplied'
    directory = tmp_path / 'arbitrary copy task directory'
    memory_components.PrepareDataset(str(directory), True)
    first = (directory / 'data.npz').read_bytes()
    memory_components.PrepareDataset(str(directory), True)
    assert (directory / 'data.npz').read_bytes() == first
    arrays = dict(nn.load_npz(str(directory / 'data.npz')))
    assert arrays['fit_input'].shape() == (128, 9, 4)


def test_invalid_split_does_not_start_download(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    with pytest.raises(RuntimeError, match='split must be train or test'):
        assets.LoadMnist(str(tmp_path), 'invalid')
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == 'nt', reason='POSIX fake curl executable')
def test_cifar_mirror_failure_tries_author_and_cleans_up(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    curl = bin_dir / 'curl'
    curl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_CURL_LOG"\nexit 22\n')
    curl.chmod(0o755)
    log = tmp_path / 'requests'
    monkeypatch.setenv('PATH', str(bin_dir))
    monkeypatch.setenv('FAKE_CURL_LOG', str(log))
    monkeypatch.delenv('DATAMUNGE_DATA_OFFLINE', raising=False)
    monkeypatch.delenv('DATAMUNGE_DATA_MIRROR', raising=False)
    destination = tmp_path / 'cifar'
    with pytest.raises(RuntimeError, match='dataset download failed'):
        assets.EnsureCifar(str(destination))
    calls = log.read_text().splitlines()
    assert len(calls) == 2
    assert 'storage.googleapis.com/mirror.tensorflow.org/' in calls[0]
    assert '--url https://www.cs.toronto.edu/' in calls[1]
    assert not list(destination.iterdir())


def test_corrupt_mr_corpus_is_rejected_before_preparation(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAMUNGE_DATA_OFFLINE', '1')
    (tmp_path / 'rt-polaritydata.tar.gz').write_bytes(b'corrupt corpus')
    with pytest.raises(RuntimeError, match='MR corpus checksum mismatch'):
        assets.EnsureSentences(str(tmp_path))
    assert not (tmp_path / 'sentence.npz').exists()
    assert not list(tmp_path.glob('.prepare-*'))
