"""Run the numerical examples that previously assumed a specific RNG library."""
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['batch_norm', 'normalization_layers', 'pooling'])
def test_normalization_and_pooling_examples_check_their_actual_samples(name):
    source = ROOT / 'examples/python/nn/0_foundations/0.2_reusable_layer_library' / (name + '.py')
    example = runpy.run_path(str(source))
    assert example['main'](['epochs=5'] if name == 'batch_norm' else []) == 0
    if name == 'batch_norm':
        # The original full statistics budget must obey the same recurrence.
        example['moving_statistics'](300)
