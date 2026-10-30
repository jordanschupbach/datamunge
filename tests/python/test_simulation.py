"""Native task binding transitions, reset semantics and owned result lifetimes."""
import gc
import math
import pytest
from pydatamunge import datamunge as dm


def test_native_task_transitions_and_owned_outputs():
    task = dm.TaskBatch.create(dm.Task_InvertedPendulum, 2, 2, 42, 2)
    assert task.size() == 2
    assert task.observation_size() == 4
    assert task.action_size() == 1
    initial = task.output()
    task.step([0, 0])
    result = task.step([0, 0])
    assert list(result.truncated) == [1, 1]
    assert list(result.terminated) == [0, 0]
    assert list(result.final_valid) == [1, 1]
    assert list(result.final_observations) == list(result.observations)
    assert list(initial.truncated) == [0, 0]  # results are owned snapshots
    with pytest.raises(RuntimeError):
        task.step([0, 0])
    reset = task.reset([1, 0])
    assert list(reset.truncated) == [0, 1]
    assert list(reset.final_observations) == list(result.final_observations)
    del task
    gc.collect()
    assert list(result.rewards) == [1, 1]
    assert len(initial.observations) == 8


def test_hopper_and_validation_use_the_native_core():
    task = dm.TaskBatch.create(dm.Task_Hopper, 1)
    assert task.observation_size() == 11
    assert task.action_size() == 3
    with pytest.raises(ValueError):
        task.step([0, 0])
    with pytest.raises(ValueError):
        task.step([0, math.nan, 0])
    task.seed(123)
    first = list(task.reset().observations)
    task.step([0.1, 0, 0])
    assert list(task.output().observations) != first
    task.seed(123)
    assert list(task.reset().observations) == first
    with pytest.raises(ValueError):
        dm.TaskBatch.create(dm.Task_Hopper, 0)
