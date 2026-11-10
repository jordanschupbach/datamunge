"""Training-only batch-axis competition and independent deployment endpoints."""
import gc
from pydatamunge import nn


def test_lifetime_training_changes_with_companions_but_deployment_does_not():
    p = nn.ParameterStore(1931)
    sample = nn.ones([1, 3])
    nn.CompetitiveAutoencoder(p, sample, True, False, 3, 2)
    p.set('encoder/W', nn.full([3, 2], .2))
    p.set('encoder/b', nn.full([2], .1))
    low = nn.CompetitiveAutoencoder(p, nn.concat([sample, nn.zeros([7, 3])], 0), True, False, 3, 2)
    high = nn.CompetitiveAutoencoder(p, nn.concat([sample, nn.full([7, 3], 2)], 0), True, False, 3, 2)
    assert list(nn.take_rows(low.mask, [0]).values()) == [1, 1]
    assert list(nn.take_rows(high.mask, [0]).values()) == [0, 0]
    assert list(nn.take_rows(low.code, [0]).values()) == list(nn.take_rows(high.code, [0]).values())
    assert list(nn.take_rows(low.reconstruction, [0]).values()) == list(nn.take_rows(high.reconstruction, [0]).values())
    assert list(nn.take_rows(low.training_reconstruction, [0]).values()) != list(nn.take_rows(high.training_reconstruction, [0]).values())
    assert sum(v > 0 for v in low.code.values()) == 16
    assert sum(low.mask.values()) == 2 and not low.mask.requires_grad()
    reconstruction, code = low.training_reconstruction, low.code
    expected = tuple(code.values())
    del low, high
    gc.collect()
    assert tuple(code.values()) == expected
    gradients = nn.grad(nn.sum(reconstruction), p.parameters())
    assert len(gradients) == 4 and all(any(v != 0 for v in g.values()) for g in gradients)


def test_lifetime_selected_zero_slots_and_partial_batch_ceiling_are_explicit():
    for batch in [1, 7, 8, 9, 32]:
        scores = nn.zeros([batch, 2]).requires_grad_()
        mask = nn.CompetitiveMask(scores, True)
        keep = (batch + 7) // 8
        assert sum(mask.values()) == 2 * keep
        assert list(mask.values()) == [int(row < keep) for row in range(batch) for _ in range(2)]
        assert all(v == 0 for v in nn.mul(scores, mask).values())
        assert list(nn.grad(nn.sum(nn.mul(scores, mask)), [scores])[0].values()) == list(mask.values())
