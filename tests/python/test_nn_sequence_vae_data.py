"""Native corpus preparation, complete validation and owning selected pool bindings."""
import gc
from pathlib import Path
import pytest
from pydatamunge import nn


def corpus(tmp_path):
    return Path(nn.prepare_example_data('sequence-vae-assets',str(tmp_path)))/'sequence_vae'


def test_selected_pool_and_all_tensor_fields_outlive_corpus_and_files(tmp_path):
    root=corpus(tmp_path)
    pools=nn.LoadSequenceVaeData(root)
    assert len(pools)==3
    fit=pools[0]
    assert tuple(fit.tokens.shape())==(288,7)
    exported=dict(nn.SequenceDataArrays(pools))
    assert len(exported)==15 and 'config' not in exported
    del pools
    gc.collect()
    fields=[getattr(fit,name) for name in ['tokens','inputs','lengths','factors','source']]
    del fit
    for path in root.iterdir():path.unlink()
    gc.collect()
    assert [tuple(t.shape()) for t in fields]==[(288,7),(288,7),(288,),(288,6),(288,)]
    assert nn.SequenceGrammarValid(fields[0],0)
    assert list(exported['fit_tokens'].values())==list(fields[0].values())


def test_vocabulary_copy_and_prefix_suffix_semantics():
    words=nn.SequenceVocabulary()
    assert len(words)==19 and words[:3]==['[pad]','[bos]','[eos]']
    words[0]='changed'
    assert nn.SequenceVocabulary()[0]=='[pad]'
    tokens=nn.from_vector_int64([3,5,8,11,13,2,999],[1,7])
    assert nn.SequenceText(tokens,0)=='a red circle moves left [eos]'
    assert nn.SequenceGrammarValid(tokens,0)
    assert not nn.SequenceGrammarValid(nn.zeros([1,7],nn.DType.Int32),0)
    with pytest.raises(RuntimeError):nn.SequenceText(tokens,1)
    with pytest.raises(RuntimeError):nn.SequenceGrammarValid(nn.zeros([1,7]),0)


def test_exact_loader_vocabulary_and_export_errors(tmp_path):
    root=corpus(tmp_path)
    (root/'vocabulary.txt').write_text('wrong\n')
    with pytest.raises(RuntimeError):nn.LoadSequenceVaeData(root)
    with pytest.raises(RuntimeError):nn.SequenceDataArrays([])
    with pytest.raises(RuntimeError):nn.LoadSequenceVaeData(root/'absent')
