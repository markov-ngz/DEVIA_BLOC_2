from pprint import pprint
from tensorflow.python.framework.ops import EagerTensor
from transformers.tokenization_utils_base import BatchEncoding


def test_model_tokenizer(model_and_tokenizer,sample_text):
    tokenizer , model = model_and_tokenizer
    batch = tokenizer([sample_text], return_tensors="tf")
    assert isinstance(batch,BatchEncoding)
    assert [key for key in {**batch}.keys()] == ['input_ids','attention_mask']
    gen = model.generate(**batch)
    assert isinstance(gen,EagerTensor)
    preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
    assert isinstance(preds,list)
    assert isinstance(preds[0],str)
    assert 'generate' in model.__dir__()
    assert 'compile' in model.__dir__()
    assert 'fit' in model.__dir__()
    assert 'batch_decode' in tokenizer.__dir__()
    assert 1 == 1 
