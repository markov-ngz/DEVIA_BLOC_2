import pytest 
from tune_model import train_model
from transformers import TFMarianMTModel

def test_fine_tune_model(processed_ds, model_and_tokenizer,epochs):
    model , tokenizer = model_and_tokenizer
    model ,ds = processed_ds

    test_model = train_model(int(epochs), model, ds['train'], ds['valid'])

    assert isinstance(test_model,TFMarianMTModel) == True 