# import pytest 
# from tune_model import train_model
# from transformers import TFMarianMTModel
# import time
# from  evaluate_model import get_bleu_score

# def test_train_model(processed_ds, model_and_tokenizer,epochs,sample_text):
#     tokenizer , _ = model_and_tokenizer
#     model ,ds = processed_ds

#     # try to train the model for 2 minutes 
#     test_model = train_model(int(epochs), model, ds['train'], ds['valid'])
    
#     # test the type of the element of response
#     assert isinstance(test_model,TFMarianMTModel) == True 

#     score = get_bleu_score(ds['valid'],test_model,tokenizer)

#     assert 'score' in score.keys()

#     batch = tokenizer([sample_text], return_tensors="tf")
    
#     # asynchronous here 
#     gen = model.generate(**batch)
#     preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

#     assert isinstance(preds[0],str)
    
