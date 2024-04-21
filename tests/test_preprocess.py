import pytest
from preprocess import * 
import pandas as pd
from transformers import MarianTokenizer, TFMarianMTModel
from tensorflow.python.data.ops.prefetch_op import _PrefetchDataset


def test_csv(csv_dict_path):

    pd_df = pd.read_csv(csv_dict_path["test"],quotechar="}", header=None)

    assert pd_df.shape[0] > 0


def test_dict_formatter():
    """Tests dict_formatter with various input scenarios."""
    data = {'index': 1, 'source_language': 'French', 'target_language': 'English'}
    col_origin = 'source_language'
    col_target = 'target_language'

    expected_output = {"id": 1, "translation": {col_origin: "French", col_target: "English"}}

    assert dict_formatter(data.copy(), col_origin, col_target) == expected_output

def test_format_pandas_df(sample_dataframe):
    """Tests format_pandas_df with various input scenarios."""
    df = sample_dataframe.copy()
    col_origin = 'source'
    col_target = 'target'

    expected_output = pd.DataFrame({
        "translation": [{"id":0,"translation":{"source": "Bonjour", "target": "Hello"}},{"id":1,"translation":{"source": "Hola", "target": "Hi"}}]
    })

    result_df = format_pandas_df(df.copy(), col_origin, col_target)

    # Assert expected columns and data types
    assert result_df.columns.tolist() == expected_output.columns.tolist()
    for col in expected_output.columns:
        assert result_df[col].dtype == expected_output[col].dtype

    # Test with DataFrame containing only one row
    one_row_df = df.iloc[0:1]
    expected_output_one_row = pd.DataFrame({"translation":[{"id":0,"translation":{"source": "Bonjour", "target": "Hello"}}]})
    result_df = format_pandas_df(one_row_df.copy(), col_origin, col_target)
    assert result_df.equals(expected_output_one_row)


def test_pd_to_hf_dataset(cleaned_df):
    """Tests pd_to_hf_dataset with mocked train_test_split."""

    result_ds = pd_to_hf_dataset(cleaned_df)

    assert isinstance(result_ds, Dataset)

    assert result_ds.column_names == ['id','translation'] 

def test_to_datasetdict(hf_ds):

    test_dict = to_datasetdict({"train":hf_ds})

    for key in test_dict.keys() :
        assert key in ["train",'valid',"test"]
        assert test_dict[key].column_names == ['id','translation'] 


def test_tokenize_hf_ds(model_and_tokenizer_checkpoint,hf_ds, col_origin_target):
    col_origin , col_target = col_origin_target
    _ , tokenizer_checkpoint = model_and_tokenizer_checkpoint
    ds = to_datasetdict({"test":hf_ds})
    result = tokenize_hf_ds(ds,tokenizer_checkpoint,col_origin, col_target)

    assert isinstance(result,tuple) == True 

    tokenizer, ds = result

    assert isinstance(tokenizer, MarianTokenizer) == True 
    assert isinstance(ds, DatasetDict) == True 

# def test_to_df_dataset(tokened_ds, model_and_tokenizer, model_and_tokenizer_checkpoint):

#     model , tokenizer = model_and_tokenizer
#     model_path , token_path = model_and_tokenizer_checkpoint

#     _ , ds = tokened_ds

#     print(type(tokenizer),tokenizer)
#     result =  to_tf_dataset(ds,tokenizer, model_path)

#     assert isinstance(result,tuple) == True 

#     model , dict_ds  = result 

#     assert isinstance(dict_ds,dict) == True 

#     assert [ key for key in dict_ds.keys()] == ['train','valid','test']

#     for key in  dict_ds.keys() : 
#         assert isinstance(dict_ds[key],_PrefetchDataset) == True 

#     assert isinstance(model,TFMarianMTModel) == True 

def test_preprocess(csv_dict_path,col_origin_target,model_and_tokenizer_checkpoint, quotechar):
    col_o , col_t = col_origin_target

    quotechar = quotechar
    
    model_path, token_path = model_and_tokenizer_checkpoint

    result = preprocess(csv_dict_path, col_o, col_t,model_path,token_path, quotechar)

    assert len(result) == 3 

    assert isinstance(result,tuple) == True 

    m , t , tf_ds = result 

    assert isinstance(m,TFMarianMTModel) == True 

    assert isinstance(t, MarianTokenizer) == True 

    assert isinstance(tf_ds, dict) == True 


