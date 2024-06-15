import logging
import pandas as pd
from datasets import Dataset , DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq , TFAutoModelForSeq2SeqLM


logger = logging.getLogger(__name__)
logging.basicConfig(filename="app.log",level=logging.INFO)

def dict_formatter(x,col_origin, col_target):
    return {"id":x['index'],"translation":{col_origin : x[col_origin], col_target : x[col_target]}}

def format_pandas_df(df : pd.DataFrame,col_origin:str,col_target:str)->pd.DataFrame : 

    df = df.rename(columns={0:col_origin,1:col_target})[[col_origin,col_target]]

    if col_origin not in df.columns and col_target not in df.columns : 
        msg = "DataFrame columns could not be renamed , ensure that your csv is headless "
        logger.error(msg)
        raise ValueError(msg)
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['translation'] = df.apply(dict_formatter,axis=1,args=(col_origin, col_target))
    df.drop(columns="index",inplace=True)
    df.drop(columns=[col_origin,col_target], inplace=True)

    return df 

def pd_to_hf_dataset(df:pd.DataFrame)-> Dataset:
    """
    From a {'train','test','valid'} dataset , 
    Return a Dataset 
    """
    # pd.DataFrame -> np.array
    array_df  = df['translation'].values

    # np.array => list => datasets.arrow_dataset.Dataset
    tds = Dataset.from_list(list(array_df))

    return tds 

def to_datasetdict(dict_datasets:dict)->DatasetDict:
    """
    From a dict of Datasets ( huggingface )
    Return DataSet dict ready to tokenize 
    """
    ds = DatasetDict()
    for key in dict_datasets.keys():
        if key not in ('train','valid','test'):
            msg = "Variable key must be equal to either train , valid or test "
            logger.error(msg)
            raise ValueError(msg)
        ds[key] = dict_datasets[key]
    return ds

def tokenize_hf_ds(hf_ds:DatasetDict,tokenizer_path:str,col_origin:str, col_target:str,column_names:list=["id","translation"], max_length:int=256)->tuple:
    """
    Return (tokenizer: MarianTokenizer, dataset_tokenized:DatasetDict)
    """
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)

    def tokenizer_func(example):

        origin=[i[col_origin] for i in example['translation']]
        target=[i[col_target] for i in example['translation']]

        return tokenizer(origin,text_target=target,max_length=max_length,truncation =True)
    
    ds_tokenized = hf_ds.map(tokenizer_func,batched=True,remove_columns=column_names)

    return tokenizer, ds_tokenized

def to_tf_dataset(hf_ds_tokenized:DatasetDict,tokenizer, model_checkpoint:str)->tuple:
    """
    From a DatasetDict already tokenized , 
    Return ( model loaded from the given checkpoint and , _dict{key:_PrefetchDataset as a tensorflow usable dataset})
    """
    model=TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,local_files_only=True)
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,return_tensors='tf')

    logger.info("Model and data collator loaded ")

    dict_tf_ds = {}
    for key in hf_ds_tokenized : 
        dict_tf_ds[key] = hf_ds_tokenized[key].to_tf_dataset(columns=['input_ids', 'attention_mask'], batch_size=32
                                                , shuffle=True, collate_fn=data_collator
                                                ,label_cols='labels')

    return model, dict_tf_ds

def preprocess(csv_paths:dict, col_origin:str,col_target:str, model_path:str,tokenizer_path:str, quotechar:str="}")->tuple:
    """
    Preprocess  .csv (pd.DataFrame->hf.DatasetDict->tf..._PrefetchDataset) \n
    Return a tuple (model ,tokenizer , tensorflow_dataset)
    """
    wrap = {}
    for key in csv_paths.keys() : 
        if key not in ('train','valid','test'):
            msg = "csv_paths argument's keys must be equal to either train , valid or test "
            logger.error(msg)
            raise ValueError(msg)
        
        pd_df = pd.read_csv(csv_paths[key],quotechar=quotechar, header=None)
        f_df = format_pandas_df(pd_df,col_origin, col_target)
        hf_ds = pd_to_hf_dataset(f_df)
        wrap[key] = hf_ds
    datasetdict = to_datasetdict(wrap)
    tokenizer , hf_ds_tokenized = tokenize_hf_ds(datasetdict,tokenizer_path,col_origin,col_target)
    model, tf_ds = to_tf_dataset(hf_ds_tokenized,tokenizer,model_path)

    return model, tokenizer, tf_ds



