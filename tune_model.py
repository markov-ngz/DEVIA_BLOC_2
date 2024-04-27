from transformers import create_optimizer
import os 
from dotenv import load_dotenv
import logging 
import tensorflow as tf 
from .preprocess import preprocess
from .evaluate_model import get_bleu_score
import json  
from datetime import datetime
load_dotenv()

tf_logs = tf.get_logger()
fh = logging.FileHandler("tf.log")
tf_logs.addHandler(fh)


def train_model(num_epochs:int, model, ds_train, ds_valid):

    num_train_steps = len(ds_train) * num_epochs

    optimizer, schedule = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
        )
    model.compile(optimizer=optimizer)

    model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=num_epochs,
    )

    return model 

def tune_model(csv_dict:dict,
               COL_ORIGIN:str,
               COL_TARGET:str,
               MODEL_CHECKPOINT:str,
               TOKENIZER_CHECKPOINT:str, 
               EPOCHS:str, 
               BLEU_PATH:str)->None: 
    """
    This function will : 
    Preprocess a dataset ( train test split + tokenize ) 
    Train a model 
    Evaluate it with BLEU Score 
    Save the model 
    """

    print(f"[{datetime.now().strftime('%H:%M:%S')}] : Begginning Preprocess ")

    model, tokenizer, tf_ds = preprocess(csv_dict,COL_ORIGIN,COL_TARGET,MODEL_CHECKPOINT,TOKENIZER_CHECKPOINT)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] :  Preprocess Finished ")  

    print(f"[{datetime.now().strftime('%H:%M:%S')}] :  Begginning training ")  

    model = train_model(EPOCHS, model, tf_ds['train'], tf_ds['valid'])

    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Training Finished ")  

    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for each dataset ")  

    # Computing Scores for different dataset 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Train dataset ") 
    bleu_score_train = get_bleu_score(tf_ds['train'],model,tokenizer)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Train dataset ") 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Validation dataset ") 
    bleu_score_valid = get_bleu_score(tf_ds['valid'],model,tokenizer)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Validation dataset ") 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Test dataset ") 
    bleu_score_test = get_bleu_score(tf_ds['test'],model,tokenizer)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Test dataset ") 

    bleu_scores  = {"train":bleu_score_train, "valid":bleu_score_valid, "test":bleu_score_test}

    # if only check model 
    if os.path.exists(BLEU_PATH):
        with open(BLEU_PATH, 'r') as f : 
            max_blue_score = json.load(f)

        if bleu_score_test['score'] > max_blue_score['test']['score'] : 
            with open(BLEU_PATH, 'w') as f:
                # write dict in to file
                json.dump(bleu_scores, f, indent=4)
                # save model
            model.save_pretrained(MODEL_CHECKPOINT)
            # save tokenizer
            tokenizer.save_pretrained(TOKENIZER_CHECKPOINT)
    else:
        with open(BLEU_PATH, 'w') as f:
            json.dump(bleu_scores, f, indent=4)
        model.save_pretrained(MODEL_CHECKPOINT)
        # save tokenizer
        tokenizer.save_pretrained(TOKENIZER_CHECKPOINT)
        # Ecrire le dictionnaire dans le fichier
            



