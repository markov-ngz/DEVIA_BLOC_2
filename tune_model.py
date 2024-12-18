"""Evaluate or train the model"""
from transformers import create_optimizer
import os 
from dotenv import load_dotenv
import logging 
import tensorflow as tf 
from preprocess import preprocess
from evaluate_model import get_bleu_score
from load_resources import upload_ressources
import json   , logging
from datetime import datetime
load_dotenv()

logger  = logging.getLogger(__name__)
logging.basicConfig(filename="app.log",level=logging.INFO)

tf_log = tf.get_logger()
fh = logging.FileHandler("tf.log")
tf_log.addHandler(fh)


def train_model(num_epochs:int, model, ds_train, ds_valid):
    """
    Train the model with Adam Optimizer  : \n\t
    with default parameters : \n
    \t\t lr = 5e-5 , no warmup steps, weight decay = .01  \n
    Return the model trained ( TFMarianMTModel)
    """
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
               quotechar:str,
               COL_ORIGIN:str,
               COL_TARGET:str,
               MODEL_CHECKPOINT:str,
               TOKENIZER_CHECKPOINT:str, 
               BLEU_PATH:str,
               ds_version:int,
               threshold:dict,
               s3_credentials:dict, 
               only_eval_model:bool=True,
               EPOCHS:int=0)->None: 
    """
    This function will : \n
    Preprocess a dataset ( train test split + tokenize )  \n
    if evaluate_mode set tot False ( default value) : \n
        Train a model  \n
    Evaluate it with BLEU Score  \n
    Save the model into S3 bucket  \n
    WARNING :  please check your credentials before launching training ( can't check before training as session migt expire)
    """
    if only_eval_model == False and EPOCHS <= 0 :
        msg = "If only_eval_model is set to False, please set the numbers of epochs wanted to train the model on"
        logger.error(msg)
        raise ValueError(msg)
    if [key for key in  s3_credentials.keys()] != ["access_key","secret_key","bucket_name"]:
        msg = f"Error for  S3_credentials : Dictionary key for S3_credentials must have keys : access_key and secret_key and bucket_name"
        logger.error(msg)
        raise ValueError(msg)
    # check type
    if not isinstance(s3_credentials['access_key'],str) or not isinstance(s3_credentials['secret_key'],str):
        msg = f"access_key, secret_key and bucket_name key's values must be of type str "
        logger.error(msg)
        raise TypeError(msg)
    
    if not isinstance( quotechar,str):
        msg = 'quotechar argument value must be of type str'
        logger.error(msg)
        raise TypeError(msg)

    if not (isinstance(threshold['min'],(float,int))) and  not (isinstance(threshold['min'],(float,int))):
        msg = "Minimum and maximum thresholds must be of type int or float "
        logger.error(msg)
        raise TypeError(msg)
        
    resources_paths = [{"local_path":MODEL_CHECKPOINT,"remote_path":"prod/"},{"local_path":TOKENIZER_CHECKPOINT,"remote_path":"prod/"},{"local_path":BLEU_PATH,"remote_path":BLEU_PATH}]

    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] : Begginning Preprocess ")

    model, tokenizer, tf_ds = preprocess(
                                        csv_dict,
                                        COL_ORIGIN,
                                        COL_TARGET,
                                        MODEL_CHECKPOINT,
                                        TOKENIZER_CHECKPOINT,
                                        quotechar=quotechar)

    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :  Preprocess Finished ")  

    bleu_scores = {"version": ds_version}

    if not only_eval_model :

        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :  Begginning training ")  

        model = train_model(EPOCHS, 
                            model, 
                            tf_ds['train'],
                            tf_ds['valid'])

        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Training Finished ")  

        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for each dataset ")  

        # Computing Scores for different dataset 
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Train dataset ") 
        bleu_score_train = get_bleu_score(tf_ds['train'],model,tokenizer)
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Train dataset ") 
        bleu_scores['train'] = bleu_score_train
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Validation dataset ") 
        bleu_score_valid = get_bleu_score(tf_ds['valid'],model,tokenizer)
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Validation dataset ") 
        bleu_scores['valid'] = bleu_score_valid

    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Computing Bleu Score for Test dataset ") 
    bleu_score_test = get_bleu_score(tf_ds['test'],model,tokenizer)
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] :   Finished Computing Bleu Score for Test dataset, BLEU SCORE on Test: {str(bleu_score_test['score'])} ") 
    bleu_scores['test'] = bleu_score_test
    bleu_scores['version'] = ds_version

    # if only check model 
    if os.path.exists(BLEU_PATH):
        with open(BLEU_PATH, 'r') as f : 
            max_blue_score = json.load(f)

        if ( # is the model better than the current one on the same version ? 
            bleu_score_test['score'] > max_blue_score['test']['score'] and 
            ds_version == max_blue_score['version']
            ) or ( # new version of the dataset , does the model match the teams requirements ? 
            ds_version > max_blue_score['version'] and 
            bleu_score_test['score'] >= threshold['min']
            ): 
            
            with open(BLEU_PATH, 'w') as f:
                # write dict in to file
                json.dump(bleu_scores, f, indent=4)
                # save model
            model.save_pretrained(MODEL_CHECKPOINT)
            # save tokenizer
            tokenizer.save_pretrained(TOKENIZER_CHECKPOINT)
            
            upload_ressources(s3_credentials['access_key'],
                            s3_credentials['secret_key'], 
                            s3_credentials['bucket_name'],
                            resources_paths )
        elif bleu_score_test['score'] <= threshold['min']  : # model do not match teams requirements
            msg = f" Model test score  on bleu metric : {bleu_score_test['score']} is not matching the requirements. Bleu score must be between [{threshold['min']}:{threshold['max']}]"
            logger.error(msg)
            raise Exception(msg) 
        else : # model is just not the best 
            msg = f"""[{datetime.now().strftime('%H:%M:%S')}] : \n  
                Versions : dataset version : {ds_version} | bleu_score_version : {max_blue_score['version']} \n
                Bleu Scores : computed : {bleu_score_test['score']} | current : { max_blue_score['test']['score']} \n
                Message : Bleu Score too low or Invalid Versions 
                Actions : Model not loaded into production
            """
            logger.error(msg)
            raise Exception(msg)
    else:

        with open(BLEU_PATH, 'w') as f:
            json.dump(bleu_scores, f, indent=4)
        model.save_pretrained(MODEL_CHECKPOINT)
        # save tokenizer
        tokenizer.save_pretrained(TOKENIZER_CHECKPOINT)
        # Ecrire le dictionnaire dans le fichier

        upload_ressources(s3_credentials['access_key'],
                    s3_credentials['secret_key'], 
                    s3_credentials['bucket_name'],
                    resources_paths )
            



