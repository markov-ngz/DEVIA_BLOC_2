from tune_model import tune_model
from dotenv import load_dotenv 
from argparse import ArgumentParser
import argparse
import json , os  
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="app.log",level=logging.INFO)

load_dotenv()

# evaluate or train ? 
parser = ArgumentParser( description="Train or Evaluate a model on blue score ")
parser.add_argument("--only_evaluate",action=argparse.BooleanOptionalAction,help="Choose if you wish to train the model or only evaluate it")
parser.add_argument("--epochs",default=0,required=False,type=int,help="Choose the number of epochs you wish to train the model on")

only_eval_model = parser.parse_args().only_evaluate
if only_eval_model != True :
      only_eval_model = False
epochs = parser.parse_args().epochs


if not isinstance(only_eval_model, bool) : 
      msg = "Command line argument : --only_evaluate must be of type bool"
      logger.error(msg)
      raise TypeError(msg)
if not isinstance(epochs, int) : 
      msg = "Command line argument : --epochs must be of type int"
      raise TypeError(msg)

if not only_eval_model and epochs <= 0 : 
      msg = "If you wish to train the model please fill the --epochs argument with the number of epochs you wish to train the model on "
      logger.error(msg)
      raise ValueError(msg)



S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH")
with open("s3_model.json")as f:
      S3_MODEL = json.load(f)
with open("s3_datasets.json")as f:
      S3_DS = json.load(f)

bleu_path = os.path.join(S3_MODEL['scoring']['folder'],S3_MODEL['scoring']['file_name'])
model_path = os.path.join(DOWNLOAD_PATH, S3_MODEL["model"])
tokenizer_path =  os.path.join(DOWNLOAD_PATH, S3_MODEL["tokenizer"])

for key in S3_DS['datasets'].keys() : 
      S3_DS['datasets'][key] =  os.path.join(DOWNLOAD_PATH, S3_DS['datasets'][key])


tune_model(
      S3_DS['datasets'],
      S3_DS["quotechar"],
      S3_DS['cols']['origin'],
      S3_DS['cols']['target'],
      model_path,
      tokenizer_path,
      bleu_path,
      S3_DS['version'],
      {"min":S3_DS['threshold']['min']},
      {"access_key":S3_ACCESS_KEY,"secret_key":S3_SECRET_KEY,"bucket_name":S3_BUCKET},
      only_eval_model=only_eval_model,
      EPOCHS=epochs
)