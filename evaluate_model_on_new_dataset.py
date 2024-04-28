from tune_model import tune_model
from dotenv import load_dotenv 
import json , os
load_dotenv()

BLEU_PATH = "bleu_score.json"
S3_BUCKET = os.getenv('S3_BUCKET')
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH")
with open("s3_model.json")as f:
        S3_MODEL = json.load(f)
with open("s3_datasets.json")as f:
        S3_DS = json.load(f)

model_path = os.path.join(DOWNLOAD_PATH, S3_MODEL["model"])
tokenizer_path =  os.path.join(DOWNLOAD_PATH, S3_MODEL["tokenizer"])

for key in S3_DS['datasets'].keys() : 
    S3_DS['datasets'][key] =  os.path.join(DOWNLOAD_PATH, S3_DS['datasets'][key])


if not os.path.exists("bleu_score.json"):
    raise SystemError("File : bleu_score.json not found at the root directory")

tune_model(
        S3_DS['datasets'],
        S3_DS["quotechar"],
        S3_DS['cols']['origin'],
        S3_DS['cols']['target'],
        model_path,
        tokenizer_path,
        BLEU_PATH,
        S3_DS['version'],
        {"min":S3_DS['threshold']['min'],"max":S3_DS['threshold']['max']},
        {"access_key":S3_ACCESS_KEY,"secret_key":S3_SECRET_KEY},
        only_eval_model=True,
)