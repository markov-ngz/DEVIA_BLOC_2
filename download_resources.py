import boto3
from dotenv import load_dotenv
import os , json
from pprint import pprint
load_dotenv()
from datetime import datetime



def get_datetime():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def create_s3_session(access_key:str, secret_key:str)->boto3.Session:
        """
        Instantiate a session 
        """
        session = boto3.Session(
                access_key,
                secret_key)
        s3_client = session.client('s3')
        return s3_client

def download_files_from_folder(s3_client:boto3.Session,bucket_name:str,folder_name:str,download_path:str)->None:
        """
        Download a given folder content from a s3 bucket to the specified path
        **The folder will be 'copied' into the specified folder path
        """
        response =s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

        if not 'Content' in response.keys() : 
                print(f"[{get_datetime()}] Folder {folder_name} not found in bucket {bucket_name}")

        for obj in response.get('Contents', []):
                print(obj, type(obj))
                key = obj['Key']

                if not key.endswith('/'):
                        local_file_path = os.path.join(download_path, os.path.basename(key))
                        print(local_file_path)
                        s3_client.download_file(bucket_name, key, local_file_path)

def download_resources(access_key:str, secret_key:str, bucket_name:str, bucket_resources_paths:list[str], download_folder_path:str="./", os_windows=False)->None:
        """
        Download the given list of resources from a S3 bucket to the specified directory  
        """
        for key in (access_key, secret_key):
                if not isinstance(key,str):
                        raise ValueError("Variable Environments used to connect to cloud storage not found ")
                
        s3_client = create_s3_session(access_key=access_key, secret_key=secret_key)

        for resource_path in bucket_resources_paths :
                print(resource_path)
                if not resource_path.endswith('/'):
                        resource_path += '/'
                download_path = os.path.join(download_folder_path,resource_path)
                
                if os_windows : 
                        download_path = download_path.replace("/","\\")
                print(download_path)

                if not os.path.exists(download_path):
                        os.makedirs(download_path)
                else :
                        raise SystemError(f"[{get_datetime()}] Folder already exists  : {download_path}")

                download_files_from_folder(s3_client,bucket_name,resource_path,download_path)


S3_BUCKET = os.getenv('S3_BUCKET')
with open("s3_resources.json")as f:
        S3_RESOURCES = json.load(f)
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# download_resources(S3_ACCESS_KEY,
#                    S3_SECRET_KEY,
#                    bucket_name,
#                    [
#                            S3_RESOURCES['model'], 
#                            S3_RESOURCES['tokenizer']
#                    ],
#                    download_folder_path=directory_to_download_to, 
#                    os_windows=True)