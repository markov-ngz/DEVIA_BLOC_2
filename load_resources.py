import boto3
from boto3.exceptions import S3UploadFailedError
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
                key = obj['Key']

                if not key.endswith('/'):
                        local_file_path = os.path.join(download_path, os.path.basename(key))
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
                if not resource_path.endswith('/'):
                        resource_path += '/'
                download_path = os.path.join(download_folder_path,resource_path)
                
                if os_windows : 
                        download_path = download_path.replace("/","\\")

                if not os.path.exists(download_path):
                        os.makedirs(download_path)
                else :
                        raise SystemError(f"[{get_datetime()}] Folder already exists  : {download_path}")

                download_files_from_folder(s3_client,bucket_name,resource_path,download_path)

def upload_ressources(access_key:str, 
                      secret_key:str, 
                      bucket_name:str, 
                      resources_paths:list[dict]
                      )-> None : 
        """
        Upload a file or a folder's file recursively to a specified remote path.
        resources_paths format : [{"local_path":<local_path>,"remote_path":<remote_path>},...]
        """
        for key in (access_key, secret_key):
                if not isinstance(key,str):
                        raise ValueError("Variable Environments used to connect to cloud storage not found ")
        s3_client = create_s3_session(access_key=access_key, secret_key=secret_key)

        for resource_paths in resources_paths :

                if [key for key in  resource_paths.keys()] != ["local_path","remote_path"]:
                        raise ValueError(f"Error for {resource_paths} value : Dictionary key for resources paths must have keys : local_path and remote_path")
                
                # get value from dict 
                local_path = resource_paths['local_path'] 
                remote_path = resource_paths['remote_path']
                
                # check type
                if not isinstance(local_path,str) or not isinstance(remote_path,str):
                        raise TypeError(f"local_path and key remote_path key value must be of type str ")
                
                if not local_path.endswith("/"): # folder or file ? 

                         #? file uploaded to folder => takes the name of local file
                        if remote_path.endswith("/") or  remote_path.endswith("\\") : # file uploaded to folder => it takes the name of local file
                                remote_path = os.path.join(remote_path,local_path).replace("\\","/")
                        
                        try : 
                                s3_client.upload_file(local_path, bucket_name, remote_path)
                                print(f"[{get_datetime()}] File {local_path} successfully uploaded to S3 ")
                        except S3UploadFailedError as e:
                                raise e 
                else : # folders
                        for root, dirs, files in os.walk(local_path):
                                for f in files:
                                        local_path_file = os.path.join(root,f).replace("\\","/")
                                        remote_file_path = os.path.join(remote_path,root,f).replace("\\","/")
                                        try : 
                                                s3_client.upload_file(local_path_file, bucket_name, remote_file_path)
                                                print(f"[{get_datetime()}] File {local_path_file} successfully uploaded to S3 ")
                                        except S3UploadFailedError as e:
                                                raise e   




S3_BUCKET = os.getenv('S3_BUCKET')
with open("s3_model.json")as f:
        S3_MODEL = json.load(f)
with open("s3_datasets.json")as f:
        S3_DS = json.load(f)
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# download_resources(S3_ACCESS_KEY,
#                    S3_SECRET_KEY,
#                    S3_BUCKET,
#                    [
#                            S3_MODEL['model'], 
#                            S3_MODEL['tokenizer'],
#                            S3_DS['datasets']
#                    ],
#                    download_folder_path=DOWNLOAD_PATH, 
#                    os_windows=True)
