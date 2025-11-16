import os
import boto3
import botocore
from typing import List, Dict

def download_s3_folders(bucket_name: str, prefix: str, local_dir: str):
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    objects = response.get("Contents", [])

    local_path = local_dir + '/' + prefix     
    print(f"Creating local directory: {local_path}")
    os.makedirs(local_path, exist_ok=True)

    for obj in objects:
        key = obj['Key']
        print(f"Processing key: {key}")
        filename = key[len(prefix):].lstrip('/')
        print(f"Derived filename: {filename}")
        local_file_path = local_path + filename
        print(f"Downloading s3://{bucket_name}/{key} to {local_file_path}")
        s3.download_file(bucket_name, key, local_file_path)

if __name__ == "__main__":
    bucket_name = "rldb"
    prefix = "mecka/folding_shirt/annotations/"
    local_dir = "/coc/cedarp-dxu345-0/datasets/egoverse/mecka_tests"
    
    download_s3_folders(bucket_name, prefix, local_dir)