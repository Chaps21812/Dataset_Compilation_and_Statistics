import boto3
import json
import io
import os
from astropy.io import fits
from tqdm import tqdm
from collections import defaultdict

class S3Client:
    def __init__(self, directory:str):
        self.client = boto3.client('s3')
        self.bucket = "silt-annotations"
        self.directory = directory

        self.annotations = []
        self.fits_files = []
        self.folders = []
        self.annotation_to_fits = {}
        self.folder_file_count = {}


    def get_data(self, directory):
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=directory)

        for i, page in enumerate(tqdm(page_iterator, desc="Loading file locations")):
            if "Contents" in page:
                for key in page[ "Contents" ]:
                    keyString = key["Key"]
                    if ".fits" in keyString:
                        self.fits_files.append(keyString)
                        os.path.dirname(keyString)
                    elif ".json" in keyString:
                        self.annotations.append(keyString)
                    else:
                        self.folders.append(keyString)
        self._create_annotation_mapping()

    def _create_annotation_mapping(self) -> dict:
        for annotation_path in self.annotations:
            directory = os.path.dirname(annotation_path)
            image_directory = os.path.join(directory, "ImageFiles")
            annotation_file = annotation_path.split("/")[-1]
            fits_file = annotation_file.replace(".json", ".fits")
            fits_path = os.path.join(image_directory, fits_file)
            self.annotation_to_fits[annotation_path] = fits_path
        return self.annotation_to_fits

    def download_annotation(self, annotation_path:str) -> dict:
        response = self.client.get_object(Bucket=self.bucket, Key=annotation_path)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        return json_content
    
    def download_fits(self, fits_path:str) -> fits:
        response = self.client.get_object(Bucket=self.bucket, Key=fits_path)
        fits_content = io.BytesIO(response['Body'].read())
        return fits.open(fits_content)
    
if __name__ == "__main__":
    s3_client = S3Client("third-party-data/PDS-RME04/Satellite/Annotations/PDS-RME04/")
    s3_client.get_data(s3_client.directory)
    print(f"# of annotations: {len(s3_client.annotations)}")
    print(f"# of fits: {len(s3_client.fits_files)}")
    print(f"# of folders: {len(s3_client.folders)}")
