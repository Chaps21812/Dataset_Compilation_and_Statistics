import boto3
import json
import io
import os
from astropy.io import fits
from tqdm import tqdm
from collections import Counter

class S3Client:
    def __init__(self, directory:str):
        self.client = boto3.client('s3')
        self.bucket = "silt-annotations"
        self.directory = directory

        self.folders = []
        self.annotation_to_fits = {}
        
        self.fits_filename_to_path = {}
        self.annotation_filename_to_path = {}
        
        self.basenames = Counter()
        self.dirnames = Counter()

        self.errors = 0


    def get_data(self, directory):
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=directory)

        for i, page in enumerate(tqdm(page_iterator, desc="Loading file locations")):
            if "Contents" in page:
                for key in page[ "Contents" ]:
                    keyString = key["Key"]
                    dirname = os.path.dirname(keyString)
                    basename = os.path.basename(keyString)
                    if ".fits" in keyString:
                        self.fits_filename_to_path[basename] = keyString
                        self.basenames.update([basename])
                    elif ".json" in keyString:
                        self.annotation_filename_to_path[basename] = keyString
                        self.dirnames.update([dirname])
                        self.basenames.update([basename])
                    else:
                        self.folders.append(keyString)
        self._create_annotation_mapping()
        print(f"FITS:{len(self.fits_filename_to_path)}, Annot:{len(self.annotation_filename_to_path)}, Errors:{self.errors}, TotalSynced:{len(self.annotation_to_fits)}")

    def _create_annotation_mapping(self) -> dict:
        for annotation_filename, annot_full_path in self.annotation_filename_to_path.items():
            fits_file = annotation_filename.replace(".json", ".fits")
            fits_full_path = self.fits_filename_to_path[fits_file]
            try: self.annotation_to_fits[annot_full_path] = fits_full_path
            except KeyError: self.errors += 1
        return self.annotation_to_fits

    def download_annotation(self, annotation_path:str) -> dict:
        response = self.client.get_object(Bucket=self.bucket, Key=annotation_path)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        return json_content
    
    def download_fits(self, fits_path:str) -> fits:
        response = self.client.get_object(Bucket=self.bucket, Key=fits_path)
        fits_content = io.BytesIO(response['Body'].read())
        return fits.open(fits_content)
    
    def summarize_s3_structure(self):
        print(f"# of annotations: {len(self.annotation_filename_to_path)}")
        print(f"# of fits: {len(self.fits_filename_to_path)}")
        print(f"# of folders: {len(self.folders)}")
        for dirname,count in self.dirnames.items():
            print(f"{dirname}: {count}")
        # for tuplee in self.basenames.most_common(5):
        #     print(f"Filename: {tuplee[0]}, Count: {tuplee[1]}")

    
if __name__ == "__main__":
    s3_client = S3Client("third-party-data/PDS-RME04/Satellite/Annotations/PDS-RME04/")
    s3_client.get_data(s3_client.directory)
    s3_client.summarize_s3_structure()
