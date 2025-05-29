from aws_s3_viewer import S3Client
from pandas_statistics import PDStatistics_calculator
import os
from tqdm import tqdm
import json
from collect_stats import collect_stats
from documentation import write_count
from collections import Counter
import numpy as np
import requests
from astropy.io import fits
import shutil
from KEY import UDL_KEY

def download_data(aws_directory:str, download_directory:str, statistics_filename:str=None, date_persistent:bool=False):
    """
    Downloads data from AWS S3 and collects statistics.
    
    Parameters:
    aws_directory (str): The S3 directory to download data from.
    download_directory (str): The local directory to save downloaded data.
    statistics_filename (str): The filename for the statistics file.
    """

    client = S3Client(aws_directory)
    db = PDStatistics_calculator()
    annotations_path = os.path.join(download_directory, "raw_annotation")
    fits_path = os.path.join(download_directory, "raw_fits")
    if statistics_filename is None: statistics_filename=os.path.basename(download_directory)
    statistics_filename = f"{statistics_filename}_Statistics.pkl"
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(fits_path, exist_ok=True)

    counter = Counter()

    for annotT,fitsT in tqdm(client.annotation_to_fits.items(), desc="Downloading and Collecting Statistics"):
    # for annotT,fitsT in client.annotation_to_fits.items():
        sample_attributes = {}
        object_attributes = []
        try:
            json_content = client.download_annotation(annotT)
            fits_content = client.download_fits(fitsT)

            random_int = np.random.randint(0,9223372036854775806)
            fits_filename = str(random_int)+".fits"
            json_filename = str(random_int)+".json"
            # fits_filename = os.path.basename(fitsT)
            # json_filename = os.path.basename(annotT)

            hdu = fits_content[0].header

            if "TRKMODE" in hdu.keys():
                if hdu["TRKMODE"] != 'rate':
                    continue #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
            
            sample_attributes, object_attributes = collect_stats(json_content, fits_content)
            counter.update([sample_attributes["dates"]])

            # os.makedirs(os.path.join(fits_path,sample_attributes["dates"]), exist_ok=True)
            # os.makedirs(os.path.join(annotations_path,sample_attributes["dates"]), exist_ok=True)
            # fits_local_path = os.path.join(fits_path, sample_attributes["dates"])
            # annot_local_path = os.path.join(annotations_path, sample_attributes["dates"])

            fits_local_path = os.path.join(fits_path, fits_filename)
            annot_local_path = os.path.join(annotations_path, json_filename)

            sample_attributes["json_path"] = annot_local_path
            sample_attributes["fits_path"] = fits_local_path
            for dictionary in object_attributes:
                dictionary["json_path"] = annot_local_path
                dictionary["fits_path"] = fits_local_path

            fits_content.writeto(fits_local_path, overwrite=True)
            with open(annot_local_path, "w") as f:
                json.dump(json_content, f, indent=4)
        
            db.add_sample_attributes(sample_attributes)
            db.add_annotation_attributes(object_attributes)

        except Exception as e:
            print(f"Error processing {annotT}: {e}")

        db.save(os.path.join(download_directory, statistics_filename))
    write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)

def summarize_s3_structure(aws_directory:str):
    client = S3Client(aws_directory)
    client.summarize_s3_structure()

def download_UDL_data(aws_directory:str, download_directory:str, Authorization_key:str=UDL_KEY):
    """
    Downloads data from AWS S3 and collects statistics.
    
    Parameters:
    aws_directory (str): The S3 directory to download data from.
    download_directory (str): The local directory to save downloaded data.
    statistics_filename (str): The filename for the statistics file.
    """

    client = S3Client(aws_directory)
    db = PDStatistics_calculator()
    db
    annotations_path = os.path.join(download_directory, "raw_annotation")
    fits_path = os.path.join(download_directory, "raw_fits")
    statistics_filename = f"{os.path.basename(download_directory)}_Statistics.pkl"
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(fits_path, exist_ok=True)

    counter = Counter()
    
    for annot_name,annot_path in tqdm(client.unmatched_annotations.items(), desc="Downloading and Collecting Statistics"):
    # for annotT,fitsT in client.annotation_to_fits.items():
        sample_attributes = {}
        object_attributes = []
        try:
            json_content = client.download_annotation(annot_path)
            url = f"https://unifieddatalibrary.com/udl/skyimagery/getFile/{json_content["image_id"]}"
            headers = {
                "Authorization": f"Basic {Authorization_key}",
                "accept": "application/octet-stream"
            }

            random_int = np.random.randint(0,9223372036854775806)
            fits_filename = str(random_int)+".fits"
            json_filename = str(random_int)+".json"

            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()  # Raises an exception for HTTP errors
                file_location=os.path.join(fits_path, fits_filename)
                with open(file_location, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            fits_content = fits.open(file_location)
            


            hdu = fits_content[0].header

            if "TRKMODE" in hdu.keys():
                if hdu["TRKMODE"] != 'rate':
                    os.remove(file_location)
                    continue #This causes discrepancy between downloaded files and the local files. Only downloads rate tracked images
            
            sample_attributes, object_attributes = collect_stats(json_content, fits_content)
            counter.update([sample_attributes["dates"]])


            fits_local_path = os.path.join(fits_path, fits_filename)
            annot_local_path = os.path.join(annotations_path, json_filename)

            sample_attributes["json_path"] = annot_local_path
            sample_attributes["fits_path"] = fits_local_path
            for dictionary in object_attributes:
                dictionary["json_path"] = annot_local_path
                dictionary["fits_path"] = fits_local_path

            with open(annot_local_path, "w") as f:
                json.dump(json_content, f, indent=4)
        
            db.add_sample_attributes(sample_attributes)
            db.add_annotation_attributes(object_attributes)

        except TypeError as e:
            print(f"Error processing {annot_path}: {e}")

        # except Exception as e:
        #     print(f"Error processing {annot_path}: {e}")

        db.save(os.path.join(download_directory, statistics_filename))
    write_count(os.path.join(download_directory, "count.txt"),len(db.annotation_attributes), len(db.sample_attributes),counter)

def summarize_UDL_data(aws_directory:str):
    client = S3Client(aws_directory)
    client.summarize_UDL_data()




if __name__ == "__main__":
    #Enter in the parameters you wish to download
    aws_directory = "third-party-data/PDS-KWAJ01/Satellite/Annotations/"
    download_directory = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset-Statistics/data/KWAJData"

    download_UDL_data(aws_directory, download_directory)