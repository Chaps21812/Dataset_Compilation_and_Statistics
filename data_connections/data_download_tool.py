from aws_s3_viewer import S3Client
from pandas_statistics import PDStatistics_calculator
import os
from datetime import datetime
from tqdm import tqdm
import json
from collect_stats import collect_stats


def download_data(aws_directory:str, download_directory:str, statistics_filename:str):
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
    statistics_filename = f"{statistics_filename}_Statistics.pkl"
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(fits_path, exist_ok=True)

    client.get_data(client.directory)
    for annotT,fitsT in tqdm(client.annotation_to_fits.items(), desc="Downloading and Collecting Statistics"):
        sample_attributes = {}
        object_attributes = []
        try:
            json_content = client.download_annotation(annotT)
            fits_content = client.download_fits(fitsT)

            fits_filename = os.path.basename(fitsT)
            json_filename = os.path.basename(annotT)
            fits_local_path = os.path.join(fits_path, fits_filename)
            annot_local_path = os.path.join(annotations_path, json_filename)

            hdu = fits_content[0].header
            data = fits_content[0].data

            if "TRKMODE" in hdu.keys():
                if hdu["TRKMODE"] != 'rate':
                    continue
            
            sample_attributes, object_attributes = collect_stats(json_content, fits_content)

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

def summarize_s3_structure(aws_directory:str):
    client = S3Client(aws_directory)
    client.summarize_s3_structure()
    