from aws_s3_viewer import S3Client
from pandas_statistics import PDStatistics_calculator
import os
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

directory = "third-party-data/PDS-RME03/CombinedAnnotations/Annotations/PDS-RME03/"
download_directory = "./data/RME03Star/"
statistics_filename = "RME03Star_Statistics.pkl"
allow_stars = True
allow_sats = False

client = S3Client(directory)
db = PDStatistics_calculator()
annotations_path = os.path.join(download_directory, "raw_annotation")
fits_path = os.path.join(download_directory, "raw_fits")
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
        hdu = fits_content[0].header
        data = fits_content[0].data
        

        hdu = fits_content[0].header
        if "TRKMODE" in hdu.keys():
            if hdu["TRKMODE"] != 'rate':
                continue

        sample_attributes["filename"] = json_content["file"]["filename"]
        sample_attributes["id_sensor"] = json_content["file"]["id_sensor"]
        sample_attributes["too_few_stars"] = json_content["too_few_stars"]
        sample_attributes["empty_image"] = json_content["empty_image"]
        sample_attributes["num_objects"] = len(json_content["objects"])
        sample_attributes["exposure"] = hdu["EXPTIME"]

        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        date_object = datetime.strptime(json_content["created"], date_format)
        sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
        sample_attributes["times"] = date_object.strftime("%H:%M:%S")

        sats = 0
        stars = 0
        for object in json_content["objects"]:
            detection_dict = {}
            detection_dict["flux"] = object['iso_flux']
            if object['class_name']=="Satellite" and allow_sats: 
                sats+=1
                detection_dict["filename"] = json_content["file"]["filename"]
                detection_dict["object_type"] = object['class_name']
                detection_dict["x_center"] = object['x_center']
                detection_dict["y_center"] = object['y_center']
                detection_dict["x_min"] = object['x_min']
                detection_dict["y_min"] = object['y_min']
                detection_dict["x_max"] = object['x_max']
                detection_dict["y_max"] = object['y_max']
                detection_dict["delta_x"] = object['x_max']-object['x_min']
                detection_dict["delta_y"] = object['y_max']-object['y_min']
                detection_dict["snr"] = object['snr']
                detection_dict["area"] = detection_dict["delta_x"]*detection_dict["delta_y"]

            if object['class_name']=="Star" and allow_stars: 
                stars+=1
                detection_dict["filename"] = json_content["file"]["filename"]
                detection_dict["object_type"] = object['class_name']
                detection_dict["x_center"] = object['x_center']
                detection_dict["y_center"] = object['y_center']
                detection_dict["x1"] = object['x1']
                detection_dict["y1"] = object['y1']
                detection_dict["x2"] = object['x2']
                detection_dict["y2"] = object['y2']
                detection_dict["delta_x"] = object['x2']-object['x1']
                detection_dict["delta_y"] = object['y2']-object['y1']
                detection_dict["length"] = np.sqrt(detection_dict["delta_x"]**2 + detection_dict["delta_y"]**2)
                detection_dict["angle"] = np.arctan2(detection_dict["delta_y"]/detection_dict["delta_x"])
            object_attributes.append(detection_dict)

        sample_attributes["num_stars"] = stars
        sample_attributes["num_sats"] = sats        

        try:
            sample_attributes["rain_condition"] = hdu["SK.WEATHER.RAINCONDITION"]
            sample_attributes["rain"] = hdu["SK.WEATHER.RAIN"]
            sample_attributes["humidity"] = hdu["SK.WEATHER.HUMIDITY"]
            sample_attributes["windspeed"] = hdu["SK.WEATHER.WINDSPEED"]
        except Exception as e:
            pass

        db.add_sample_attributes(sample_attributes)
        db.add_annotation_attributes(object_attributes)


        fits_filename = os.path.basename(fitsT)
        json_filename = os.path.basename(annotT)
        fits_local_path = os.path.join(fits_path, fits_filename)
        annot_local_path = os.path.join(annotations_path, json_filename)

        fits_content.writeto(fits_local_path, overwrite=True)
        with open(annot_local_path, "w") as f:
            json.dump(json_content, f, indent=4)
    
    except Exception as e:
        print(f"Error processing {annotT}: {e}")

    db.save(os.path.join(download_directory, statistics_filename))