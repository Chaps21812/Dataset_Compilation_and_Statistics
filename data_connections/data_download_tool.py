from aws_s3_viewer import S3Client
from pandas_statistics import PDStatistics_calculator
import os
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

#Enter in the parameters you wish to download
aws_directory = "third-party-data/PDS-RME03/CombinedAnnotations/Annotations/PDS-RME03/"
download_directory = "./data/RME03Star/"
statistics_filename = "RME03Star"



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

            x_res = hdu["NAXIS2"]
            y_res = hdu["NAXIS1"]

            hdu = fits_content[0].header
            if "TRKMODE" in hdu.keys():
                if hdu["TRKMODE"] != 'rate':
                    continue

            sample_attributes["filename"] = json_content["file"]["filename"]
            sample_attributes["json_path"] = annot_local_path
            sample_attributes["fits_path"] = fits_local_path
            sample_attributes["id_sensor"] = json_content["file"]["id_sensor"]
            sample_attributes["too_few_stars"] = json_content["too_few_stars"]
            sample_attributes["empty_image"] = json_content["empty_image"]
            sample_attributes["num_objects"] = len(json_content["objects"])
            sample_attributes["exposure"] = hdu["EXPTIME"]
            sample_attributes["std_intensity"] = np.std(data)
            sample_attributes["median_intensity"] = np.median(data)
            if sample_attributes["median_intensity"] == 0:
                continue
            if sample_attributes["std_intensity"] == 0:
                continue


            date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
            date_object = datetime.strptime(json_content["created"], date_format)
            sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
            sample_attributes["times"] = date_object.strftime("%H:%M:%S")

            sats = 0
            stars = 0
            for object in json_content["objects"]:
                detection_dict = {}
                detection_dict["json_path"] = annot_local_path
                detection_dict["fits_path"] = fits_local_path
                detection_dict["flux"] = object['iso_flux']
                detection_dict["measured_snr"] = data[int(object['x_center']*x_res),int(object['y_center']*y_res)]/sample_attributes["std_intensity"]
                detection_dict["measured_intensity_over_median"] = data[int(object['x_center']*x_res),int(object['y_center']*y_res)]/sample_attributes["median_intensity"]

                if object['class_name']=="Satellite": 
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

                if object['class_name']=="Star": 
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
                    if detection_dict["delta_x"] == 0:  
                        detection_dict["delta_x"] = 1e-10
                    detection_dict["angle"] = np.arctan(detection_dict["delta_y"]/detection_dict["delta_x"])*180/np.pi
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

            fits_content.writeto(fits_local_path, overwrite=True)
            with open(annot_local_path, "w") as f:
                json.dump(json_content, f, indent=4)
        
            db.add_sample_attributes(sample_attributes)
            db.add_annotation_attributes(object_attributes)

        except Exception as e:
            print(f"Error processing {annotT}: {e}")

        db.save(os.path.join(download_directory, statistics_filename))