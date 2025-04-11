from pandas_statistics import file_path_loader
from datetime import datetime
from tqdm import tqdm
import numpy as np

#Enter in the parameters you wish to download
dataset_directory = "./data/TakoTruckSatellite/"

local_files = file_path_loader(dataset_directory)


local_files.new_db()

for annotT,fitsT in tqdm(local_files.annotation_to_fits.items(), desc="Recalculating Statistics"):
    sample_attributes = {}
    object_attributes = []
    try:
        json_content = local_files.open_json(annotT)
        fits_content = local_files.open_fits(fitsT)
        hdu = fits_content[0].header
        data = fits_content[0].data
        
        x_res = hdu["NAXIS2"]
        y_res = hdu["NAXIS1"]

        hdu = fits_content[0].header
        if "TRKMODE" in hdu.keys():
            if hdu["TRKMODE"] != 'rate':
                continue

        sample_attributes["filename"] = json_content["file"]["filename"]
        sample_attributes["json_path"] = annotT
        sample_attributes["fits_path"] = fitsT
        sample_attributes["id_sensor"] = json_content["file"]["id_sensor"]
        # sample_attributes["too_few_stars"] = json_content["too_few_stars"]
        # sample_attributes["empty_image"] = json_content["empty_image"]
        sample_attributes["num_objects"] = len(json_content["objects"])
        sample_attributes["exposure"] = hdu["EXPTIME"]
        sample_attributes["std_intensity"] = np.std(data)
        sample_attributes["median_intensity"] = np.median(data)

        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        date_object = datetime.strptime(json_content["created"], date_format)
        sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
        sample_attributes["times"] = date_object.strftime("%H:%M:%S")

        sats = 0
        stars = 0
        for object in json_content["objects"]:
            detection_dict = {}
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

        local_files.statistics_file.add_sample_attributes(sample_attributes)
        local_files.statistics_file.add_annotation_attributes(object_attributes)

    except Exception as e:
        print(f"Error processing {annotT}: {e}")

    local_files.save_db()