from astropy.io import fits
from datetime import datetime
import numpy as np
from math import atan2
from plots import plot_image_with_line, z_scale_image


def directed_circular_stats(angles_deg:list) -> tuple[float, float, float]:
    """
    Compute the mean and spread of a list of angles using circular statistics.

    Args:
        angles_deg (list or np.ndarray): List of angles in degrees.

    Returns:
        mean_angle_deg (float): Mean angle in degrees (0 to 360)
        spread (float): Circular standard deviation (in radians)
        R (float): Resultant vector length (0 to 1), inversely proportional to spread
    """
    angles_rad = np.deg2rad(angles_deg)
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)

    mean_sin = np.mean(sin_vals)
    mean_cos = np.mean(cos_vals)

    # Mean angle in radians
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360

    # Resultant vector length
    R = np.hypot(mean_cos, mean_sin)

    # Circular standard deviation
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf

    return mean_angle_deg, circular_std, R

def circular_stats(angles_deg:list) -> tuple[float, float, float]:
    """
    Compute circular mean and spread of undirected angles (mod 180°).
    
    Args:
        angles_deg (list or np.ndarray): List of angles in degrees

    Returns:
        mean_angle_deg (float): Mean orientation (0-180°)
        circular_std (float): Circular standard deviation (radians)
        R (float): Resultant vector length (0 to 1), inverse of spread
    """
    angles_rad = np.deg2rad(angles_deg)
    doubled_angles = 2 * angles_rad

    sin_vals = np.sin(doubled_angles)
    cos_vals = np.cos(doubled_angles)

    mean_sin = np.mean(sin_vals)
    mean_cos = np.mean(cos_vals)

    mean_angle_2rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_rad = mean_angle_2rad / 2
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 180

    R = np.hypot(mean_cos, mean_sin)
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf

    return mean_angle_deg, circular_std, R

def collect_stats(json_content:dict, fits_content:fits, padding:int=20) -> tuple[dict,dict]:
    sample_attributes = {}
    object_attributes = []
    padding = padding

    hdu = fits_content[0].header
    data = fits_content[0].data

    x_res = hdu["NAXIS1"]
    y_res = hdu["NAXIS2"]

    sample_attributes["filename"] = json_content["file"]["filename"]
    sample_attributes["id_sensor"] = json_content["file"]["id_sensor"]
    # sample_attributes["too_few_stars"] = json_content["too_few_stars"]
    # sample_attributes["empty_image"] = json_content["empty_image"]
    sample_attributes["num_objects"] = len(json_content["objects"])
    sample_attributes["exposure"] = hdu["EXPTIME"]
    sample_attributes["std_intensity"] = np.std(data)
    sample_attributes["median_intensity"] = np.median(data)
    
    if sample_attributes["median_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10
    if sample_attributes["std_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10

    date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    date_object = datetime.strptime(json_content["created"], date_format)
    sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
    sample_attributes["times"] = date_object.strftime("%H:%M:%S")

    sats = 0
    stars = 0
    streak_angles = []
    streak_lengths = []
    for object in json_content["objects"]:
        detection_dict = {}
        detection_dict["flux"] = object['iso_flux']

        x_cord= object["x_center"]*x_res
        y_cord= object["y_center"]*y_res
        signal = data[int(y_cord), int(x_cord)]

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
            # detection_dict["snr"] = object['snr']
            detection_dict["area"] = detection_dict["delta_x"]*detection_dict["delta_y"]

            x_max = max(detection_dict["x_min"], detection_dict["x_max"])*x_res
            x_min = min(detection_dict["x_min"], detection_dict["x_max"])*x_res
            y_max = max(detection_dict["y_min"], detection_dict["y_max"])*y_res
            y_min = min(detection_dict["y_min"], detection_dict["y_max"])*y_res

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
            detection_dict["delta_x"] = (object['x2']-object['x1'])*x_res
            detection_dict["delta_y"] = (object['y2']-object['y1'])*y_res
            detection_dict["length"] = np.sqrt(detection_dict["delta_x"]**2 + detection_dict["delta_y"]**2)
            detection_dict["angle"] = atan2(detection_dict["delta_y"], detection_dict["delta_x"])*180/np.pi

            x_max = max(detection_dict["x1"], detection_dict["x2"])*x_res
            x_min = min(detection_dict["x1"], detection_dict["x2"])*x_res
            y_max = max(detection_dict["y1"], detection_dict["y2"])*y_res
            y_min = min(detection_dict["y1"], detection_dict["y2"])*y_res

            streak_angles.append(detection_dict["angle"])
            streak_lengths.append(detection_dict["length"])    

        y_start = max(0, y_min - padding)
        y_end   = min(data.shape[0], y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(data.shape[1], x_max + padding)

        window = data[int(y_start):int(y_end),int(x_start):int(x_end)]
        local_minimum = np.min(window)
        local_median = np.median(window)
        local_maximum = np.max(window)
        local_std = np.std(window)
        
        detection_dict["local_prominence"] = (signal-local_minimum)/(local_std+1*10**-5)
        detection_dict["local_snr"] = (signal-local_median)/(local_std+1*10**-5)
        detection_dict["max_signal_diff"] = (local_maximum-signal)/(local_maximum-local_minimum+1*10**-5)
        detection_dict["global_snr"] = (signal-local_median)/sample_attributes["std_intensity"]
        
        object_attributes.append(detection_dict)

    if len(streak_angles) > 0: 
        mean_angle_deg, circular_std, R = directed_circular_stats(streak_angles)
        sample_attributes["streak_direction_std"] = circular_std
        sample_attributes["streak_direction_mean"] = mean_angle_deg
        sample_attributes["streak_length_mean"] = np.mean(streak_lengths)
        sample_attributes["streak_length_std"] = np.std(streak_lengths)
    else:
        sample_attributes["streak_direction_std"] = 0
        sample_attributes["streak_direction_mean"] = 0
        sample_attributes["streak_length_mean"] = 0
        sample_attributes["streak_length_std"] = 0

    sample_attributes["num_stars"] = stars
    sample_attributes["num_sats"] = sats        

    try:
        sample_attributes["rain_condition"] = hdu["SK.WEATHER.RAINCONDITION"]
        sample_attributes["rain"] = hdu["SK.WEATHER.RAIN"]
        sample_attributes["humidity"] = hdu["SK.WEATHER.HUMIDITY"]
        sample_attributes["windspeed"] = hdu["SK.WEATHER.WINDSPEED"]
    except Exception as e:
        pass


    return sample_attributes, object_attributes



def collect_satsim_stats(json_content:dict, fits_content:fits, padding:int=20) -> tuple[dict,dict]:
    sample_attributes = {}
    object_attributes = []
    padding = padding

    hdu = fits_content[0].header
    data = fits_content[0].data

    x_res = hdu["NAXIS1"]
    y_res = hdu["NAXIS2"]

    json_content = json_content["data"]

    sample_attributes["filename"] = json_content["file"]["filename"]
    sample_attributes["num_objects"] = len(json_content["objects"])
    sample_attributes["exposure"] = hdu["EXPTIME"]
    sample_attributes["std_intensity"] = np.std(data)
    sample_attributes["median_intensity"] = np.median(data)
    
    if sample_attributes["median_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10
    if sample_attributes["std_intensity"] == 0:
        sample_attributes["median_intensity"] = 1e-10

    date_format = "%Y-%m-%dT%H-%M-%S.%f"
    date_object = datetime.strptime(json_content["file"]["dirname"], date_format)
    sample_attributes["dates"] = date_object.strftime("%Y-%m-%d")
    sample_attributes["times"] = date_object.strftime("%H:%M:%S")

    sats = 0
    stars = 0
    streak_angles = []
    streak_lengths = []
    for object in json_content["objects"]:
        detection_dict = {}

        x_cord= object["x_center"]*x_res
        y_cord= object["y_center"]*y_res
        if x_cord < 0 or x_cord > data.shape[1] or y_cord < 0 or y_cord > data.shape[0]:
            continue
        signal = data[int(y_cord), int(x_cord)]


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
            detection_dict["delta_x"] = object['bbox_width']
            detection_dict["delta_y"] = object['bbox_height']
            detection_dict["magnitude"] = object['magnitude']
            detection_dict["area"] = detection_dict["delta_x"]*detection_dict["delta_y"]

            x_max = max(detection_dict["x_min"], detection_dict["x_max"])*x_res
            x_min = min(detection_dict["x_min"], detection_dict["x_max"])*x_res
            y_max = max(detection_dict["y_min"], detection_dict["y_max"])*y_res
            y_min = min(detection_dict["y_min"], detection_dict["y_max"])*y_res

        if object['class_name']=="Star": 
            stars+=1
            detection_dict["filename"] = json_content["file"]["filename"]
            detection_dict["object_type"] = object['class_name']
            detection_dict["x1"] = object['y_start']
            detection_dict["y1"] = object['y_start']
            detection_dict["x2"] = object['x_end']
            detection_dict["y2"] = object['y_end']
            detection_dict["x_center"] = object['x_center']
            detection_dict["y_center"] = object['y_center']
            detection_dict["x_min"] = object['x_min']
            detection_dict["y_min"] = object['y_min']
            detection_dict["x_max"] = object['x_max']
            detection_dict["y_max"] = object['y_max']
            detection_dict["delta_x"] = (detection_dict['x2']-detection_dict['x1'])*x_res
            detection_dict["delta_y"] = (detection_dict['y2']-detection_dict['y1'])*y_res
            detection_dict["length"] = np.sqrt(detection_dict["delta_x"]**2 + detection_dict["delta_y"]**2)
            detection_dict["angle"] = atan2(detection_dict["delta_y"], detection_dict["delta_x"])*180/np.pi
            detection_dict["magnitude"] = object['magnitude']

            x_max = max(detection_dict["x1"], detection_dict["x2"])*x_res
            x_min = min(detection_dict["x1"], detection_dict["x2"])*x_res
            y_max = max(detection_dict["y1"], detection_dict["y2"])*y_res
            y_min = min(detection_dict["y1"], detection_dict["y2"])*y_res

            streak_angles.append(detection_dict["angle"])
            streak_lengths.append(detection_dict["length"])    

        y_start = max(0, y_min - padding)
        y_end   = min(data.shape[0], y_max + padding)
        x_start = max(0, x_min - padding)
        x_end   = min(data.shape[1], x_max + padding)


        window = data[int(y_start):int(y_end),int(x_start):int(x_end)]
        local_minimum = np.min(window)
        local_median = np.median(window)
        local_maximum = np.max(window)
        local_std = np.std(window)
        
        detection_dict["local_prominence"] = (signal-local_minimum)/(local_std+1*10**-5)
        detection_dict["local_snr"] = (signal-local_median)/(local_std+1*10**-5)
        detection_dict["max_signal_diff"] = (local_maximum-signal)/(local_maximum-local_minimum+1*10**-5)
        detection_dict["global_snr"] = (signal-local_median)/sample_attributes["std_intensity"]
        
        object_attributes.append(detection_dict)

    if len(streak_angles) > 0: 
        mean_angle_deg, circular_std, R = directed_circular_stats(streak_angles)
        sample_attributes["streak_direction_std"] = circular_std
        sample_attributes["streak_direction_mean"] = mean_angle_deg
        sample_attributes["streak_length_mean"] = np.mean(streak_lengths)
        sample_attributes["streak_length_std"] = np.std(streak_lengths)
    else:
        sample_attributes["streak_direction_std"] = 0
        sample_attributes["streak_direction_mean"] = 0
        sample_attributes["streak_length_mean"] = 0
        sample_attributes["streak_length_std"] = 0

    sample_attributes["num_stars"] = stars
    sample_attributes["num_sats"] = sats        

    try:
        sample_attributes["rain_condition"] = hdu["SK.WEATHER.RAINCONDITION"]
        sample_attributes["rain"] = hdu["SK.WEATHER.RAIN"]
        sample_attributes["humidity"] = hdu["SK.WEATHER.HUMIDITY"]
        sample_attributes["windspeed"] = hdu["SK.WEATHER.WINDSPEED"]
    except Exception as e:
        pass


    return sample_attributes, object_attributes