import boto3
from statistics_obj import Statistics_obj
import json
import io
from astropy.io import fits
import os
from datetime import datetime
import numpy as np
import random
from PIL import Image
from astropy.visualization import ZScaleInterval
from tqdm import tqdm

client = boto3.client('s3')
bucket = "silt-annotations"
directory = "third-party-data/PDS-RME04/"

# Local folders
os.makedirs("RME04v4_annotations", exist_ok=True)
os.makedirs("RME04v4_fits_files", exist_ok=True)
os.makedirs("RME04v4_preprocessed_images", exist_ok=True)

paginator = client.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=bucket, Prefix=directory)

annotations = []
fits_collection = []
folders = []

for page in page_iterator:
    if "Contents" in page:
        for key in page[ "Contents" ]:
            keyString = key["Key"]
            if ".fits" in keyString:
                fits_collection.append(keyString)
            elif ".json" in keyString:
                annotations.append(keyString)
            else:
                folders.append(keyString)
print(f"# of annotations: {len(annotations)}")
print(f"# of fits: {len(fits_collection)}")
print(f"# of folders: {len(folders)}")

attributes = ["too_few_stars", "id_sensor", "num_objects", "empty_image", "object_type", "x_center", "y_center", "flux", "num_sats", "num_stars", "exposure", "rain_condition","rain","humidity","windspeed", "dates","times", "target_brightness", "median_intensity", "average_intensity", "QAed", "track_mode", "target_over_median"]
counters = [[] for i in attributes]

# To randomly go through the annoations instead of the first on the list
random.shuffle(annotations)

for i, annot in enumerate(tqdm(annotations[:1000], desc="Processing annotations")):
    try:
        response = client.get_object(Bucket=bucket, Key=annot)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        filename = json_content["file"]["filename"]
        fits_file = os.path.join(os.path.dirname(annot), "ImageFiles", filename)
        response = client.get_object(Bucket=bucket, Key=fits_file)
        fits_content = io.BytesIO(response['Body'].read())

        with fits.open(fits_content) as hdul:
            image_data = hdul[0].data
            hdu = hdul[0].header
            tm = hdu["TRKMODE"]
            # Save FITS locally
            if tm == 'rate':
                fits_local_path = os.path.join("RME04v4_fits_files", filename)
                with open(fits_local_path, "wb") as f:
                    f.write(fits_content.getbuffer())
        
        # Save JSON locally
        if tm == 'rate':
            json_filename = os.path.basename(annot)
            with open(os.path.join("RME04v4_annotations", json_filename), "w") as f:
                json.dump(json_content, f, indent=4)
        
        # Z-scale normalization
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_data)

        # Prevent divide by zero
        if vmax - vmin == 0:
            image_data_zscaled = np.zeros_like(image_data)
        else:
            image_data_zscaled = (image_data - vmin) / (vmax - vmin)
            image_data_zscaled = np.nan_to_num(image_data_zscaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip and scale to 16-bit
        image_data_zscaled = np.clip(image_data_zscaled, 0, 1)
        image_data_png = (image_data_zscaled * 65535).astype(np.uint16)

        # Save as 16-bit PNG
        if tm == 'rate':
            png_filename = os.path.splitext(filename)[0] + ".png"
            Image.fromarray(image_data_png).save(os.path.join("RME04v4_preprocessed_images", png_filename))

        sensor_name = json_content["file"]["id_sensor"]
        too_few_stars = json_content["too_few_stars"]
        is_empty = json_content["empty_image"]
        num_objects = len(json_content["objects"])
        exposure = json_content["exposure"]
        created_time = json_content["created"]
        quality_analysis = json_content["approved"]
        img_height = json_content["sensor"]["height"]
        img_width = json_content["sensor"]["width"]

        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        date_object = datetime.strptime(created_time, date_format)

        counters[0].append(too_few_stars)
        counters[1].append(sensor_name)
        counters[2].append(num_objects)
        counters[3].append(is_empty)
        counters[10].append(exposure)
        counters[15].append(date_object.strftime("%Y-%m-%d"))
        counters[16].append(date_object.strftime("%H:%M:%S"))
        counters[20].append(quality_analysis)
    
        median_intensity = np.median(image_data)
        average_intensity = np.mean(image_data)
        counters[18].append(median_intensity)
        counters[19].append(average_intensity)

        if num_objects >0:
            stars = 0
            sats = 0

            for object in json_content["objects"]:
                name = object['class_name']
                y_center = object['y_center']
                x_center = object['x_center']
                flux = object['iso_flux']
                if name=="Satellite": 
                    sats+=1
                    height, width = image_data.shape

                    x = int(x_center * width)
                    y = int(y_center * height)
                    if x <= width and y <= height:
                        target_intensity = image_data[int(y_center * height), int(x_center * width)]
                        counters[17].append(target_intensity)
                        counters[22].append(target_intensity/median_intensity)

                        
                if name=="Star": stars+=1

                counters[4].append(name)
                counters[5].append(x_center)
                counters[6].append(y_center)
                counters[7].append(flux)
            counters[8].append(sats)
            counters[9].append(stars)

            direc = os.path.dirname(annot)
            filepath = f"ImageFiles/{filename}"
            # print(json_content)

            rc = hdu["SK.WEATHER.RAINCONDITION"]
            r = hdu["SK.WEATHER.RAIN"]
            h = hdu["SK.WEATHER.HUMIDITY"]
            w = hdu["SK.WEATHER.WINDSPEED"]

            counters[11].append(rc)
            counters[12].append(r)
            counters[13].append(h)
            counters[14].append(w)
            counters[21].append(tm)


    except KeyError: pass

stats = Statistics_obj(attributes,counters)
stats.save("rme04v4_med.pkl")