import boto3
from statistics_obj import Statistics_obj
import json
import io
from astropy.io import fits
import os
from datetime import datetime

attributes = ["too_few_stars", "id_sensor", "num_objects", "empty_image", "object_type", "x_center", "y_center", "flux", "num_sats", "num_stars", "exposure", "rain_condition","rain","humidity","windspeed", "dates","times" ]
counters = [[] for i in attributes]

for i,annot in enumerate(annotations):
    try:
        response = client.get_object(Bucket=bucket, Key=annot)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        
        filename = json_content["file"]["filename"]

        sensor_name = json_content["file"]["id_sensor"]
        too_few_stars = json_content["too_few_stars"]
        is_empty = json_content["empty_image"]
        num_objects = len(json_content["objects"])
        exposure = json_content["exposure"]
        created_time = json_content["created"]

        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        date_object = datetime.strptime(created_time, date_format)

        counters[0].append(too_few_stars)
        counters[1].append(sensor_name)
        counters[2].append(num_objects)
        counters[3].append(is_empty)
        counters[10].append(exposure)
        counters[15].append(date_object.strftime("%Y-%m-%d"))
        counters[16].append(date_object.strftime("%H:%M:%S"))
        

        if num_objects >0:
            stars = 0
            sats = 0
            for object in json_content["objects"]:
                name = object['class_name']
                y_center = object['y_center']
                x_center = object['x_center']
                flux = object['iso_flux']
                if name=="Satellite": sats+=1
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

            fits_file = os.path.join(os.path.dirname(annot), "ImageFiles", filename)
            response = client.get_object(Bucket=bucket, Key=fits_file)
            fits_content = io.BytesIO(response['Body'].read())

            with fits.open(fits_content) as hdul:
                hdu = hdul[0].header
                rc = hdu["SK.WEATHER.RAINCONDITION"]
                r = hdu["SK.WEATHER.RAIN"]
                h = hdu["SK.WEATHER.HUMIDITY"]
                w = hdu["SK.WEATHER.WINDSPEED"]

                counters[11].append(rc)
                counters[12].append(r)
                counters[13].append(h)
                counters[14].append(w)
    except KeyError: pass

stats = Statistics_obj(attributes,counters)
stats.save("./data_connections/data/RMEO4_data_stats_large.pkl")