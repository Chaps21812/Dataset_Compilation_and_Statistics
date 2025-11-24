from ratelimit import limits, sleep_and_retry
import os
import io
import requests
import json
from datetime import datetime, timedelta
from astropy.io import fits
from .UDL_KEY import UNIQUE_UDLKEY, KILI_KEY_ENV_NAME, UDL_SENSOR_TO_KILI_ID
from .HTTPUtils import retry_on_443
from kili.client import Kili
import numpy as np
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from .preprocess_functions import iqr_log, zscale, raw_file
from collections import Counter

KILI_CONVERT_DIR = "/data/Dataset_Compilation_and_Statistics/data_connections/UDL_KILI_Upload"


class KILIConverter():
    def __init__(self):
        self.current_collects = []
        if not os.path.exists(os.path.join(KILI_CONVERT_DIR,"PREV_QUERY.json")):
            self._write_last_query()

        with open(os.path.join(KILI_CONVERT_DIR,"PREV_QUERY.json"),'r') as f:
            data = json.load(f)
        self.last_date = datetime.fromisoformat(data["last_query"].replace("Z", "+00:00"))
        self.last_collect_ids = set(data["downloaded_collects"])
        # create the Kili client using the US cloud as the endpoint
        self.kili = Kili(
            api_key=KILI_KEY_ENV_NAME,  
            api_endpoint="https://cloud.eastus.kili-technology.com/api/label/v2/graphql"
        )
        self.downloaded_collects = set(data["downloaded_collects"])
        self.undownloaded_collects = set(data["undownloaded_collects"])

        metadata_properties = {
        "expStartTime": {
            "type": "date",
            "filterable": True,
            "visibleByLabeler": True,
            "visibleByReviewer": True,
        },
        }
        for key,value in UDL_SENSOR_TO_KILI_ID.items():
            self.kili.update_properties_in_project(value, metadata_properties=metadata_properties)



    def _write_last_query(self, date=None):
        if date is None:
            date = datetime.now(tz=None).isoformat()+"Z"
            temp_date = temp_date.replace(tzinfo=None)
        data = {"last_query":date,
                "last_collects":list(self.current_collects),
                "downloaded_collects":list(self.downloaded_collects),
                "undownloaded_collects":list(self.undownloaded_collects),
        }
        with open(os.path.join(KILI_CONVERT_DIR,"PREV_QUERY.json"),'w') as f:
            json.dump(data, f, indent=4)

    @sleep_and_retry
    @retry_on_443(max_retries=6, delay=1)
    @limits(calls=29, period=timedelta(minutes=1).total_seconds())
    @limits(calls=299, period=timedelta(hours=1).total_seconds())
    def _query_recently_uploaded(self) -> dict:

        current_time = datetime.now(tz=None)
        temp_date = self.last_date
        temp_date = temp_date.replace(tzinfo=None)
        queried_images = 1
        while temp_date < current_time:
            # https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E2025-11-20T00:00:00.000000Z
            url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E{temp_date.isoformat().replace("+00:00", "Z").replace(":","%3A")+"Z"}&createdBy=system.machina'
            headers = {
                "Authorization": UNIQUE_UDLKEY,
                "accept": "application/json"
            }

            newest_date = self.last_date
            oldest_date = self.last_date
            collect_to_ids = {}
            group_ids = Counter()
            group_id_desired_len = {}
            with requests.get(url,headers=headers, stream=True) as response:
                data = response.json()
                for image_json in data:
                    if "origSensorId" in image_json.keys(): image_json["idSensor"] = image_json["origSensorId"]
                    if "idSensor" in image_json.keys() and image_json["idSensor"] not in UDL_SENSOR_TO_KILI_ID.keys(): continue
                    if image_json["imageSetId"] in self.last_collect_ids: continue
                    newest_date = max(newest_date, datetime.fromisoformat(image_json["expStartTime"].replace("Z", "+00:00")))
                    oldest_date = min(oldest_date, datetime.fromisoformat(image_json["expStartTime"].replace("Z", "+00:00")))
                    image_json["expStartTime"] = oldest_date.isoformat()
                    if image_json["imageSetId"] in collect_to_ids.keys():
                        collect_to_ids[image_json["imageSetId"]][image_json["id"]] = image_json
                    else: 
                        collect_to_ids[image_json["imageSetId"]] = {image_json["id"]:image_json}
                    group_ids.update([image_json["imageSetId"]])
                    group_id_desired_len[image_json["imageSetId"]] = image_json["imageSetLength"]
            temp_date = newest_date.replace(tzinfo=None)
            queried_images = len(data)

        for collect_id,intended_len in tqdm(group_id_desired_len.items(), desc="Searching incomplete collects"):
            if group_ids[collect_id] != intended_len:
                # url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E2020-01-01T00:00:00.000000Z&imageSetId={collect_id}'
                url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E2020-01-01T00:00:00.000000Z&imageSetId={collect_id}'
                headers = {
                    "Authorization": UNIQUE_UDLKEY,
                    "accept": "application/json"
                }
                with requests.get(url,headers=headers, stream=True) as response:
                    data = response.json()
                    for image_json in data:
                        if "origSensorId" in image_json.keys(): image_json["idSensor"] = image_json["origSensorId"]
                        if "idSensor" in image_json.keys() and image_json["idSensor"] not in UDL_SENSOR_TO_KILI_ID.keys(): continue
                        if image_json["imageSetId"] in self.last_collect_ids: continue
                        newest_date = max(newest_date, datetime.fromisoformat(image_json["expStartTime"].replace("Z", "+00:00")))
                        oldest_date = min(oldest_date, datetime.fromisoformat(image_json["expStartTime"].replace("Z", "+00:00")))
                        image_json["expStartTime"] = oldest_date.isoformat()
                        if image_json["imageSetId"] in collect_to_ids.keys():
                            collect_to_ids[image_json["imageSetId"]][image_json["id"]] = image_json
                        else: 
                            collect_to_ids[image_json["imageSetId"]] = {image_json["id"]:image_json}


        self.last_date = newest_date
        return collect_to_ids

    @sleep_and_retry
    @retry_on_443(max_retries=10, delay=60)
    @limits(calls=29, period=timedelta(minutes=0.2).total_seconds())
    @limits(calls=299, period=timedelta(hours=1).total_seconds())
    def _download_FITS(self,id):
        url = f"https://unifieddatalibrary.com/udl/skyimagery/getFile/{id}"
        headers = {
            "Authorization": UNIQUE_UDLKEY,
            "accept": "application/octet-stream"
        }
        with requests.get(url, headers=headers, stream=True) as response:
            fits_content = fits.open(io.BytesIO(response.content))
            return fits_content

    @sleep_and_retry
    @retry_on_443(max_retries=10, delay=60)
    @limits(calls=29, period=timedelta(minutes=0.2).total_seconds())
    @limits(calls=299, period=timedelta(hours=1).total_seconds())
    def _query_collect(self, collect_id):
        # url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E2020-01-01T00:00:00.000000Z&imageSetId={collect_id}'
        url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E2020-01-01T00:00:00.000000Z&imageSetId={collect_id}'
        headers = {
            "Authorization": UNIQUE_UDLKEY,
            "accept": "application/json"
        }
        with requests.get(url,headers=headers, stream=True) as response:
            data = response.json()
        return data

    def _convert_collect_to_mp4(self, collect_id:str, image_list:list[np.ndarray], FPS=8) -> ImageSequenceClip:
        os.makedirs(os.path.join(KILI_CONVERT_DIR,"mp4s"), exist_ok=True)
        converted_frames = []
        for image in image_list:
            img_rgb = iqr_log(image, axis=-1)
            converted_frames.append(img_rgb)
        clip = ImageSequenceClip(converted_frames, fps=FPS)
        clip.write_videofile(os.path.join(KILI_CONVERT_DIR,"mp4s",f'{collect_id}.mp4'), codec="libx264", fps=FPS, audio=False)
        return os.path.join(KILI_CONVERT_DIR,"mp4s",f'{collect_id}.mp4')

    def _upload_to_kili(self, mp4_path:str, mp4_attributes: dict) -> None:
        self.kili.append_many_to_dataset(
                project_id = UDL_SENSOR_TO_KILI_ID[mp4_attributes["idSensor"]],
                content_array = [mp4_path],
                external_id_array = [mp4_attributes["imageSetId"]],
                json_metadata_array = [mp4_attributes]
            )
        
    def download_UDL_upload_kili(self):

        query_start_time = datetime.now(tz=None).isoformat()+"Z"
        collect_to_id = self._query_recently_uploaded()
        for collect_id in collect_to_id.keys():
            self.undownloaded_collects.add(collect_id)
        for collect_id,collect_dict in tqdm(collect_to_id.items(), desc="Downloading collect"):
            dummy_index =  list(collect_dict.keys())[0]
            fits_images = [None for i in range(collect_dict[dummy_index]["imageSetLength"])]
            common_attributes = {}
            for image_id,dictionary in collect_dict.items():
                fits_data = self._download_FITS(image_id)
                header = fits_data[0].header
                header_dict = {k:v for k,v in header.items() if k != "COMMENT"}
                data = fits_data[0].data
                fits_images[dictionary["sequenceId"]] = data

                img_attributes = header_dict | dictionary
                if not common_attributes:
                    common_attributes = common_attributes | img_attributes
                else: 
                    common_attributes = {k:v for k,v in common_attributes.items() if k in img_attributes.keys() and img_attributes[k] == v}

            no_empty_images = [x for x in fits_images if x is not None]
            if len(fits_images) != len(no_empty_images):
                common_attributes["missing_frames"] = True
            else:
                common_attributes["missing_frames"] = False
            mp4_path = self._convert_collect_to_mp4(collect_id,no_empty_images)
            try:
                self._upload_to_kili(mp4_path, common_attributes)
            except:
                continue
            self.current_collects.append(collect_id)
            self.downloaded_collects.add(collect_id)
            self.undownloaded_collects.remove(collect_id)
            self._write_last_query(date = query_start_time)
            os.remove(mp4_path)

    def download_undownloaded_collects(self):
        for collect_id in tqdm(self.undownloaded_collects):
            image_list = self._query_collect(collect_id)
            fits_images = [None for i in range(image_list[0]["imageSetLength"])]
            common_attributes = {}
            for index,dictionary in enumerate(image_list):
                fits_data = self._download_FITS(dictionary["id"])
                header = fits_data[0].header
                header_dict = {k:v for k,v in header.items() if k != "COMMENT"}
                data = fits_data[0].data
                fits_images[dictionary["sequenceId"]] = data

                img_attributes = header_dict | dictionary
                if not common_attributes:
                    common_attributes = common_attributes | img_attributes
                else: 
                    common_attributes = {k:v for k,v in common_attributes.items() if k in img_attributes.keys() and img_attributes[k] == v}

            no_empty_images = [x for x in fits_images if x is not None]
            if len(fits_images) != len(no_empty_images):
                common_attributes["missing_frames"] = True
            else:
                common_attributes["missing_frames"] = False
            mp4_path = self._convert_collect_to_mp4(collect_id,no_empty_images)
            self._upload_to_kili(mp4_path, common_attributes)
            self.current_collects.append(collect_id)
            self.downloaded_collects.add(collect_id)
            self.undownloaded_collects.remove(collect_id)
            self._write_last_query(date = self.last_date)
            os.remove(mp4_path)
       
            

