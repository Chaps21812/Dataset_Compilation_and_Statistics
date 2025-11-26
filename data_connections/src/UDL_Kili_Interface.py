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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone

KILI_CONVERT_DIR = "/data/Dataset_Compilation_and_Statistics/data_connections/UDL_KILI_Upload"

class KILIConverter():
    def __init__(self, num_workers=20):
        self.date_ranges = {}
        self.current_date_ranges = {}
        self.downloaded = set()
        self.last_date = ""
        self.num_workers = num_workers
        self.log = {}
        
        if not os.path.exists(os.path.join(KILI_CONVERT_DIR,"log.json")):
            with open(os.path.join(KILI_CONVERT_DIR,"log.json"),'w') as f:
                json.dump(self.log,f,indent=4)
        else:
            with open(os.path.join(KILI_CONVERT_DIR,"log.json"),'r') as f:
                self.log = json.load(f)

        if not os.path.exists(os.path.join(KILI_CONVERT_DIR,"date_ranges.json")):
            with open(os.path.join(KILI_CONVERT_DIR,"date_ranges.json"),'w') as f:
                json.dump(self.date_ranges,f,indent=4)
        else:
            with open(os.path.join(KILI_CONVERT_DIR,"date_ranges.json"),'r') as f:
                self.date_ranges = json.load(f)

        if not os.path.exists(os.path.join(KILI_CONVERT_DIR,"downloaded.json")):
            with open(os.path.join(KILI_CONVERT_DIR,"downloaded.json"),'w') as f:
                json.dump(list(self.downloaded),f,indent=4)
        else:
            with open(os.path.join(KILI_CONVERT_DIR,"downloaded.json"),'r') as f:
                self.downloaded = set(json.load(f))

        if not os.path.exists(os.path.join(KILI_CONVERT_DIR,"last_date.json")):
            with open(os.path.join(KILI_CONVERT_DIR,"last_date.json"),'w') as f:
                now = datetime.now().isoformat()
                json.dump({"last_date":now},f,indent=4)
                self.last_date = now
        else:
            with open(os.path.join(KILI_CONVERT_DIR,"last_date.json"),'r') as f:
                self.last_date = datetime.fromisoformat(json.load(f)["last_date"].replace("Z", "+00:00"))


        # create the Kili client using the US cloud as the endpoint
        self.kili = Kili(
            api_key=KILI_KEY_ENV_NAME,  
            api_endpoint="https://cloud.eastus.kili-technology.com/api/label/v2/graphql"
        )
        self._set_kili_attributes()

    def _set_kili_attributes(self):
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

    def write_log_status(self):
        with open(os.path.join(KILI_CONVERT_DIR,"log.json"),'w') as f:
            json.dump(self.log,f,indent=4)

    def prune_excess_dates(self):
        for date_range, values in self.current_date_ranges:
            end_date = datetime.fromisoformat(date_range.split("@")[-1].replace("Z","+00:00").replace("%3A",":"))
            if end_date + timedelta(moths=1) < end_date:
                self.current_date_ranges.pop(date_range)

    def _save_date_ranges(self):
        with open(os.path.join(KILI_CONVERT_DIR,"date_ranges.json"),'w') as f:
            json.dump(self.date_ranges,f,indent=4)

    def _save_downloaded_collects(self):
        with open(os.path.join(KILI_CONVERT_DIR,"downloaded.json"),'w') as f:
            json.dump(list(self.downloaded),f,indent=4)

    def _save_last_date(self):
        with open(os.path.join(KILI_CONVERT_DIR,"last_date.json"),'w') as f:
            now = self.last_date.isoformat()
            json.dump({"last_date":now},f,indent=4)
            self.last_date = now

    def _load_date_ranges(self):
        with open(os.path.join(KILI_CONVERT_DIR,"date_ranges.json"),'r') as f:
            self.date_ranges = json.load(f)

    def _load_downloaded_collects(self):
        with open(os.path.join(KILI_CONVERT_DIR,"downloaded.json"),'r') as f:
            self.downloaded = set(json.load(f))

    def _write_last_query(self, date=None):
        if date is None:
            date = datetime.now().isoformat()
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

        current_time = datetime.now().astimezone(timezone.utc)
        start_time = self.last_date
        end_time = start_time+timedelta(hours=6)
        self.log[current_time.isoformat().replace("+00:00", "Z").replace(":","%3A")] = {"Query":"In Progress", "Download": "Not Started"}
        self.write_log_status()
        
        collect_set = set()
        date_range_dict = {}
        while start_time < current_time:
            formatted_start = start_time.isoformat().replace("+00:00", "Z").replace(":","%3A")
            formatted_end = end_time.isoformat().replace("+00:00", "Z").replace(":","%3A")
            
            url = f"https://unifieddatalibrary.com/udl/skyimagery?expStartTime={formatted_start}..{formatted_end}&createdBy=system.machina"
            # url = f'https://unifieddatalibrary.com/udl/skyimagery?expStartTime=%3E{temp_date.isoformat().replace("+00:00", "Z").replace(":","%3A")+"Z"}&createdBy=system.machina'
            headers = {
                "Authorization": UNIQUE_UDLKEY,
                "accept": "application/json"
            }
            date_range_set = set()
            with requests.get(url,headers=headers, stream=True) as response:
                data = response.json()
                for image_json in data:
                    if "origSensorId" in image_json.keys(): image_json["idSensor"] = image_json["origSensorId"]
                    if "idSensor" in image_json.keys() and image_json["idSensor"] not in UDL_SENSOR_TO_KILI_ID.keys(): continue
                    if image_json["imageSetId"] in self.downloaded: continue
                    collect_set.add(image_json["imageSetId"])
                    date_range_set.add(image_json["imageSetId"])
            
            date_range_dict[f"{formatted_start}@{formatted_end}"] = [group_id for group_id in date_range_set if group_id not in self.downloaded]
            self.log[f"{formatted_start}@{formatted_end}"] = {"Query":"Completed", "Download": "Not Started"}
            self.write_log_status()
            start_time = start_time+timedelta(hours=6)
            end_time = end_time+timedelta(hours=6)
            end_time = min(end_time,current_time)

        self.last_date = end_time
        self.date_ranges = self.date_ranges | date_range_dict
        self.current_date_ranges = date_range_dict
        self._save_last_date()
        self._save_date_ranges()
        return date_range_dict

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
        
    def _download_UDL_upload_kili(self):

        query_start_time = datetime.now().isoformat()+"Z"
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

    def _multiprocess_download(self,collect_id):
        try:
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
            print(f"Uploaded: {collect_id}")
            os.remove(mp4_path)
            return True, collect_id
        except Exception as e:
            print(e)
            return False, None

    def download_collects(self):
        date_collect_ranges = self._query_recently_uploaded()
        for date_range,collect_id_list in date_collect_ranges.items():
            self.log[date_range] = {"Query":"Completed", "Download": "In Progress"}
            self.write_log_status()
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._multiprocess_download, collect_id): collect_id for collect_id in collect_id_list}
                for future in as_completed(futures):
                    completed, collect_id = future.result()
                    if completed:
                        self.downloaded.add(collect_id)
                    self._save_downloaded_collects()
            self.log[date_range] = {"Query":"Completed", "Download": "Completed"}
            self.write_log_status()

    def download_undownloaded_collects(self):
        self._load_date_ranges()
        self._load_downloaded_collects()
        for date_range,collect_id_list in self.date_ranges.items():
            self.log[date_range] = {"Query":"Completed", "Download": "In Progress"}
            self.write_log_status()
            undownloaded_collects = [cid for cid in collect_id_list if cid not in self.downloaded]
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._multiprocess_download, collect_id): collect_id for collect_id in undownloaded_collects}
                for future in as_completed(futures):
                    completed, collect_id = future.result()
                    if completed:
                        self.downloaded.add(collect_id)
                    self._save_downloaded_collects()
            self.log[date_range] = {"Query":"Completed", "Download": "Completed"}
            self.write_log_status()
                        
                    



        
                



