import pickle
import pandas as pd
import os
import json
from astropy.io import fits
from tqdm import tqdm
import numpy as np
from datetime import datetime

class PickleSerializable:
    def save(self, filename):
        """Save the object as a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load an object from a pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class PDStatistics_calculator(PickleSerializable):
    def __init__(self):
        self.sample_attributes = pd.DataFrame()
        self.annotation_attributes = pd.DataFrame()
        
    def add_sample_attributes(self, item_sample:dict):
        sample = pd.DataFrame([item_sample])
        self.sample_attributes = pd.concat([self.sample_attributes, sample], ignore_index=True)

    def add_annotation_attributes(self, annotation_sample:list):
        annotation = pd.DataFrame(annotation_sample)
        self.annotation_attributes = pd.concat([self.annotation_attributes, annotation], ignore_index=True)

class file_path_loader():
    def __init__(self, dataset_path:str):
        self.directory = dataset_path
        self.statistics_file = PDStatistics_calculator.load(os.path.join(self.directory,[f for f in os.listdir(self.directory) if f.endswith(".pkl")][0]))
        self.statistics_filename = os.path.join(self.directory,[f for f in os.listdir(self.directory) if f.endswith(".pkl")][0])
        self.annotation_path = os.path.join(self.directory, "raw_annotation")
        self.fits_file_path = os.path.join(self.directory, "raw_fits")
        self.update_annotation_to_fits()

    def update_annotation_to_fits(self):
        self.annotations = os.listdir(self.annotation_path)
        self.fits_files = os.listdir(self.fits_file_path)
        self.annotation_to_fits = {}

        for annotation in self.annotations:
            fits_file = annotation.replace(".json", ".fits")
            if fits_file not in self.fits_files:
                print(f"Warning: {fits_file} not found in {self.fits_file_path}.")
                continue
            full_annotation_path = os.path.join(self.annotation_path, annotation)
            full_fits_path = os.path.join(self.fits_file_path, fits_file)
            self.annotation_to_fits[full_annotation_path] = full_fits_path

    def open_json(self, json_path:str):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        return json_content
    
    def open_fits(self, fits_path:str):
        fits_content = fits.open(fits_path)
        return fits_content

    def new_db(self):
        self.statistics_file = PDStatistics_calculator()
        self.statistics_file.save(self.statistics_filename)
        print(f"New database created at {self.statistics_filename}")

    def save_db(self):
        self.statistics_file.save(self.statistics_filename)

    def delete_files_from_annotation(self, path_series: pd.DataFrame):
        """
        Deletes files from the filesystem given a pandas Series of file paths. AI Generated

        Args:
            path_series (pd.Series): Series of file paths (as strings).
        """
        self.statistics_file.annotation_attributes.drop(path_series.index, inplace=True)
        for path in tqdm(path_series["json_path"].dropna(), desc="Deleting files"):
            fits_path = self.annotation_to_fits[path]
            try:
                #Deleting annotation path
                if os.path.isfile(path):
                    os.remove(path)

                #Deleting Fits path
                if os.path.isfile(fits_path):
                    os.remove(fits_path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        self.save_db()

    def delete_files_from_sample(self, path_series: pd.DataFrame):
        """
        Deletes files from the filesystem given a pandas Series of file paths. AI Generated

        Args:
            path_series (pd.Series): Series of file paths (as strings).
        """
        self.statistics_file.sample_attributes.drop(path_series.index, inplace=True)
        for path in tqdm(path_series["json_path"].dropna(), desc="Deleting files"):
            fits_path = self.annotation_to_fits[path]
            try:
                #Deleting annotation path
                if os.path.isfile(path):
                    os.remove(path)

                #Deleting Fits path
                if os.path.isfile(fits_path):
                    os.remove(fits_path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        self.save_db()
    

    def recalculate_statistics(self):
        self.update_annotation_to_fits()
        self.new_db()

        for annotT,fitsT in tqdm(self.annotation_to_fits.items(), desc="Recalculating Statistics"):
            sample_attributes = {}
            object_attributes = []
            try:
                json_content = self.open_json(annotT)
                fits_content = self.open_fits(fitsT)
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
                    detection_dict["json_path"] = annotT
                    detection_dict["fits_path"] = fitsT
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
                        # detection_dict["snr"] = object['snr']
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

                self.statistics_file.add_sample_attributes(sample_attributes)
                self.statistics_file.add_annotation_attributes(object_attributes)
            except FileNotFoundError as e:
                pass
            except Exception as e:
                print(f"Error processing {annotT}: {e}")

            self.save_db()

            
