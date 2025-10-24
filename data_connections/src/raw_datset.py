import pickle
import pandas as pd
import os
import json
from astropy.io import fits
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display

from .astrometric_localization import detect_stars, match_to_catalogue, skycoord_to_pixels
from .collect_stats import collect_stats, collect_satsim_stats, _find_new_centroid, _find_centroid_COM
from .documentation import write_count
from .plots import plot_single_annotation, plot_error_evaluator, plot_animated_collect, plot_star_selection
from .target_injection import extract_segmented_patches_from_json_and_fits, inject_target_into_fits
from .constants import SPACECRAFT, ERROR_TYPES
from .preprocess_functions import iqr_log

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

class StatisticsFile(PickleSerializable):
    def __init__(self):
        self.sample_attributes = pd.DataFrame()
        self.annotation_attributes = pd.DataFrame()
        self.selected_fits_files = []
        self.selected_annotation_files = []
        self.current_index=0
        self.collect_dict = {}
        
    def add_sample_attributes(self, item_sample:dict):
        sample = pd.DataFrame([item_sample])
        self.sample_attributes = pd.concat([self.sample_attributes, sample], ignore_index=True)

    def add_annotation_attributes(self, annotation_sample:list):
        annotation = pd.DataFrame(annotation_sample)
        self.annotation_attributes = pd.concat([self.annotation_attributes, annotation], ignore_index=True)

    def summarize_sample_attribute_columns(self, attributes:str="sample_attributes", columns:list=[]):
        """
        Prints a summary table of unique values, counts, and percentages for given categorical columns.

        Parameters:
        - df: pandas DataFrame
        - columns: list of column names to summarize
        """
        if attributes == "sample_attributes":
            df = self.sample_attributes
        elif attributes == "annotation_attributes":
            df = self.annotation_attributes
        else:
            df = self.sample_attributes

        for col in columns:
            print(f"\nColumn: {col}")
            print("-" * (len(col) + 9))
            counts = df[col].value_counts(dropna=False)
            percentages = df[col].value_counts(normalize=True, dropna=False) * 100

            summary = pd.DataFrame({
                'Value': counts.index.astype(str),
                'Count': counts.values,
                'Percentage': percentages.values
            })

            # Format percentage column
            summary['Percentage'] = summary['Percentage'].map("{:.2f}%".format)

            # Print as text table
            print(summary.to_string(index=False))
            

class raw_dataset():
    def __init__(self, dataset_path:str):
        """
        Raw dataset that gathers information about data downloaded from UDL. Enables curation, allows annotation correction, custom annotations, statistics, and more. 

        Args:
            dataset_path (str): Path to raw dataset

        Returns:
            self
        """

        self.directory = dataset_path
        self.annotation_path = os.path.join(self.directory, "raw_annotation")
        self.fits_file_path = os.path.join(self.directory, "raw_fits")
        if "dataset_statistics.pkl" in os.listdir(self.directory):
            self.statistics_file = StatisticsFile.load(os.path.join(self.directory,"dataset_statistics.pkl"))
            self.statistics_filename = os.path.join(self.directory,"dataset_statistics.pkl")
            self.update_annotation_to_fits()
        else:
            self.statistics_filename = os.path.join(self.directory,f"dataset_statistics.pkl")
            self.recalculate_statistics()
            self.update_annotation_to_fits()
        self.collect_dict = self.statistics_file.collect_dict

    def clear_cache(self):
        """
        Deletes preprocessed data that is turned into coco datasets

        Args:
            None
        Returns:
            None
        """

        pathA = os.path.join(self.directory, "annotations")
        pathB = os.path.join(self.directory, "images")
        if os.path.exists(pathA):
            shutil.rmtree(pathA)
            print(f"Removed: {pathA}")
        if os.path.exists(pathB):
            shutil.rmtree(pathB)
            print(f"Removed: {pathB}")
        
    def update_annotation_to_fits(self):
        """
        Scans the raw directory file and matches corresponding fits files and annotations. Useful for error checking

        Args:
            None
        Returns:
            None
        """

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

    def _open_json(self, json_path:str) -> dict:
        """
        Opens a json file stored locally.

        Args:
            json_path (str): Local path to json annotation

        Returns:
            json_content (dict): Contents of desired json
        """

        with open(json_path, 'r') as f:
            json_content = json.load(f)
        return json_content
    
    def _open_fits(self, fits_path:str):
        """
        Open fits file

        Args:
            fits_path (str): Local path to fits file

        Returns:
            fits_content (fits): Raw fits file data
        """

        fits_content = fits.open(fits_path)
        return fits_content

    def _new_db(self):
        """
        Deletes the old statistics database and creats a new one

        Args:
            None
            
        Returns:
            None
        """

        self.statistics_file = StatisticsFile()
        self.statistics_file.save(self.statistics_filename)
        print(f"New database created at {self.statistics_filename}")

    def _save_db(self):
        """
        Saves info in the statistics dataframe

        Args:
            None
            
        Returns:
            None
        """
        self.statistics_file.save(self.statistics_filename)
        write_count(os.path.join(self.directory, "count.txt"), len(self.statistics_file.sample_attributes), len(self.statistics_file.annotation_attributes), self.statistics_file.sample_attributes['dates'].value_counts().to_dict())

    def delete_files_from_annotation(self, path_series: pd.DataFrame):
        """
        Deletes files from the filesystem given a pandas Series of file paths. AI Generated

        Args:
            path_series (pd.Series): Series of file paths (as strings).
        """
        unique_files = path_series["json_path"].dropna().unique()
        self.statistics_file.sample_attributes = self.statistics_file.sample_attributes[~self.statistics_file.sample_attributes["json_path"].isin(unique_files)].copy()
        self.statistics_file.annotation_attributes = self.statistics_file.annotation_attributes[~self.statistics_file.annotation_attributes["json_path"].isin(unique_files)].copy()

        # self.statistics_file.annotation_attributes.drop(path_series.index, inplace=True, axis=0)
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
        self._save_db()

    def delete_files_from_sample(self, path_series: pd.DataFrame):
        """
        Deletes files from the filesystem given a pandas Series of file paths. AI Generated

        Args:
            path_series (pd.Series): Series of file paths (as strings).
        """
        unique_files = path_series["json_path"].dropna().unique()
        self.statistics_file.annotation_attributes = self.statistics_file.annotation_attributes[~self.statistics_file.annotation_attributes["json_path"].isin(unique_files)].copy()
        self.statistics_file.sample_attributes = self.statistics_file.sample_attributes[~self.statistics_file.sample_attributes["json_path"].isin(unique_files)].copy()

        # self.statistics_file.sample_attributes.drop(path_series.index, inplace=True)
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
        self._save_db()
    
    def recalculate_statistics(self):
        """
        Calculates the statistics that plots information about our dataset. 

        Args:
            None

        Returns:
            None
        """
        self.update_annotation_to_fits()
        self._new_db()

        for annotT,fitsT in tqdm(self.annotation_to_fits.items(), desc="Recalculating Statistics"):
            try:
                json_content = self._open_json(annotT)
                fits_content = self._open_fits(fitsT)
                hdu = fits_content[0].header
                data = fits_content[0].data

                if "image_set_id" in json_content.keys():
                    if json_content["image_set_id"] not in self.statistics_file.collect_dict.keys():
                        self.statistics_file.collect_dict[json_content["image_set_id"]] = []
                        self.statistics_file.collect_dict[json_content["image_set_id"]].append({"json_path":annotT,"fits_path":fitsT})
                    else:
                        self.statistics_file.collect_dict[json_content["image_set_id"]].append({"json_path":annotT,"fits_path":fitsT})
                else:
                    if "No_Collect" not in self.statistics_file.collect_dict.keys():
                        self.statistics_file.collect_dict["No_Collect"] = []
                        self.statistics_file.collect_dict["No_Collect"].append({"json_path":annotT,"fits_path":fitsT})
                    else:
                        self.statistics_file.collect_dict["No_Collect"].append({"json_path":annotT,"fits_path":fitsT})
                
                hdu = fits_content[0].header
                if "TRKMODE" in hdu.keys():
                    if hdu["TRKMODE"] != 'rate':
                        continue

                sample_attributes, object_attributes = collect_stats(json_content, fits_content)

                sample_attributes["json_path"] = annotT
                sample_attributes["fits_path"] = fitsT
                for dictionary in object_attributes:
                    dictionary["json_path"] = annotT
                    dictionary["fits_path"] = fitsT

                self.statistics_file.add_sample_attributes(sample_attributes)
                self.statistics_file.add_annotation_attributes(object_attributes)
            except Exception as e:
                print(f"Error processing {annotT}: {e}")

        self._save_db()

    def __len__(self):
        return len(self.statistics_file.sample_attributes)

    def recenter_bounding_boxes(self, apply_changes:bool=False, require_approval:bool=False):
        """
        Resets the error characterization. 

        Args:
            None

        Returns:
            None
        """


        '''
                {
            "type": "line",
            "class_name": "Satellite",
            "class_id": 1,
            "x1": 0.39021124192350176,
            "y1": 0.8454258609775411,
            "x2": 0.38390235781024695,
            "y2": 0.8424744035365839,
            "x_center": 0.38705679986687436,
            "y_center": 0.8439501322570625,
            "source": "turk_new",
            "correlation_id": "f1d3d283-5246-4008-ad78-00bb7f5a7c75",
            "index": 4,
            "iso_flux": 1651756
        }'''
        for annotation, fits_pat in tqdm(self.annotation_to_fits.items()):
            json_path = annotation
            fits_path = fits_pat
            title = os.path.basename(annotation)
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    continue

                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]

                original_bbox = (x_corner, y_corner, width, height)
                # new_bbox = find_new_centroid(image, original_bbox)
                new_bbox = _find_centroid_COM(image, original_bbox)

                if require_approval:
                    plot_single_annotation(image, original_bbox, new_bbox, title)
                    current_input = input("Keep New Annotation? [Enter any letter for yes]: ")
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
                    if current_input and apply_changes:
                        new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
                        data["objects"][j]["x_min"] = new_bbox[0]
                        data["objects"][j]["y_min"] = new_bbox[1]
                        data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
                        data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
                        data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
                        data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
                        data["objects"][j]["bbox_width"] = new_bbox[2]
                        data["objects"][j]["bbox_height"] = new_bbox[3]
                else:
                    new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
                    data["objects"][j]["x_min"] = new_bbox[0]
                    data["objects"][j]["y_min"] = new_bbox[1]
                    data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
                    data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
                    data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
                    data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
                    data["objects"][j]["bbox_width"] = new_bbox[2]
                    data["objects"][j]["bbox_height"] = new_bbox[3]

            # Save the updated JSON back to the same file
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        self.update_annotation_to_fits()
        self.recalculate_statistics()

    def reset_errors(self):
        """
        Resets the error characterization. 

        Args:
            None

        Returns:
            None
        """
        os.remove(os.path.join(self.directory, "errors.pkl"))

    def characterize_errors(self):
        """
        Creates a UI to label and classify errors in a dataset. Only works in python notebook. Each number is associated with an error code. 

        Args:
            None

        Returns:
            None
        """
        if os.path.exists(os.path.join(self.directory, "errors.pkl")):
            error_database = StatisticsFile.load(os.path.join(self.directory, "errors.pkl"))
        else: 
            error_database = StatisticsFile()

        for image_index, (annotation, fits_pat) in tqdm(enumerate(self.annotation_to_fits.items())):
            if image_index < error_database.current_index:
                continue

            json_path = annotation
            fits_path = fits_pat
            bboxes = []
            properties = {}
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            properties["fits_file"] = data["file"]["filename"]
            properties["sensor"] = data["file"]["id_sensor"]
            properties["QA"] = data["approved"]
            properties["labeler_id"] = data["labeler_id"]
            properties["request_id"] = data["request_id"]
            properties["created"] = datetime.strptime(data["created"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["updated"] = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["num_objects"] = len(data["objects"])

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    properties["error_type"] = 9
                    error_database.add_sample_attributes(properties)
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
                    continue


                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]

                original_bbox = (x_corner, y_corner, width, height)
                bboxes.append(original_bbox)


            if len(bboxes) ==0:
                plot_error_evaluator(image, [], 0, properties)
                error = _error_input_prompt()
                properties["error_type"] = error
                error_database.add_sample_attributes(properties)
                plt.clf()   # Clears the current figure
                plt.cla()   # Clears the current axes (optional)
                plt.close()
                clear_output(wait=True)
            else: 
                for index in range(len(bboxes)):
                    plot_error_evaluator(image, bboxes, index, properties)
                    error = _error_input_prompt()
                    properties["error_type"] = error
                    error_database.add_sample_attributes(properties)
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)

            error_database.sample_attributes['error_type_str'] = error_database.sample_attributes.apply(lambda row: ERROR_TYPES[row['error_type']] if row["error_type"] < len(ERROR_TYPES) else ERROR_TYPES[8], axis=1)
            error_database.current_index +=1
            error_database.save(os.path.join(self.directory, "errors.pkl"))

    def inject_targets(self, segmentations_path:str, threshold_factor=1.2, bbox_scale=1.5):
        #Collect targets to inject
        local_segmentations = raw_dataset(segmentations_path)
        target_array = []
        for annotation_path, fits_path in local_segmentations.annotation_to_fits.items():
            targets_to_inject = extract_segmented_patches_from_json_and_fits(annotation_path, fits_path,threshold_factor=threshold_factor, bbox_scale=bbox_scale)
            target_array.extend(targets_to_inject)

        for annotation, fits_pat in tqdm(self.annotation_to_fits.items()):
            json_path = annotation
            fits_path = fits_pat
            title = os.path.basename(annotation)
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            num_targets_to_inject = np.random.randint(0,4)
            for i in range(len(num_targets_to_inject)):
                patch = np.random.choice(target_array)
                image, injection_bbox = inject_target_into_fits(image,patch["original_patch"],random_rotation=True,display=True,seed=None)
                injected_target_dict = {
                    "type": "box",
                    "class_name": "Satellite",
                    "class_id": 1,
                    "y_min": injection_bbox[1],
                    "x_min": injection_bbox[0],
                    "y_max": injection_bbox[1]+injection_bbox[3],
                    "x_max": injection_bbox[0]+injection_bbox[2],
                    "y_center": injection_bbox[1]+injection_bbox[3]/2,
                    "x_center": injection_bbox[0]+injection_bbox[2]/2,
                    "bbox_height": injection_bbox[3],
                    "bbox_width": injection_bbox[2], 
                    "datatype":"injected"}
                data["objects"].append(injected_target_dict)
            hdu[0].data = image

            #DONT FORGET TO SAVE THE NEW IMAGE
            # Save the updated JSON back to the same file
            hdu.writeto(fits_path, overwrite=True)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        self.update_annotation_to_fits()
        self.recalculate_statistics()

    def inject_targets_from_numpy(self, segmentations_path:str):
        #Collect targets to inject
        numpy_arrays = os.listdir(segmentations_path)
        target_arrays = []
        for array_path in numpy_arrays:
            target = np.load(os.path.join(segmentations_path, array_path))
            target_arrays.append(target)

        for json_path, fits_path in tqdm(self.annotation_to_fits.items()):
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            num_targets_to_inject = np.random.randint(0,4)
            for i in range(num_targets_to_inject):
                idx = np.random.choice(len(target_arrays))
                patch = target_arrays[idx]
                image, injection_bbox = inject_target_into_fits(image,patch,random_rotation=True,display=False,seed=None)
                injected_target_dict = {
                    "type": "box",
                    "class_name": "Satellite",
                    "correlation_id": "404", 
                    "iso_flux": "404",
                    "class_id": 1,
                    "y_min": injection_bbox[1],
                    "x_min": injection_bbox[0],
                    "y_max": injection_bbox[1]+injection_bbox[3],
                    "x_max": injection_bbox[0]+injection_bbox[2],
                    "y_center": injection_bbox[1]+injection_bbox[3]/2,
                    "x_center": injection_bbox[0]+injection_bbox[2]/2,
                    "bbox_height": injection_bbox[3],
                    "bbox_width": injection_bbox[2], 
                    "datatype":"Injected"}
                data["objects"].append(injected_target_dict)
            hdu[0].data = image

            #DONT FORGET TO SAVE THE NEW IMAGE
            # Save the updated JSON back to the same file
            hdu.writeto(fits_path, overwrite=True)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        self.update_annotation_to_fits()
        self.recalculate_statistics()

    def reset_subset_selection(self):
        """
        Resets the temp file that selects and moves items into a new dataset. 

        Args:
            None

        Returns:
            None
        """
        os.remove(os.path.join(self.directory, "subset.pkl"))

    def create_hand_selected_dataset(self, new_dataset_directory:str, move_mode:str="copy"):
        """
        Creates a UI to hand select a dataset by image. Only runs in jupyter notebook, press any key and enter to select an image. Press enter to leave the collect.

        Args:
            new_dataset_directory (str): The directory of the new dataset to copy or move files to
            move_mode (bool): "copy" copies the original file to the new directory. "move" removes the file from the old dataset and places it in the new dataset

        Returns:
            None
        """
        os.makedirs(new_dataset_directory, exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_fits"), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_annotation"), exist_ok=True)

        if os.path.exists(os.path.join(self.directory, "subset.pkl")):
            subset_database = StatisticsFile.load(os.path.join(self.directory, "subset.pkl"))
        else: 
            subset_database = StatisticsFile()

        for image_index, (annotation, fits_pat) in tqdm(enumerate(self.annotation_to_fits.items())):
            if image_index < subset_database.current_index:
                continue

            json_path = annotation
            fits_path = fits_pat
            bboxes = []
            properties = {}
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            properties["fits_file"] = data["file"]["filename"]
            properties["sensor"] = data["file"]["id_sensor"]
            properties["QA"] = data["approved"]
            properties["labeler_id"] = data["labeler_id"]
            properties["request_id"] = data["request_id"]
            properties["created"] = datetime.strptime(data["created"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["updated"] = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["num_objects"] = len(data["objects"])

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
                    continue
                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]
                original_bbox = (x_corner, y_corner, width, height)
                bboxes.append(original_bbox)


            plot_error_evaluator(image, bboxes, 0, properties)
            if _select_input_prompt():
                subset_database.selected_annotation_files.append(json_path)
                subset_database.selected_fits_files.append(fits_path)
                if move_mode == "move":
                    shutil.move(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.move(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                elif move_mode == "copy":
                    shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                else: 
                    shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
            plt.clf()   # Clears the current figure
            plt.cla()   # Clears the current axes (optional)
            plt.close()
            clear_output(wait=True)

            subset_database.current_index +=1
            subset_database.save(os.path.join(self.directory, "subset.pkl"))
        self.recalculate_statistics()

    def create_hand_selected_dataset_by_collect(self, new_dataset_directory:str, move_mode:str="copy"):
        """
        Creates a UI to hand select a dataset by collect ID. Only runs in jupyter notebook, press any key and enter to select a collect. Press enter to leave the collect.

        Args:
            new_dataset_directory (str): The directory of the new dataset to copy or move files to
            move_mode (bool): "copy" copies the original file to the new directory. "move" removes the file from the old dataset and places it in the new dataset

        Returns:
            None
        """
        os.makedirs(new_dataset_directory, exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_fits"), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_annotation"), exist_ok=True)

        if os.path.exists(os.path.join(self.directory, "collect_subset.pkl")):
            subset_database = StatisticsFile.load(os.path.join(self.directory, "collect_subset.pkl"))
        else: 
            subset_database = StatisticsFile()

        anis = None

        for collect_index, (collect_id, paths) in tqdm(enumerate(self.collect_dict.items()), desc="Collect", total=len(self.collect_dict)):
            if collect_index < subset_database.current_index:
                continue
            images_list = []
            bboxes_list = []
            attrs_list = []

            index_list = {}
            
            for j, pathset in enumerate(paths):
                json_path = pathset["json_path"]
                fits_path = pathset["fits_path"]
                bboxes = []
                properties = {}
                # Open and load the JSON file
                with open(json_path, 'r') as f:
                    data = json.load(f)
                hdu = fits.open(fits_path)
                hdul = hdu[0]
                image = hdul.data
                images_list.append(image)
                index_list[data["sequence_id"]] = j
                
                properties["fits_file"] = data["file"]["filename"]
                properties["sensor"] = data["file"]["id_sensor"]
                properties["QA"] = data["approved"]
                properties["labeler_id"] = data["labeler_id"]
                properties["image_set_id"] = data["image_set_id"]
                properties["sequence_id"] = data["sequence_id"]
                properties["request_id"] = data["request_id"]
                properties["created"] = datetime.strptime(data["created"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
                properties["updated"] = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
                properties["num_objects"] = len(data["objects"])

                for j,box in enumerate(data["objects"]):
                    if box["type"] == "line":
                        continue
                    x_corner = box["x_min"]*image.shape[1]
                    y_corner = box["y_min"]*image.shape[0]
                    width = (box["x_max"]-box["x_min"])*image.shape[1]
                    height = (box["y_max"]-box["y_min"])*image.shape[0]
                    original_bbox = (x_corner, y_corner, width, height)
                    bboxes.append(original_bbox)
                bboxes_list.append(bboxes)
                attrs_list.append(properties)

            ordered_images = [images_list[index_list[k]] for k in sorted(index_list.keys())]
            ordered_bboxes = [bboxes_list[index_list[k]] for k in sorted(index_list.keys())]
            ordered_attrs = [attrs_list[index_list[k]] for k in sorted(index_list.keys())]

            fig, ani = plot_animated_collect(ordered_images, ordered_bboxes, ordered_attrs)
            from IPython.display import display, HTML
            display(HTML(ani.to_jshtml()))  # reliable inline display
            # ani.save("animation.mp4", writer="ffmpeg", fps=5)
            # from IPython.display import Video
            # display(Video("animation.mp4", embed=True))
                        
            if _select_input_prompt():
                for pathset in paths:
                    json_path = pathset["json_path"]
                    fits_path = pathset["fits_path"]
                    subset_database.selected_annotation_files.append(json_path)
                    subset_database.selected_fits_files.append(fits_path)
                    if move_mode == "move":
                        shutil.move(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.move(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                    elif move_mode == "copy":
                        shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                    else: 
                        shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
            # plt.clf()   # Clears the current figure
            # plt.cla()   # Clears the current axes (optional)
            # plt.close()
            clear_output(wait=True)

            subset_database.current_index +=1
            subset_database.save(os.path.join(self.directory, "collect_subset.pkl"))
        self.recalculate_statistics()

    def create_calsat_dataset(self, new_dataset_directory, move_mode:str="copy"):
        """
        Creates a dataset with Calsats registered in the constants SPACECRAFT dictionary. 

        Args:
            new_dataset_directory (str): The directory of the new dataset to copy or move files to
            move_mode (bool): "copy" copies the original file to the new directory. "move" removes the file from the old dataset and places it in the new dataset

        Returns:
            None
        """
        os.makedirs(new_dataset_directory, exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_fits"), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_annotation"), exist_ok=True)

        if os.path.exists(os.path.join(self.directory, "calsats.pkl")):
            subset_database = StatisticsFile.load(os.path.join(self.directory, "calsats.pkl"))
        else: 
            subset_database = StatisticsFile()

        calsats = 0

        for image_index, (annotation, fits_pat) in tqdm(enumerate(self.annotation_to_fits.items())):
            json_path = annotation
            fits_path = fits_pat
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            try:
                norad_id = data["file"]["filename"].split(".")[0].split("sat_")[1]
                calsat = SPACECRAFT[norad_id]
                if calsat is not None:
                    subset_database.selected_annotation_files.append(json_path)
                    subset_database.selected_fits_files.append(fits_path)
                    if move_mode == "move":
                        shutil.move(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.move(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                    elif move_mode == "copy":
                        shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                    else: 
                        shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                        shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                    calsats += 1
            except KeyError:
                continue
            except IndexError:
                continue
            subset_database.current_index +=1
            subset_database.save(os.path.join(self.directory, "calsats.pkl"))
        print(f"Num calsats: {calsats}")
        self.recalculate_statistics()

    def create_target_quality_dataset(self, new_dataset_directory:str, move_mode:str="copy"):
        """
        Creates a UI which labels a raw dataset with star quality and target quality. Only runs in a jupyter notebook.

        Args:
            new_dataset_directory (str): The directory of the new dataset to copy or move files to
            move_mode (bool): "copy" copies the original file to the new directory. "move" removes the file from the old dataset and places it in the new dataset

        Returns:
            None
        """
        os.makedirs(new_dataset_directory, exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_fits"), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_directory, "raw_annotation"), exist_ok=True)

        if os.path.exists(os.path.join(self.directory, "quality_subset.pkl")):
            subset_database = StatisticsFile.load(os.path.join(self.directory, "quality_subset.pkl"))
        else: 
            subset_database = StatisticsFile()

        for image_index, (annotation, fits_pat) in tqdm(enumerate(self.annotation_to_fits.items())):
            if image_index < subset_database.current_index:
                continue

            json_path = annotation
            fits_path = fits_pat
            bboxes = []
            properties = {}
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            properties["fits_file"] = data["file"]["filename"]
            properties["sensor"] = data["file"]["id_sensor"]
            properties["QA"] = data["approved"]
            properties["labeler_id"] = data["labeler_id"]
            properties["request_id"] = data["request_id"]
            properties["created"] = datetime.strptime(data["created"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["updated"] = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["num_objects"] = len(data["objects"])

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
                    continue
                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]
                original_bbox = (x_corner, y_corner, width, height)
                bboxes.append(original_bbox)
            plot_error_evaluator(image, bboxes, 0, properties)


            star_quality = _select_star_quality()
            target_quality = _select_target_quality()

            if star_quality and target_quality:
                subset_database.selected_annotation_files.append(json_path)
                subset_database.selected_fits_files.append(fits_path)
                if move_mode == "move":
                    shutil.move(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.move(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                elif move_mode == "copy":
                    shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                else: 
                    shutil.copy(json_path, os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)))
                    shutil.copy(fits_path, os.path.join(new_dataset_directory, "raw_fits", os.path.basename(fits_path)))
                with open(os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)), "r") as g:
                    annotation_data = json.load(g)
                    annotation_data["star_quality"] = star_quality
                    annotation_data["target_quality"] = target_quality
                with open(os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)), "w") as g:
                    json.dump(os.path.join(new_dataset_directory, "raw_annotation", os.path.basename(json_path)), g, indent=4)

            plt.clf()   # Clears the current figure
            plt.cla()   # Clears the current axes (optional)
            plt.close()
            clear_output(wait=True)

            subset_database.current_index +=1
            subset_database.save(os.path.join(self.directory, "quality_subset.pkl"))
        self.recalculate_statistics()

    def correct_annotations(self, apply_changes:bool=True, require_approval:bool=False):
        """
        Applies bounding box recentroiding to annotations. 

        Args:
            require_approval (bool): If True, enters a manual mode to approve each recentroid. Only works in python notebook
            apply_changes (bool): Only applicable to manual approval mode. If True, it applies the changes of recentroiding. If False, no recentroiding is applied. 

        Returns:
            None
        """
        for annotation, fits_pat in tqdm(self.annotation_to_fits.items()):
            json_path = annotation
            fits_path = fits_pat
            title = os.path.basename(annotation)
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    continue

                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]

                original_bbox = (x_corner, y_corner, width, height)
                # new_bbox = find_new_centroid(image, original_bbox)
                new_bbox = _find_centroid_COM(image, original_bbox)

                if require_approval:
                    plot_single_annotation(image, original_bbox, new_bbox, title)
                    current_input = input("Keep New Annotation? [Enter any letter for yes]: ")
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
                    if current_input and apply_changes:
                        new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
                        data["objects"][j]["x_min"] = new_bbox[0]
                        data["objects"][j]["y_min"] = new_bbox[1]
                        data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
                        data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
                        data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
                        data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
                        data["objects"][j]["bbox_width"] = new_bbox[2]
                        data["objects"][j]["bbox_height"] = new_bbox[3]
                else:
                    new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
                    data["objects"][j]["x_min"] = new_bbox[0]
                    data["objects"][j]["y_min"] = new_bbox[1]
                    data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
                    data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
                    data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
                    data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
                    data["objects"][j]["bbox_width"] = new_bbox[2]
                    data["objects"][j]["bbox_height"] = new_bbox[3]

            # Save the updated JSON back to the same file
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        self.update_annotation_to_fits()
        self.recalculate_statistics()

    def plate_solve(self):
        if os.path.exists(os.path.join(self.directory, "wcs.pkl")):
            wcs_database = StatisticsFile.load(os.path.join(self.directory, "wcs.pkl"))
        else: 
            wcs_database = StatisticsFile()
        for image_index, (annotation, fits_pat) in tqdm(enumerate(self.annotation_to_fits.items())):
            submit_flag = False
            if image_index < wcs_database.current_index:
                continue

            json_path = annotation
            fits_path = fits_pat
            bboxes = []
            properties = {}
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            properties["fits_file"] = data["file"]["filename"]
            properties["sensor"] = data["file"]["id_sensor"]
            properties["QA"] = data["approved"]
            properties["labeler_id"] = data["labeler_id"]
            properties["request_id"] = data["request_id"]
            properties["created"] = datetime.strptime(data["created"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["updated"] = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d")
            properties["num_objects"] = len(data["objects"])

            for j,box in enumerate(data["objects"]):
                if box["type"] == "line":
                    continue


                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]

                original_bbox = (x_corner, y_corner, width, height)
                bboxes.append(original_bbox)


            points = plot_star_selection(image, bboxes, 0, attributes=properties)


            plt.close()
            clear_output(wait=False)
            print
            # new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
            # data["objects"][j]["x_min"] = new_bbox[0]
            # data["objects"][j]["y_min"] = new_bbox[1]
            # data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
            # data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
            # data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
            # data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
            # data["objects"][j]["bbox_width"] = new_bbox[2]
            # data["objects"][j]["bbox_height"] = new_bbox[3]


            # Save the updated JSON back to the same file
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            wcs_database.current_index +=1
            wcs_database.save(os.path.join(self.directory, "errors.pkl"))
        self.update_annotation_to_fits()
        self.recalculate_statistics()

def _error_input_prompt():
    current_input = input("Enter error number: ")
    if current_input:
        return int(current_input)
    else: 
        return 8
    
def _select_input_prompt():
    current_input = input("Non-zero character to select ")
    if current_input:
        return True
    else: 
        return False

def _select_star_quality():
    current_input = input("Enter Star Quality (0=worst, 2=best)")
    if current_input:
        return True
    else: 
        return False
    
def _select_target_quality():
    current_input = input("Enter Target Quality (0=worst, 2=best)")
    if current_input:
        return True
    else: 
        return False

class satsim_path_loader():
    def __init__(self, dataset_path:str):
        self.directory = dataset_path
        self.annotation_to_fits = {}

        stats_files = [f for f in os.listdir(self.directory) if f.endswith(".pkl")]
        if len(stats_files) == 0:
            self.new_db(os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl"))
            self.statistics_file = StatisticsFile.load(os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl"))
            self.statistics_filename = os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl")
        else: 
            print(os.path.join(self.directory, stats_files[0]))
            self.statistics_file = StatisticsFile.load(os.path.join(self.directory, stats_files[0]))
            self.statistics_filename = os.path.join(self.directory,stats_files[0])
        self.update_annotation_to_fits()
    
    def clear_cache(self):
        pathA = os.path.join(self.directory, "annotations")
        pathB = os.path.join(self.directory, "images")
        if os.path.exists(pathA):
            shutil.rmtree(pathA)
            print(f"Removed: {pathA}")
        if os.path.exists(pathB):
            shutil.rmtree(pathB)
            print(f"Removed: {pathB}")
            
    def update_annotation_to_fits(self):
        folders = [name for name in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, name))]
        if "annotations" in folders: folders.remove("annotations")
        if "images" in folders: folders.remove("images")
        if "plots" in folders: folders.remove("plots")
        if "annotation_view" in folders: folders.remove("annotation_view")

        for folder in tqdm(folders, desc="Loading folders"):
            annotations_sub_folder = os.path.join(self.directory, folder, "Annotations")
            fits_sub_folder = os.path.join(self.directory, folder, "ImageFiles")

            annot_files = [name for name in os.listdir(annotations_sub_folder) if os.path.isfile(os.path.join(annotations_sub_folder, name))]
            for annot in annot_files:
                fits_file =annot.replace(".json", ".fits")
                fits_file_location = os.path.join(fits_sub_folder, fits_file)
                annot_file_location = os.path.join(annotations_sub_folder, annot)
                fits_folder_contents = os.listdir(fits_sub_folder)
                if fits_file not in fits_folder_contents:
                    print(f"Warning: {fits_file} not found in {fits_sub_folder}.")
                    continue
                self.annotation_to_fits[annot_file_location] = fits_file_location

    def open_json(self, json_path:str):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        return json_content
    
    def open_fits(self, fits_path:str):
        fits_content = fits.open(fits_path)
        return fits_content

    def new_db(self, filename=None):
        self.statistics_file = StatisticsFile()
        if filename is None:
            self.statistics_file.save(self.statistics_filename)
            print(f"New database created at {self.statistics_filename}")
        else:
            self.statistics_filename = filename
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
        unique_files = path_series["json_path"].dropna().unique()
        self.statistics_file.sample_attributes = self.statistics_file.sample_attributes[~self.statistics_file.sample_attributes["json_path"].isin(unique_files)].copy()

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
        unique_files = path_series["json_path"].dropna().unique()
        self.statistics_file.annotation_attributes = self.statistics_file.annotation_attributes[~self.statistics_file.annotation_attributes["json_path"].isin(unique_files)].copy()

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
            try:
                json_content = self.open_json(annotT)
                fits_content = self.open_fits(fitsT)
                hdu = fits_content[0].header
                
                if "TRKMODE" in hdu.keys():
                    if hdu["TRKMODE"] != 'rate':
                        continue

                sample_attributes, object_attributes = collect_satsim_stats(json_content, fits_content)

                sample_attributes["json_path"] = annotT
                sample_attributes["fits_path"] = fitsT
                for dictionary in object_attributes:
                    dictionary["json_path"] = annotT
                    dictionary["fits_path"] = fitsT

                self.statistics_file.add_sample_attributes(sample_attributes)
                self.statistics_file.add_annotation_attributes(object_attributes)
            except Exception as e:
                print(f"Error processing {annotT}: {e}")

        self.save_db()

    def __len__(self):
        return len(self.statistics_file.sample_attributes)

class coco_path_loader():
    def __init__(self, dataset_path:str):
        self.directory = dataset_path
        if len([f for f in os.listdir(self.directory) if f.endswith(".pkl")]) ==0:
            self.statistics_filename = os.path.join(dataset_path, f"{os.path.basename(dataset_path)}.pkl")
            self.recalculate_statistics()
        else:
            self.statistics_file = StatisticsFile.load(os.path.join(self.directory,[f for f in os.listdir(self.directory) if f.endswith(".pkl")][0]))
            self.statistics_filename = os.path.join(self.directory,[f for f in os.listdir(self.directory) if f.endswith(".pkl")][0])
        self.annotation_path = os.path.join(self.directory, "annotations")
        self.png_file_path = os.path.join(self.directory, "images")
        self.update_annotation_to_png()

    def update_annotation_to_png(self):
        self.annotations = self.open_json(os.path.join(self.annotation_path, "annotations.json"))
        self.png_files = os.listdir(self.png_file_path)
        self.image_id_to_index = {}

        for index, image_dict in self.annotations["images"]:
            self.image_id_to_index[image_dict["id"]] = index
        # self.annotations["annotations"]
        
    def open_json(self, json_path:str):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        return json_content
    
    def new_db(self):
        self.statistics_file = StatisticsFile()
        self.statistics_file.save(self.statistics_filename)
        print(f"New database created at {self.statistics_filename}")

    def save_db(self):
        self.statistics_file.save(self.statistics_filename)
        write_count(os.path.join(self.directory, "count.txt"), len(self.statistics_file.sample_attributes), len(self.statistics_file.annotation_attributes), self.statistics_file.sample_attributes['dates'].value_counts().to_dict())

    def save_annotations(self):
        with open(os.path.join(self.annotation_path, "annotations.json"), 'w') as f:
            json.dump(self.annotations, f, indent=4)

    def delete_files(self, path_series: pd.DataFrame):
        ##### TO DO FIX
        """ 
        Deletes files from the filesystem given a pandas Series of file paths. AI Generated

        Args:
            path_series (pd.Series): Series of file paths (as strings).
        """

        if "image_id" in path_series.columns:
            unique_images = path_series["image_id"].dropna().unique()
        else: 
            unique_images = path_series["id"].dropna().unique()

        self.statistics_file.sample_attributes = self.statistics_file.sample_attributes[~self.statistics_file.sample_attributes["id"].isin(unique_images)].copy()
        self.statistics_file.annotation_attributes = self.statistics_file.annotation_attributes[~self.statistics_file.annotation_attributes["image_id"].isin(unique_images)].copy()

        # self.statistics_file.annotation_attributes.drop(path_series.index, inplace=True, axis=0)
        for path in tqdm(path_series["json_path"].dropna(), desc="Deleting files"):
            fits_path = self.annotation_to_png[path]
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
        self.save_annotations()
    
    def recalculate_statistics(self):
        ##### TO DO FIX
        self.update_annotation_to_png()
        self.new_db()

        for annotAttrs in tqdm(self.annotations["annotations"], desc="Recalculating Statistics"):
            try:
                self.statistics_file.add_annotation_attributes(annotAttrs)
            except Exception as e:
                print(f"Error processing annotation {annotAttrs["id"]}: {e}")
        for imageAttrs in tqdm(self.annotations["images"], desc="Recalculating Statistics"):
            try:
                self.statistics_file.add_sample_attributes(imageAttrs)
            except Exception as e:
                print(f"Error processing image {imageAttrs["id"]}: {e}")
        self.save_db()

    def __len__(self):
        return len(self.statistics_file.sample_attributes)

    def correct_annotations(self):
        ##### TO DO FIX
        for annotation, fits_pat in tqdm(self.annotation_to_png.items()):
            json_path = annotation
            fits_path = fits_pat
            title = os.path.basename(annotation)
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            hdu = fits.open(fits_path)
            hdul = hdu[0]
            image = hdul.data

            for j,box in enumerate(data["objects"]):
                if len(data["objects"]) >=2:
                    print("2")
                x_corner = box["x_min"]*image.shape[1]
                y_corner = box["y_min"]*image.shape[0]
                width = (box["x_max"]-box["x_min"])*image.shape[1]
                height = (box["y_max"]-box["y_min"])*image.shape[0]

                original_bbox = (x_corner, y_corner, width, height)
                new_bbox = _find_new_centroid(image, original_bbox)
                plot_single_annotation(image, original_bbox, new_bbox, title)

                current_input = input("Keep New Annotation? [Enter any letter for yes]: ")
                plt.clf()   # Clears the current figure
                plt.cla()   # Clears the current axes (optional)
                plt.close()
                clear_output(wait=True)
                if current_input:
                    new_bbox = (new_bbox[0]/image.shape[1], new_bbox[1]/image.shape[0], new_bbox[2]/image.shape[1], new_bbox[3]/image.shape[0])
                    data["objects"][j]["x_min"] = new_bbox[0]
                    data["objects"][j]["y_min"] = new_bbox[1]
                    data["objects"][j]["x_max"] = new_bbox[0]+new_bbox[2]
                    data["objects"][j]["y_max"] = new_bbox[1]+new_bbox[3]
                    data["objects"][j]["x_center"] = (new_bbox[0]+new_bbox[2]/2)
                    data["objects"][j]["y_center"] = (new_bbox[1]+new_bbox[3]/2)
                    data["objects"][j]["bbox_width"] = new_bbox[2]
                    data["objects"][j]["bbox_height"] = new_bbox[3]

            # Save the updated JSON back to the same file
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        self.update_annotation_to_png()
        self.recalculate_statistics()


if __name__ == "__main__":
    # # satsim_path = "/mnt/c/Users/david.chaparro/Documents/Repos/SatSim/output"
    # # local_satsim = satsim_path_loader(satsim_path)

    # dataset_directory = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset_Statistics/data/KWAJData"

    # #Local file handling tool
    # local_files = file_path_loader(dataset_directory)
    # local_files.recalculate_statistics()
    # print(f"Num Samples: {len(local_files)}")


    path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw"
    for folder in os.listdir(path):
        local_files = raw_dataset(os.path.join(path,folder))
        local_files.create_calsat_dataset("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CalSatLMNT01-2024")