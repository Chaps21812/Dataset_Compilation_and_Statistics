import pickle
import pandas as pd
import os
import json
from astropy.io import fits
from tqdm import tqdm
from collect_stats import collect_stats, collect_satsim_stats, find_new_centroid, find_centroid_COM
from documentation import write_count
from plots import plot_single_annotation, plot_error_evaluator
from target_injection import extract_segmented_patches_from_json_and_fits, inject_target_into_fits
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil
from datetime import datetime
import numpy as np

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
        self.annotation_path = os.path.join(self.directory, "raw_annotation")
        self.fits_file_path = os.path.join(self.directory, "raw_fits")
        if len([f for f in os.listdir(self.directory) if (f.endswith(".pkl") and "error" not in f)]) > 0:
            self.statistics_file = PDStatistics_calculator.load(os.path.join(self.directory,[f for f in os.listdir(self.directory) if (f.endswith(".pkl") and "error" not in f)][0]))
            self.statistics_filename = os.path.join(self.directory,[f for f in os.listdir(self.directory) if (f.endswith(".pkl") and "error" not in f)][0])
            self.update_annotation_to_fits()
        else:
            self.statistics_filename = os.path.join(self.directory,f"{os.path.basename(self.directory)}.pkl")
            self.recalculate_statistics()

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
        self.save_db()

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
        self.save_db()
    
    def recalculate_statistics(self):
        self.update_annotation_to_fits()
        self.new_db()

        for annotT,fitsT in tqdm(self.annotation_to_fits.items(), desc="Recalculating Statistics"):
            try:
                json_content = self.open_json(annotT)
                fits_content = self.open_fits(fitsT)
                hdu = fits_content[0].header
                data = fits_content[0].data
                
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

        self.save_db()

    def __len__(self):
        return len(self.statistics_file.sample_attributes)

    def correct_annotations(self, apply_changes:bool=False, require_approval:bool=True):
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
                new_bbox = find_centroid_COM(image, original_bbox)

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

    def characterize_errors(self):
        error_database = PDStatistics_calculator()
        for annotation, fits_pat in tqdm(self.annotation_to_fits.items()):
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
                    error_database.save(os.path.join(self.directory, "errors.pkl"))
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
                error = error_input_prompt()
                properties["error_type"] = error
                error_database.add_sample_attributes(properties)
                error_database.save(os.path.join(self.directory, "errors.pkl"))
                plt.clf()   # Clears the current figure
                plt.cla()   # Clears the current axes (optional)
                plt.close()
                clear_output(wait=True)
            else: 
                for index in range(len(bboxes)):
                    plot_error_evaluator(image, bboxes, index, properties)
                    error = error_input_prompt()
                    properties["error_type"] = error
                    error_database.add_sample_attributes(properties)
                    error_database.save(os.path.join(self.directory, "errors.pkl"))
                    plt.clf()   # Clears the current figure
                    plt.cla()   # Clears the current axes (optional)
                    plt.close()
                    clear_output(wait=True)
        error_types = ["No Error", "Uncentered Box", "Severely Uncentered Box", "Missed Target", "Blank Box", "Silt Transpose Error", "Occlusion [Edge or star]", "Other", "Unknown", "Long Satellite Streak"]
        error_database.sample_attributes['error_type_str'] = error_database.sample_attributes.apply(lambda row: error_types[row['error_type']] if row["error_type"] < len(error_types) else error_types[8], axis=1)

    def inject_targets(self, segmentations_path:str, threshold_factor=1.2, bbox_scale=1.5):
        #Collect targets to inject
        local_segmentations = file_path_loader(segmentations_path)
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

def error_input_prompt():
    current_input = input("Enter error number: ")
    if current_input:
        return int(current_input)
    else: 
        return 8

class satsim_path_loader():
    def __init__(self, dataset_path:str):
        self.directory = dataset_path
        self.annotation_to_fits = {}

        stats_files = [f for f in os.listdir(self.directory) if f.endswith(".pkl")]
        if len(stats_files) == 0:
            self.new_db(os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl"))
            self.statistics_file = PDStatistics_calculator.load(os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl"))
            self.statistics_filename = os.path.join(self.directory, os.path.basename(dataset_path) + "_statistics.pkl")
        else: 
            print(os.path.join(self.directory, stats_files[0]))
            self.statistics_file = PDStatistics_calculator.load(os.path.join(self.directory, stats_files[0]))
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
        self.statistics_file = PDStatistics_calculator()
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
            self.statistics_file = PDStatistics_calculator.load(os.path.join(self.directory,[f for f in os.listdir(self.directory) if f.endswith(".pkl")][0]))
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
        self.statistics_file = PDStatistics_calculator()
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
                new_bbox = find_new_centroid(image, original_bbox)
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


    path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw/RME04Sat-2024-05-17-injected"
    segmentation_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Injection_targets"
    local_files = file_path_loader(path)
    local_files.inject_targets_from_numpy(segmentation_path)