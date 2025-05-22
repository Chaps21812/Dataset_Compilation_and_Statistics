import pickle
import pandas as pd
import os
import json
from astropy.io import fits
from tqdm import tqdm
from collect_stats import collect_stats, collect_satsim_stats, find_new_centroid
from documentation import write_count
from plots import plot_single_annotation
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil

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

    def correct_annotations(self):
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
        self.update_annotation_to_fits()
        self.recalculate_statistics()

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
    # satsim_path = "/mnt/c/Users/david.chaparro/Documents/Repos/SatSim/output"
    # local_satsim = satsim_path_loader(satsim_path)

    path="/home/davidchaparro/Repos/Dataset_Compilation_and_Statistics/data/dummydata"
    local = file_path_loader(path)
    local.correct_annotations()

