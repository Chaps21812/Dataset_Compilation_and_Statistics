import pickle
import pandas as pd
import os
import json
from astropy.io import fits
from tqdm import tqdm
from collect_stats import collect_stats, collect_satsim_stats

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
                data = fits_content[0].data
                
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



if __name__ == "__main__":
    satsim_path = "/mnt/c/Users/david.chaparro/Documents/Repos/SatSim/output"
    local_satsim = satsim_path_loader(satsim_path)

