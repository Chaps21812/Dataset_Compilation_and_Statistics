from src.raw_datset import raw_dataset
from src.utilities import get_folders_in_directory
import os

directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04_Raw"
new_calsat_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2024_Calsats"

get_folders_in_directory(directory)

fits_dir = os.path.join(new_calsat_directory, "raw_fits")
for folder in get_folders_in_directory(directory): 
    if "2025" in folder: continue
    df = raw_dataset(folder)
    df.create_calsat_dataset(new_calsat_directory, move_mode="copy")
    if len(os.listdir(fits_dir)) > 1000:
        break

