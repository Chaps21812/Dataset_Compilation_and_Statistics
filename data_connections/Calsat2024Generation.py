from src.raw_datset import raw_dataset
from src.utilities import get_folders_in_directory
import os

directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01-2024-Annotations"
new_calsat_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/LMNT01-2024-Remake"

for folder in get_folders_in_directory(directory): 
    if "2025" in folder: continue
    df = raw_dataset(folder)
    df.create_calsat_dataset(new_calsat_directory, move_mode="copy", percentage_limit=1)
new_calsats = raw_dataset(new_calsat_directory)
new_calsats.complete_calsat_dataset()
new_calsats.download_TLEs()

