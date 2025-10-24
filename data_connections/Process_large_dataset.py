from src.raw_datset import raw_dataset
from src.utilities import get_folders_in_directory
import os

directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025_Annotations"

get_folders_in_directory(directory)

for folder in get_folders_in_directory(directory):
    df = raw_dataset(folder)
    df.correct_annotations()

"/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025_Annotations/2025-09-09"