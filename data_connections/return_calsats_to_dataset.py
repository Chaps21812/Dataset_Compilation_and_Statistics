from src.utilities import search_image_id, get_folders_in_directory
from src.coco_tools import CalsatDataset
from src.raw_datset import raw_dataset
import os

CALSAT_A = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-ABQ01-2025"
CALSAT_B = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-LMNT01-2025"

for calsat in [CALSAT_A,CALSAT_B]:
    bruh2 = CalsatDataset(calsat)
    bruh2.return_files()

