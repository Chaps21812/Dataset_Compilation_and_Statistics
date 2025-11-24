
from src.raw_datset import raw_dataset
from src.coco_tools import COCODataset
import os

TELESCOPE_A = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025-Annotations"
TELESCOPE_B = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations"
TELESCOPE_C = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT02-2025_Annotations"
TELESCOPE_D = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01-2025_Annotations"
TELESCOPE_E = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/ABQ01-2025-Annotations"
calsat_dir = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets"
calsat_str = "Calsat_Final"

ALL_TELESCOPES = [TELESCOPE_A,TELESCOPE_B,TELESCOPE_C,TELESCOPE_D,TELESCOPE_E] 

for telescope_directory in [TELESCOPE_E]:
    telescope_basename = os.path.basename(telescope_directory)
    new_calsat_dir = os.path.join(calsat_dir,f"{calsat_str}-{telescope_basename.replace("-Annotations","")}")
    for date_folder in os.listdir(telescope_directory):
        subfolder = os.path.join(telescope_directory, date_folder)
        if os.path.isdir(subfolder):
            dataset = raw_dataset(subfolder)
            dataset.recalculate_statistics()
            dataset.correct_annotations()
            dataset.create_calsat_dataset(new_calsat_dir, move_mode="copy", percentage_limit=0.20)
            coco_dataset = COCODataset(subfolder)
            coco_dataset.clear_extraneous_cache()
            coco_dataset.build_annotations()
            coco_dataset.generate_TTV_split(train_ratio=0.89,val_ratio=0.11,test_ratio=0)
            coco_dataset.move_fits_to_train_test_split()
    raw_dataset(new_calsat_dir).complete_calsat_dataset()


