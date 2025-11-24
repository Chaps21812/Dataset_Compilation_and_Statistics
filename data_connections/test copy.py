from src.raw_datset import raw_dataset
from src.coco_tools import COCODataset
from src.preprocess_functions import raw_file, iqr_log
from src.s3client import S3Client
import os

from src.raw_datset import raw_dataset
from src.coco_tools import COCODataset
import os

if __name__ == "__main__":
    # test_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025-Annotations-Errors/2024-08-28"
    # new_calsat_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-RME04-2025"

    # telescope_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations-Errors"

    # telescope_basename = os.path.basename(telescope_directory)
    # for date_folder in os.listdir(telescope_directory):
    #     subfolder = os.path.join(telescope_directory, date_folder)
    #     if os.path.isdir(subfolder):
    #         dataset = raw_dataset(subfolder)
    #         dataset.reinitialize_raw_dataset()
    #         dataset.correct_annotations()
    #         dataset.create_calsat_dataset(new_calsat_directory, move_mode="move", percentage_limit=0.10)
    #         coco_dataset = COCODataset(subfolder)
    #         coco_dataset.clear_extraneous_cache()
    #         coco_dataset.build_annotations()
    #         coco_dataset.generate_TTV_split(train_ratio=0.89,val_ratio=0.11,test_ratio=0)
    #         coco_dataset.move_fits_to_train_test_split()


    dataset = raw_dataset("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsats-RME04-2024").complete_calsat_dataset()
    calsat_dir= "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets"

    for date_folder in os.listdir(calsat_dir):
        subfolder = os.path.join(calsat_dir, date_folder)
        if os.path.isdir(subfolder):
            dataset = raw_dataset(subfolder).complete_calsat_dataset()
